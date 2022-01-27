import argparse
import copy
import logging
from typing import Callable
from typing import Collection
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import torch
from typeguard import check_argument_types
from typeguard import check_return_type

from espnet2.asr.ctc import CTC
from espnet2.asr.espnet_enh_asr_model import ESPnetEnhASRModel
from espnet2.asr.espnet_model import ESPnetASRModel
from espnet2.asr.transducer.joint_network import JointNetwork
from espnet2.enh.espnet_model import ESPnetEnhancementModel
from espnet2.tasks.abs_task import AbsTask
from espnet2.tasks.asr import ASRTask
from espnet2.tasks.asr import frontend_choices
from espnet2.tasks.asr import specaug_choices
from espnet2.tasks.asr import normalize_choices
from espnet2.tasks.asr import preencoder_choices as asr_preencoder_choices_
from espnet2.tasks.asr import encoder_choices as asr_encoder_choices_
from espnet2.tasks.asr import postencoder_choices as asr_postencoder_choices_
from espnet2.tasks.asr import decoder_choices as asr_decoder_choices_
from espnet2.tasks.enh import EnhancementTask
from espnet2.tasks.enh import encoder_choices as enh_encoder_choices_
from espnet2.tasks.enh import decoder_choices as enh_decoder_choices_
from espnet2.tasks.enh import separator_choices as enh_separator_choices_
from espnet2.tasks.enh import criterion_choices as enh_criterion_choices_
from espnet2.text.phoneme_tokenizer import g2p_choices
from espnet2.torch_utils.initialize import initialize
from espnet2.train.class_choices import ClassChoices
from espnet2.train.collate_fn import CommonCollateFn
from espnet2.train.preprocessor import CommonPreprocessor_multi
from espnet2.train.trainer import Trainer
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.types import int_or_none
from espnet2.utils.types import str2bool
from espnet2.utils.types import str_or_none


enh_encoder_choices = copy.deepcopy(enh_encoder_choices_)
enh_encoder_choices.name = "enh_encoder"
enh_decoder_choices = copy.deepcopy(enh_decoder_choices_)
enh_decoder_choices.name = "enh_decoder"
enh_separator_choices = copy.deepcopy(enh_separator_choices_)
enh_separator_choices.name = "enh_separator"
enh_criterions_choices = copy.deepcopy(enh_criterion_choices_)
enh_criterions_choices.name = "enh_criterions"

asr_preencoder_choices = copy.deepcopy(asr_preencoder_choices_)
asr_preencoder_choices.name = "asr_preencoder"
asr_encoder_choices = copy.deepcopy(asr_encoder_choices_)
asr_encoder_choices.name = "asr_encoder"
asr_postencoder_choices = copy.deepcopy(asr_postencoder_choices_)
asr_postencoder_choices.name = "asr_postencoder"
asr_decoder_choices = copy.deepcopy(asr_decoder_choices_)
asr_decoder_choices.name = "asr_decoder"

MAX_REFERENCE_NUM = 100


class EnhASRTask(AbsTask):
    # If you need more than one optimizers, change this value
    num_optimizers: int = 1

    # Add variable objects configurations
    class_choices_list = [
        # --enh_encoder and --enh_encoder_conf
        enh_encoder_choices,
        # --enh_separator and --enh_separator_conf
        enh_separator_choices,
        # --enh_decoder and --enh_decoder_conf
        enh_decoder_choices,
        # --enh_criterion and --enh_criterion_conf
        enh_criterions_choices,
        # --frontend and --frontend_conf
        frontend_choices,
        # --specaug and --specaug_conf
        specaug_choices,
        # --normalize and --normalize_conf
        normalize_choices,
        # --asr_preencoder and --asr_preencoder_conf
        asr_preencoder_choices,
        # --asr_encoder and --asr_encoder_conf
        asr_encoder_choices,
        # --asr_postencoder and --asr_postencoder_conf
        asr_postencoder_choices,
        # --asr_decoder and --asr_decoder_conf
        asr_decoder_choices,
    ]

    # If you need to modify train() or eval() procedures, change Trainer class here
    trainer = Trainer

    @classmethod
    def add_task_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(description="Task related")

        # NOTE(kamo): add_arguments(..., required=True) can't be used
        # to provide --print_config mode. Instead of it, do as
        required = parser.get_default("required")
        required += ["token_list"]

        group.add_argument(
            "--token_list",
            type=str_or_none,
            default=None,
            help="A text mapping int-id to token",
        )
        group.add_argument(
            "--init",
            type=lambda x: str_or_none(x.lower()),
            default=None,
            help="The initialization method",
            choices=[
                "chainer",
                "xavier_uniform",
                "xavier_normal",
                "kaiming_uniform",
                "kaiming_normal",
                None,
            ],
        )

        group.add_argument(
            "--input_size",
            type=int_or_none,
            default=None,
            help="The number of input dimension of the feature",
        )

        group.add_argument(
            "--ctc_conf",
            action=NestedDictAction,
            default=get_default_kwargs(CTC),
            help="The keyword arguments for CTC class.",
        )
        group.add_argument(
            "--asr_model_conf",
            action=NestedDictAction,
            default=get_default_kwargs(ESPnetASRModel),
            help="The keyword arguments for model class.",
        )

        group.add_argument(
            "--enh_model_conf",
            action=NestedDictAction,
            default=get_default_kwargs(ESPnetEnhancementModel),
            help="The keyword arguments for model class.",
        )

        group.add_argument(
            "--model_conf",
            action=NestedDictAction,
            default=get_default_kwargs(ESPnetEnhASRModel),
            help="The keyword arguments for model class.",
        )

        group = parser.add_argument_group(description="Preprocess related")
        group.add_argument(
            "--use_preprocessor",
            type=str2bool,
            default=False,
            help="Apply preprocessing to data or not",
        )
        group.add_argument(
            "--token_type",
            type=str,
            default="bpe",
            choices=["bpe", "char", "word", "phn"],
            help="The text will be tokenized " "in the specified level token",
        )
        group.add_argument(
            "--bpemodel",
            type=str_or_none,
            default=None,
            help="The model file of sentencepiece",
        )
        parser.add_argument(
            "--non_linguistic_symbols",
            type=str_or_none,
            help="non_linguistic_symbols file path",
        )
        parser.add_argument(
            "--cleaner",
            type=str_or_none,
            choices=[None, "tacotron", "jaconv", "vietnamese"],
            default=None,
            help="Apply text cleaning",
        )
        parser.add_argument(
            "--g2p",
            type=str_or_none,
            choices=g2p_choices,
            default=None,
            help="Specify g2p method if --token_type=phn",
        )
        group.add_argument(
            "--enhancement_conf",
            action=NestedDictAction,
            default=None,
            help="The keyword arguments for enhancement model class.",
        )

        for class_choices in cls.class_choices_list:
            # Append --<name> and --<name>_conf.
            # e.g. --encoder and --encoder_conf
            class_choices.add_arguments(group)

    @classmethod
    def build_collate_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Callable[
        [Collection[Tuple[str, Dict[str, np.ndarray]]]],
        Tuple[List[str], Dict[str, torch.Tensor]],
    ]:
        assert check_argument_types()
        # NOTE(kamo): int value = 0 is reserved by CTC-blank symbol
        return CommonCollateFn(float_pad_value=0.0, int_pad_value=-1)

    @classmethod
    def build_preprocess_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
        assert check_argument_types()
        # TODO(Jing): ask Kamo if it ok to support several args,
        # like text_name = 'text_ref1' and 'text_ref2'
        if args.use_preprocessor:
            retval = CommonPreprocessor_multi(
                train=train,
                token_type=args.token_type,
                token_list=args.token_list,
                bpemodel=args.bpemodel,
                non_linguistic_symbols=args.non_linguistic_symbols,
                text_name=["text"],
                text_cleaner=args.cleaner,
                g2p_type=args.g2p,
            )
        else:
            retval = None
        assert check_return_type(retval)
        return retval

    @classmethod
    def required_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        if not inference:
            retval = ("speech", "speech_ref1", "text")
        else:
            # Recognition mode
            retval = ("speech",)
        return retval

    @classmethod
    def optional_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        retval = ["dereverb_ref"]
        retval += ["speech_ref{}".format(n) for n in range(2, MAX_REFERENCE_NUM + 1)]
        retval += ["text_ref{}".format(n) for n in range(2, MAX_REFERENCE_NUM + 1)]
        retval += ["noise_ref{}".format(n) for n in range(1, MAX_REFERENCE_NUM + 1)]
        retval = tuple(retval)
        assert check_return_type(retval)
        return retval

    @classmethod
    def build_model(cls, args: argparse.Namespace) -> ESPnetEnhASRModel:
        assert check_argument_types()
        if isinstance(args.token_list, str):
            with open(args.token_list, encoding="utf-8") as f:
                token_list = [line.rstrip() for line in f]

            # Overwriting token_list to keep it as "portable".
            args.token_list = list(token_list)
        elif isinstance(args.token_list, (tuple, list)):
            token_list = list(args.token_list)
        else:
            raise RuntimeError("token_list must be str or list")
        vocab_size = len(token_list)
        logging.info(f"Vocabulary size: {vocab_size }")

        # 0. Build enhancement model
        enh_conf = dict(init=None, model_conf=args.enh_model_conf)
        enh_attributes = [
            "encoder", "encoder_conf",
            "separator", "separator_conf",
            "decoder", "decoder_conf",
            "criterions",
        ]
        for attr in enh_attributes:
            enh_conf[attr] = (
                getattr(args, "enh_" + attr, None) if getattr(args, "enh_" + attr, None) is not None
                else getattr(args, attr, None)
            )
        enh_model = EnhancementTask.build_model(argparse.Namespace(**enh_conf))

        # 1. Build asr model
        asr_conf = dict(init=None, model_conf=args.asr_model_conf)
        asr_attributes = [
            "token_list",
            "input_size",
            "frontend", "frontend_conf",
            "specaug", "specaug_conf",
            "normalize", "normalize_conf",
            "preencoder", "preencoder_conf",
            "encoder", "encoder_conf",
            "postencoder", "postencoder_conf",
            "decoder", "decoder_conf",
            "ctc_conf",
        ]
        for attr in asr_attributes:
            asr_conf[attr] = (
                getattr(args, "asr_" + attr, None) if getattr(args, "asr_" + attr, None) is not None
                else getattr(args, attr, None)
            )
        asr_model = ASRTask.build_model(argparse.Namespace(**asr_conf))


        # 8. Build model
        model = ESPnetEnhASRModel(
            enh_model=enh_model,
            asr_model=asr_model,
            **args.model_conf,
        )

        # FIXME(kamo): Should be done in model?
        # 9. Initialize
        if args.init is not None:
            initialize(model, args.init)

        assert check_return_type(model)
        return model
