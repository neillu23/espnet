import argparse
from typing import Callable, Collection, Dict, List, Optional, Tuple

import numpy as np
import torch
from typeguard import check_argument_types, check_return_type

from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.frontend.asteroid_frontend import AsteroidFrontend
from espnet2.asr.frontend.default import DefaultFrontend
from espnet2.asr.frontend.fused import FusedFrontends
from espnet2.asr.frontend.melspec_torch import MelSpectrogramTorch
from espnet2.asr.frontend.s3prl import S3prlFrontend
from espnet2.asr.frontend.windowing import SlidingWindow
from espnet2.asr.frontend.s3prl import S3prlFrontend
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.asr.specaug.specaug import SpecAug
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.layers.global_mvn import GlobalMVN
from espnet2.layers.utterance_mvn import UtteranceMVN
from espnet2.spk.encoder.ecapa_tdnn_encoder import EcapaTdnnEncoder
from espnet2.spk.encoder.rawnet3_encoder import RawNet3Encoder
from espnet2.spk.espnet_model import ESPnetSpeakerModel
from espnet2.spk.hier_sv_espnet_model import ESPnetHierSVModel
from espnet2.spk.loss.aamsoftmax import AAMSoftmax
from espnet2.spk.loss.aamsoftmax_subcenter_intertopk import (
    ArcMarginProduct_intertopk_subcenter,
)
from espnet2.spk.loss.abs_loss import AbsLoss
from espnet2.spk.pooling.abs_pooling import AbsPooling
from espnet2.spk.pooling.chn_attn_stat_pooling import ChnAttnStatPooling
from espnet2.spk.projector.abs_projector import AbsProjector
from espnet2.spk.projector.rawnet3_projector import RawNet3Projector
from espnet2.tasks.abs_task import AbsTask
from espnet2.torch_utils.initialize import initialize
from espnet2.train.class_choices import ClassChoices
from espnet2.train.collate_fn import CommonCollateFn
from espnet2.train.preprocessor import (
    AbsPreprocessor,
    CommonPreprocessor,
    SpkPreprocessor,
)
from espnet2.train.spk_trainer import SpkTrainer as Trainer
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.types import int_or_none, str2bool, str_or_none




from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.preencoder.linear import LinearProjection
from espnet2.asr.preencoder.sinc import LightweightSincConvs
from espnet2.asr.encoder.resnet_encoder import ResNetEncoder
from espnet2.asr.postencoder.abs_postencoder import AbsPostEncoder
from espnet2.train.abs_espnet_model import AbsESPnetModel



# Check and understand
frontend_choices = ClassChoices(
    name="frontend",
    classes=dict(
        asteroid_frontend=AsteroidFrontend,
        default=DefaultFrontend,
        fused=FusedFrontends,
        melspec_torch=MelSpectrogramTorch,
        sliding_window=SlidingWindow,
        raw=AbsFrontend,
        s3prl=S3prlFrontend,
    ),
    type_check=AbsFrontend,
    default=None,
    optional=True,
)

specaug_choices = ClassChoices(
    name="specaug",
    classes=dict(specaug=SpecAug),
    type_check=AbsSpecAug,
    default=None,
    optional=True,
)

normalize_choices = ClassChoices(
    name="normalize",
    classes=dict(
        global_mvn=GlobalMVN,
        utterance_mvn=UtteranceMVN,
    ),
    type_check=AbsNormalize,
    default=None,
    optional=True,
)

encoder_choices = ClassChoices(
    name="encoder",
    classes=dict(
        rawnet3=RawNet3Encoder,
        ecapa_tdnn=EcapaTdnnEncoder,
    ),
    type_check=AbsEncoder,
    default="rawnet3",
)

pooling_choices = ClassChoices(
    name="pooling",
    classes=dict(
        chn_attn_stat=ChnAttnStatPooling,
    ),
    type_check=AbsPooling,
    default="chn_attn_stat",
)

projector_choices = ClassChoices(
    name="projector",
    classes=dict(
        # TODO (Jee-weon): implement additional Projectors
        # one_layer=OneLayerProjector,
        rawnet3=RawNet3Projector,
    ),
    type_check=AbsProjector,
    default="rawnet3",
)


preprocessor_choices = ClassChoices(
    name="preprocessor",
    classes=dict(
        common=CommonPreprocessor,
        spk=SpkPreprocessor,
    ),
    type_check=AbsPreprocessor,
    default="spk",
)

loss_choices = ClassChoices(
    name="loss",
    classes=dict(
        aamsoftmax=AAMSoftmax,
        aamsoftmax_sc_topk=ArcMarginProduct_intertopk_subcenter,
    ),
    default="aam",
)



model_choices = ClassChoices(
    "model",
    classes=dict(
        espnet=ESPnetSpeakerModel,
        joint_hier_espnet=ESPnetHierSVModel,
    ),
    type_check=AbsESPnetModel,
    default="espnet",
)

preencoder_choices = ClassChoices(
    name="preencoder",
    classes=dict(
        sinc=LightweightSincConvs,
        linear=LinearProjection,
    ),
    type_check=AbsPreEncoder,
    default=None,
    optional=True,
)

preencoder_lid_choices = ClassChoices(
    name="preencoder_lid",
    classes=dict(
        sinc=LightweightSincConvs,
        linear=LinearProjection,
    ),
    type_check=AbsPreEncoder,
    default=None,
    optional=True,
)


encoder_lid_choices = ClassChoices(
    "encoder_lid",
    classes=dict(
        resnet_enc=ResNetEncoder,
    ),
    type_check=AbsEncoder,
    default="rnn",
)

postencoder_lid_choices = ClassChoices(
    name="postencoder_lid",
    classes=dict(
        chn_attn_stat=ChnAttnStatPooling,  # for LID
    ),
    type_check=AbsPooling,
    default=None,
    optional=True,
)



projector_lid_choices = ClassChoices(
    name="projector_lid",
    classes=dict(
        # TODO (Jee-weon): implement additional Projectors
        # one_layer=OneLayerProjector,
        rawnet3=RawNet3Projector,
    ),
    type_check=AbsProjector,
    default="rawnet3",
)


class SpeakerTask(AbsTask):
    num_optimizers: int = 1

    class_choices_list = [
        frontend_choices,
        specaug_choices,
        normalize_choices,
        encoder_choices,
        pooling_choices,
        projector_choices,
        preprocessor_choices,
        loss_choices,
        preencoder_choices,
        preencoder_lid_choices,
        encoder_lid_choices,
        postencoder_lid_choices,
        projector_lid_choices,
        model_choices,
    ]

    trainer = Trainer

    @classmethod
    def add_task_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(description="Task related")

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
            "--use_preprocessor",
            type=str2bool,
            default=True,
            help="Apply preprocessing to data or not",
        )

        group.add_argument(
            "--lid_tokens",
            type=str_or_none,
            default=None,
            help="A text mapping int-id to lid token",
        )
        

        group.add_argument(
            "--input_size",
            type=int_or_none,
            default=None,
            help="The number of input dimension of the feature",
        )

        group.add_argument(
            "--target_duration",
            type=float,
            default=3.0,
            help="Duration (in seconds) of samples in a minibatch",
        )

        group.add_argument(
            "--spk2utt",
            type=str,
            default="",
            help="Directory of spk2utt file to be used in label mapping",
        )

        group.add_argument(
            "--spk_num",
            type=int,
            default=None,
            help="specify the number of speakers during training",
        )

        group.add_argument(
            "--sample_rate",
            type=int,
            default=16000,
            help="Sampling rate",
        )

        group.add_argument(
            "--num_eval",
            type=int,
            default=10,
            help="Number of segments to make from one utterance in the "
            "inference phase",
        )

        group.add_argument(
            "--rir_scp",
            type=str,
            default="",
            help="Directory of the rir data to be augmented",
        )

        # group.add_argument(
        #     "--model_conf",
        #     action=NestedDictAction,
        #     default=get_default_kwargs(ESPnetSpeakerModel),
        #     help="The keyword arguments for model class.",
        # )

        for class_choices in cls.class_choices_list:
            class_choices.add_arguments(group)

    @classmethod
    def build_collate_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Callable[
        [Collection[Tuple[str, Dict[str, np.ndarray]]]],
        Tuple[List[str], Dict[str, torch.Tensor]],
    ]:
        assert check_argument_types()
        return CommonCollateFn()

    @classmethod
    def build_preprocess_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
        assert check_argument_types()
        if args.use_preprocessor:
            if train:
                retval = preprocessor_choices.get_class(args.preprocessor)(
                    spk2utt=args.spk2utt,
                    train=train,
                    **args.preprocessor_conf,
                )
            else:
                retval = preprocessor_choices.get_class(args.preprocessor)(
                    train=train,
                    **args.preprocessor_conf,
                )

        else:
            retval = None
        assert check_return_type(retval)
        return retval

    @classmethod
    def required_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        if train:
            retval = ("speech", "spk_labels")
        else:
            # Recognition mode
            retval = ("speech",)
        return retval

    @classmethod
    def optional_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        # When calculating EER, we need trials where each trial has two
        # utterances. speech2 corresponds to the second utterance of each
        # trial pair in the validation/inference phase.
        retval = ("speech2", "trial", "spk_labels", "task_tokens", "langs", "langs2")

        assert check_return_type(retval)
        return retval

    @classmethod
    def build_model(cls, args: argparse.Namespace) -> ESPnetSpeakerModel:
        assert check_argument_types()

        if args.frontend is not None:
            frontend_class = frontend_choices.get_class(args.frontend)
            frontend = frontend_class(**args.frontend_conf)
            input_size = frontend.output_size()
        else:
            # Give features from data-loader (e.g., precompute features).
            frontend = None
            input_size = args.input_size

        if args.specaug is not None:
            specaug_class = specaug_choices.get_class(args.specaug)
            specaug = specaug_class(**args.specaug_conf)
        else:
            specaug = None

        if args.normalize is not None:
            normalize_class = normalize_choices.get_class(args.normalize)
            normalize = normalize_class(**args.normalize_conf)
        else:
            normalize = None


        # for lid_tokens
        if isinstance(args.lid_tokens, str):
            with open(args.lid_tokens, encoding="utf-8") as f:
                lid_tokens = [line.rstrip() for line in f]

            # Overwriting lid_tokens to keep it as "portable".
            args.lid_tokens = list(lid_tokens)
            langs_num = len(lid_tokens)
        elif isinstance(args.lid_tokens, (tuple, list)):
            lid_tokens = list(args.lid_tokens)
            langs_num = len(lid_tokens)
        else:
            lid_tokens= None
            langs_num = 0

        if getattr(args, "preencoder", None) is not None:
            preencoder_class = preencoder_choices.get_class(args.preencoder)
            preencoder = preencoder_class(**args.preencoder_conf)
            input_size = preencoder.output_size()
        else:
            preencoder = None



        encoder_class = encoder_choices.get_class(args.encoder)
        encoder = encoder_class(input_size=input_size, **args.encoder_conf)
        encoder_output_size = encoder.output_size()

        pooling_class = pooling_choices.get_class(args.pooling)
        pooling = pooling_class(input_size=encoder_output_size, **args.pooling_conf)
        pooling_output_size = pooling.output_size()

        projector_class = projector_choices.get_class(args.projector)
        projector = projector_class(
            input_size=pooling_output_size, **args.projector_conf
        )
        projector_output_size = projector.output_size()

        loss_class = loss_choices.get_class(args.loss)
        loss = loss_class(
            nout=projector_output_size, nclasses=args.spk_num, **args.loss_conf
        )


        # 7. Build model
        try:
            model_class = model_choices.get_class(args.model)
        except AttributeError:
            model_class = model_choices.get_class("espnet")


        if model_class == ESPnetHierSVModel:
                
            preencoder_lid_nums = args.model_conf.get("preencoder_lid_nums", 1)
            if preencoder_lid_nums > 1:
                preencoder_lid_class = preencoder_lid_choices.get_class(args.preencoder_lid)
                preencoder_lid = torch.nn.ModuleList([preencoder_lid_class(**args.preencoder_lid_conf) for i in range(preencoder_lid_nums)])

                lid_input_size = preencoder_lid[0].output_size()
            else:
                if getattr(args, "preencoder_lid", None) is not None:
                    preencoder_lid_class = preencoder_lid_choices.get_class(args.preencoder_lid)
                    preencoder_lid = preencoder_lid_class(**args.preencoder_lid_conf)
                    lid_input_size = preencoder_lid.output_size()
                else:
                    preencoder_lid = None
                    if args.input_size is None:
                        lid_input_size = frontend.output_size()
                    else:
                        lid_input_size = args.input_size

            encoder_lid_class = encoder_lid_choices.get_class(args.encoder_lid)
            encoder_lid = encoder_lid_class(input_size=lid_input_size, **args.encoder_lid_conf)
            encoder_lid_output_size = encoder_lid.output_size()
            if getattr(args, "postencoder_lid", None) is not None:
                postencoder_lid_class = postencoder_lid_choices.get_class(args.postencoder_lid)
                postencoder_lid = postencoder_lid_class(
                    input_size=encoder_lid_output_size, **args.postencoder_lid_conf
                )
                encoder_lid_output_size = postencoder_lid.output_size()

            projector_lid_class = projector_lid_choices.get_class(args.projector_lid)
            projector_lid = projector_lid_class(**args.projector_lid_conf)

            model = ESPnetHierSVModel(
                frontend=frontend,
                specaug=specaug,
                normalize=normalize,
                preencoder=preencoder,
                encoder=encoder,
                pooling=pooling,
                projector=projector,
                loss=loss,
                lid_tokens=lid_tokens,
                langs_num=langs_num,
                preencoder_lid=preencoder_lid,
                encoder_lid=encoder_lid,
                postencoder_lid=postencoder_lid,
                projector_lid=projector_lid,
                **args.model_conf, # uncomment when model_conf exists
            )
        else:
            model = ESPnetSpeakerModel(
                frontend=frontend,
                specaug=specaug,
                normalize=normalize,
                preencoder=preencoder,
                encoder=encoder,
                pooling=pooling,
                projector=projector,
                loss=loss,
                lid_tokens=lid_tokens,
                langs_num=langs_num,
                **args.model_conf, # uncomment when model_conf exists
            )

        if args.init is not None:
            initialize(model, args.init)

        assert check_return_type(model)
        return model
