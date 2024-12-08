import argparse
import logging
from typing import Callable, Collection, Dict, List, Optional, Tuple

import numpy as np
import torch
from typeguard import typechecked

from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.decoder.hugging_face_transformers_decoder import (  # noqa: H301
    HuggingFaceTransformersDecoder,
)
from espnet2.asr.decoder.mlm_decoder import MLMDecoder
from espnet2.asr.decoder.rnn_decoder import RNNDecoder
from espnet2.asr.decoder.s4_decoder import S4Decoder
from espnet2.asr.decoder.transducer_decoder import TransducerDecoder
from espnet2.asr.decoder.transformer_decoder import (
    DynamicConvolution2DTransformerDecoder,
    DynamicConvolutionTransformerDecoder,
    LightweightConvolution2DTransformerDecoder,
    LightweightConvolutionTransformerDecoder,
    TransformerDecoder,
)
from espnet2.asr.decoder.whisper_decoder import OpenAIWhisperDecoder
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.encoder.avhubert_encoder import FairseqAVHubertEncoder
from espnet2.asr.encoder.branchformer_encoder import BranchformerEncoder
from espnet2.asr.encoder.conformer_encoder import ConformerEncoder
from espnet2.asr.encoder.contextual_block_conformer_encoder import (
    ContextualBlockConformerEncoder,
)
from espnet2.asr.encoder.contextual_block_transformer_encoder import (
    ContextualBlockTransformerEncoder,
)
from espnet2.asr.encoder.e_branchformer_encoder import EBranchformerEncoder
from espnet2.asr.encoder.hubert_encoder import (
    FairseqHubertEncoder,
    FairseqHubertPretrainEncoder,
    TorchAudioHuBERTPretrainEncoder,
)
from espnet2.asr.encoder.longformer_encoder import LongformerEncoder
from espnet2.asr.encoder.resnet_encoder import ResNetEncoder
from espnet2.asr.encoder.multiconvformer_encoder import MultiConvConformerEncoder
from espnet2.asr.encoder.rnn_encoder import RNNEncoder
from espnet2.asr.encoder.transformer_encoder import TransformerEncoder
from espnet2.asr.encoder.transformer_encoder_multispkr import (
    TransformerEncoder as TransformerEncoderMultiSpkr,
)
from espnet2.asr.encoder.vgg_rnn_encoder import VGGRNNEncoder
from espnet2.asr.encoder.wav2vec2_encoder import FairSeqWav2Vec2Encoder
from espnet2.asr.encoder.whisper_encoder import OpenAIWhisperEncoder
from espnet2.asr.espnet_model import ESPnetASRModel
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.frontend.default import DefaultFrontend
from espnet2.asr.frontend.fused import FusedFrontends
from espnet2.asr.frontend.s3prl import S3prlFrontend
from espnet2.asr.frontend.ca_sslr_s3prl import S3prlCASSLRFrontend
from espnet2.asr.frontend.whisper import WhisperFrontend
from espnet2.asr.frontend.windowing import SlidingWindow
from espnet2.asr.maskctc_model import MaskCTCModel
from espnet2.asr.pit_espnet_model import ESPnetASRModel as PITESPnetModel
from espnet2.asr.ca_sslr_espnet_model import ESPnetCASSLRModel

from espnet2.asr.postencoder.abs_postencoder import AbsPostEncoder
from espnet2.asr.postencoder.hugging_face_transformers_postencoder import (
    HuggingFaceTransformersPostEncoder,
)
from espnet2.asr.postencoder.length_adaptor_postencoder import LengthAdaptorPostEncoder
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.preencoder.linear import LinearProjection
from espnet2.asr.preencoder.sinc import LightweightSincConvs
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.asr.specaug.specaug import SpecAug
from espnet2.asr_transducer.joint_network import JointNetwork
from espnet2.spk.loss.aamsoftmax import AAMSoftmax
from espnet2.spk.loss.abs_loss import AbsLoss
from espnet2.spk.projector.abs_projector import AbsProjector
from espnet2.spk.projector.rawnet3_projector import RawNet3Projector
from espnet2.spk.pooling.abs_pooling import AbsPooling
from espnet2.spk.pooling.chn_attn_stat_pooling import ChnAttnStatPooling


from espnet2.spk.loss.aamsoftmax_subcenter_intertopk import (
    ArcMarginProduct_intertopk_subcenter,
)
from espnet2.spk.encoder.ecapa_tdnn_encoder import EcapaTdnnEncoder
from espnet2.spk.encoder.rawnet3_encoder import RawNet3Encoder

from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.layers.global_mvn import GlobalMVN
from espnet2.layers.utterance_mvn import UtteranceMVN
from espnet2.tasks.abs_task import AbsTask
from espnet2.text.phoneme_tokenizer import g2p_choices
from espnet2.torch_utils.initialize import initialize
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet2.train.class_choices import ClassChoices
from espnet2.train.collate_fn import CommonCollateFn
from espnet2.train.preprocessor import (
    AbsPreprocessor,
    CommonPreprocessor,
    CommonPreprocessor_multi,
    JointASRLIDPreprocessor,
    JointASRLIDSVPreprocessor,
)
# from espnet2.train.trainer import Trainer
from espnet2.train.joint_trainer import JointTrainer as Trainer
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.types import float_or_none, int_or_none, str2bool, str_or_none

frontend_choices = ClassChoices(
    name="frontend",
    classes=dict(
        default=DefaultFrontend,
        sliding_window=SlidingWindow,
        s3prl=S3prlFrontend,
        s3prl_casslr=S3prlCASSLRFrontend,
        fused=FusedFrontends,
        whisper=WhisperFrontend,
    ),
    type_check=AbsFrontend,
    default="default",
)
specaug_choices = ClassChoices(
    name="specaug",
    classes=dict(
        specaug=SpecAug,
    ),
    type_check=AbsSpecAug,
    default=None,
    optional=True,
)
normalize_choices = ClassChoices(
    "normalize",
    classes=dict(
        global_mvn=GlobalMVN,
        utterance_mvn=UtteranceMVN,
    ),
    type_check=AbsNormalize,
    default="utterance_mvn",
    optional=True,
)
model_choices = ClassChoices(
    "model",
    classes=dict(
        espnet=ESPnetASRModel,
        maskctc=MaskCTCModel,
        pit_espnet=PITESPnetModel,
        casslr_espnet=ESPnetCASSLRModel,
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


preencoder_spk_choices = ClassChoices(
    name="preencoder_spk",
    classes=dict(
        sinc=LightweightSincConvs,
        linear=LinearProjection,
    ),
    type_check=AbsPreEncoder,
    default=None,
    optional=True,
)

encoder_choices = ClassChoices(
    "encoder",
    classes=dict(
        conformer=ConformerEncoder,
        transformer=TransformerEncoder,
        transformer_multispkr=TransformerEncoderMultiSpkr,
        contextual_block_transformer=ContextualBlockTransformerEncoder,
        contextual_block_conformer=ContextualBlockConformerEncoder,
        vgg_rnn=VGGRNNEncoder,
        resnet_enc=ResNetEncoder,
        rnn=RNNEncoder,
        wav2vec2=FairSeqWav2Vec2Encoder,
        hubert=FairseqHubertEncoder,
        hubert_pretrain=FairseqHubertPretrainEncoder,
        torchaudiohubert=TorchAudioHuBERTPretrainEncoder,
        longformer=LongformerEncoder,
        branchformer=BranchformerEncoder,
        whisper=OpenAIWhisperEncoder,
        e_branchformer=EBranchformerEncoder,
        avhubert=FairseqAVHubertEncoder,
        multiconv_conformer=MultiConvConformerEncoder,
    ),
    type_check=AbsEncoder,
    default="rnn",
)

encoder_lid_choices = ClassChoices(
    "encoder_lid",
    classes=dict(
        resnet_enc=ResNetEncoder,
    ),
    type_check=AbsEncoder,
    default="rnn",
)

postencoder_choices = ClassChoices(
    name="postencoder",
    classes=dict(
        hugging_face_transformers=HuggingFaceTransformersPostEncoder,
        length_adaptor=LengthAdaptorPostEncoder,
    ),
    type_check=AbsPostEncoder,
    default=None,
    optional=True,
)

postencoder_lid_choices = ClassChoices(
    name="postencoder_lid",
    classes=dict(
        chn_attn_stat=ChnAttnStatPooling,
    ),
    type_check=AbsPooling, #AbsPostEncoder,
    default=None,
    optional=True,
)

decoder_choices = ClassChoices(
    "decoder",
    classes=dict(
        transformer=TransformerDecoder,
        lightweight_conv=LightweightConvolutionTransformerDecoder,
        lightweight_conv2d=LightweightConvolution2DTransformerDecoder,
        dynamic_conv=DynamicConvolutionTransformerDecoder,
        dynamic_conv2d=DynamicConvolution2DTransformerDecoder,
        rnn=RNNDecoder,
        transducer=TransducerDecoder,
        mlm=MLMDecoder,
        whisper=OpenAIWhisperDecoder,
        hugging_face_transformers=HuggingFaceTransformersDecoder,
        s4=S4Decoder,
    ),
    type_check=AbsDecoder,
    default=None,
    optional=True,
)
preprocessor_choices = ClassChoices(
    "preprocessor",
    classes=dict(
        default=CommonPreprocessor,
        multi=CommonPreprocessor_multi,
        joint_asr_lid=JointASRLIDPreprocessor,
        joint_asr_lid_spk=JointASRLIDSVPreprocessor,
    ),
    type_check=AbsPreprocessor,
    default="default",
)

projector_lid_choices = ClassChoices(
    name="projector_lid",
    classes=dict(
        rawnet3=RawNet3Projector,
    ),
    type_check=AbsProjector,
    default="rawnet3",
)

loss_lid_choices = ClassChoices(
    name="loss_lid",
    classes=dict(
        aamsoftmax=AAMSoftmax,
    ),
    type_check=AbsLoss,
    default="aam",
)

 
encoder_spk_choices = ClassChoices(
    name="encoder_spk",
    classes=dict(
        rawnet3=RawNet3Encoder,
        ecapa_tdnn=EcapaTdnnEncoder,
    ),
    type_check=AbsEncoder,
    default=None,
    optional=True,
)


pooling_spk_choices = ClassChoices(
    name="pooling_spk",
    classes=dict(
        chn_attn_stat=ChnAttnStatPooling,
    ),
    type_check=AbsPooling,
    # default="chn_attn_stat",
    default=None,
    optional=True,
)

projector_spk_choices = ClassChoices(
    name="projector_spk",
    classes=dict(
        # TODO (Jee-weon): implement additional Projectors
        # one_layer=OneLayerProjector,
        rawnet3=RawNet3Projector,
    ),
    type_check=AbsProjector,
    default=None,
    optional=True,
)


loss_spk_choices = ClassChoices(
    name="loss_spk",
    classes=dict(
        aamsoftmax=AAMSoftmax,
        aamsoftmax_sc_topk=ArcMarginProduct_intertopk_subcenter,
    ),
    default=None,
    optional=True,
)





class ASRTask(AbsTask):
    # If you need more than one optimizers, change this value
    num_optimizers: int = 1

    # Add variable objects configurations
    class_choices_list = [
        # --frontend and --frontend_conf
        frontend_choices,
        # --specaug and --specaug_conf
        specaug_choices,
        # --normalize and --normalize_conf
        normalize_choices,
        # --model and --model_conf
        model_choices,
        # --preencoder and --preencoder_conf
        preencoder_choices,
        # --preencoder_lid and --preencoder_lid_conf
        preencoder_lid_choices,
        # --preencoder_spk and --preencoder_spk_conf
        preencoder_spk_choices,
        # --encoder and --encoder_conf
        encoder_choices,
        # # --encoder_lid and --encoder_lid_conf
        encoder_lid_choices,
        # --postencoder and --postencoder_conf
        postencoder_choices,
        # # --postencoder_lid and --postencoder_lid_conf
        postencoder_lid_choices,
        # --decoder and --decoder_conf
        decoder_choices,
        # --preprocessor and --preprocessor_conf
        preprocessor_choices,
        # --projector_lid and --projector_lid_conf
        projector_lid_choices,
        # --loss_lid and --loss_lid_conf
        loss_lid_choices,
        # --encoder_spk and --encoder_spk_conf
        encoder_spk_choices,
        # --pooling_spk and --pooling_spk_conf
        pooling_spk_choices,
        # --projector_spk and --projector_spk_conf
        projector_spk_choices,
        # --loss_spk and --loss_spk_conf
        loss_spk_choices,


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
            "--lid_tokens",
            type=str_or_none,
            default=None,
            help="A text mapping int-id to lid token",
        )
        
        group.add_argument(
            "--spk2utt",
            type=str_or_none,
            default=None,
            help="Directory of spk2utt file to be used in label mapping",
        )


        group.add_argument(
            "--spk_num",
            type=int,
            default=None,
            help="specify the number of speakers during training",
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
            "--joint_net_conf",
            action=NestedDictAction,
            default=None,
            help="The keyword arguments for joint network class.",
        )

        group = parser.add_argument_group(description="Preprocess related")
        group.add_argument(
            "--use_preprocessor",
            type=str2bool,
            default=True,
            help="Apply preprocessing to data or not",
        )
        group.add_argument(
            "--use_lang_prompt",
            type=str2bool,
            default=False,
            help="Use language id as prompt",
        )
        group.add_argument(
            "--use_nlp_prompt",
            type=str2bool,
            default=False,
            help="Use natural language phrases as prompt",
        )
        group.add_argument(
            "--token_type",
            type=str,
            default="bpe",
            choices=[
                "bpe",
                "char",
                "word",
                "phn",
                "hugging_face",
                "whisper_en",
                "whisper_multilingual",
            ],
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
        group.add_argument(
            "--cleaner",
            type=str_or_none,
            choices=[
                None,
                "tacotron",
                "jaconv",
                "vietnamese",
                "whisper_en",
                "whisper_basic",
            ],
            default=None,
            help="Apply text cleaning",
        )
        group.add_argument(
            "--g2p",
            type=str_or_none,
            choices=g2p_choices,
            default=None,
            help="Specify g2p method if --token_type=phn",
        )
        group.add_argument(
            "--speech_volume_normalize",
            type=float_or_none,
            default=None,
            help="Scale the maximum amplitude to the given value.",
        )
        group.add_argument(
            "--rir_scp",
            type=str_or_none,
            default=None,
            help="The file path of rir scp file.",
        )
        group.add_argument(
            "--rir_apply_prob",
            type=float,
            default=1.0,
            help="THe probability for applying RIR convolution.",
        )
        group.add_argument(
            "--noise_scp",
            type=str_or_none,
            default=None,
            help="The file path of noise scp file.",
        )
        group.add_argument(
            "--noise_apply_prob",
            type=float,
            default=1.0,
            help="The probability applying Noise adding.",
        )
        group.add_argument(
            "--noise_db_range",
            type=str,
            default="13_15",
            help="The range of noise decibel level.",
        )
        group.add_argument(
            "--short_noise_thres",
            type=float,
            default=0.5,
            help="If len(noise) / len(speech) is smaller than this threshold during "
            "dynamic mixing, a warning will be displayed.",
        )
        group.add_argument(
            "--aux_ctc_tasks",
            type=str,
            nargs="+",
            default=[],
            help="Auxillary tasks to train on using CTC loss. ",
        )

        group.add_argument(
            "--pretrained_models",
            type=str,
            nargs="+",
            default=[],

        )

        for class_choices in cls.class_choices_list:
            # Append --<name> and --<name>_conf.
            # e.g. --encoder and --encoder_conf
            class_choices.add_arguments(group)

    @classmethod
    @typechecked
    def build_collate_fn(cls, args: argparse.Namespace, train: bool) -> Callable[
        [Collection[Tuple[str, Dict[str, np.ndarray]]]],
        Tuple[List[str], Dict[str, torch.Tensor]],
    ]:
        # NOTE(kamo): int value = 0 is reserved by CTC-blank symbol
        return CommonCollateFn(float_pad_value=0.0, int_pad_value=-1)

    @classmethod
    @typechecked
    def build_preprocess_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
        if args.use_preprocessor:
            try:
                _ = getattr(args, "preprocessor")
            except AttributeError:
                setattr(args, "preprocessor", "default")
                setattr(args, "preprocessor_conf", dict())
            except Exception as e:
                raise e

            preprocessor_class = preprocessor_choices.get_class(args.preprocessor)
            retval = preprocessor_class(
                train=train,
                token_type=args.token_type,
                token_list=args.token_list,
                bpemodel=args.bpemodel,
                non_linguistic_symbols=args.non_linguistic_symbols,
                text_cleaner=args.cleaner,
                g2p_type=args.g2p,
                # NOTE(kamo): Check attribute existence for backward compatibility
                rir_scp=args.rir_scp if hasattr(args, "rir_scp") else None,
                rir_apply_prob=(
                    args.rir_apply_prob if hasattr(args, "rir_apply_prob") else 1.0
                ),
                noise_scp=args.noise_scp if hasattr(args, "noise_scp") else None,
                noise_apply_prob=(
                    args.noise_apply_prob if hasattr(args, "noise_apply_prob") else 1.0
                ),
                noise_db_range=(
                    args.noise_db_range if hasattr(args, "noise_db_range") else "13_15"
                ),
                short_noise_thres=(
                    args.short_noise_thres
                    if hasattr(args, "short_noise_thres")
                    else 0.5
                ),
                speech_volume_normalize=(
                    args.speech_volume_normalize if hasattr(args, "rir_scp") else None
                ),
                aux_task_names=(
                    args.aux_ctc_tasks if hasattr(args, "aux_ctc_tasks") else None
                ),
                use_lang_prompt=(
                    args.use_lang_prompt if hasattr(args, "use_lang_prompt") else None
                ),
                **args.preprocessor_conf,
                use_nlp_prompt=(
                    args.use_nlp_prompt if hasattr(args, "use_nlp_prompt") else None
                ),
                spk2utt=(
                    args.spk2utt if hasattr(args, "spk2utt") else None
                ),
            )
        else:
            retval = None
        return retval

    @classmethod
    def required_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        if not inference:
            retval = ("speech", "text")
        else:
            # Recognition mode
            retval = ("speech",)
        return retval

    @classmethod
    def optional_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        MAX_REFERENCE_NUM = 4

        retval = ["text_spk{}".format(n) for n in range(2, MAX_REFERENCE_NUM + 1)]
        retval.append("langs")
        retval.append("spk_labels")
        retval = retval + ["prompt"]
        retval = tuple(retval)

        logging.info(f"Optional Data Names: {retval }")
        return retval

    @classmethod
    @typechecked
    def build_model(cls, args: argparse.Namespace) -> ESPnetASRModel:
        if isinstance(args.token_list, str):
            with open(args.token_list, encoding="utf-8") as f:
                token_list = [line.rstrip() for line in f]

            # Overwriting token_list to keep it as "portable".
            args.token_list = list(token_list)
        elif isinstance(args.token_list, (tuple, list)):
            token_list = list(args.token_list)
        else:
            raise RuntimeError("token_list must be str or list")
        
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

        # If use multi-blank transducer criterion,
        # big blank symbols are added just before the standard blank
        if args.model_conf.get("transducer_multi_blank_durations", None) is not None:
            sym_blank = args.model_conf.get("sym_blank", "<blank>")
            blank_idx = token_list.index(sym_blank)
            for dur in args.model_conf.get("transducer_multi_blank_durations"):
                if f"<blank{dur}>" not in token_list:  # avoid this during inference
                    token_list.insert(blank_idx, f"<blank{dur}>")
            args.token_list = token_list

        vocab_size = len(token_list)
        logging.info(f"Vocabulary size: {vocab_size }")

        # 1. frontend
        if args.input_size is None:
            # Extract features in the model
            frontend_class = frontend_choices.get_class(args.frontend)
            if "extra_conf" in args.frontend_conf:
                args.frontend_conf["frontend_conf"]["extra_conf"]["langs_num"] = langs_num
            frontend = frontend_class(**args.frontend_conf)
            input_size = frontend.output_size()
        else:
            # Give features from data-loader
            args.frontend = None
            args.frontend_conf = {}
            frontend = None
            input_size = args.input_size

        # 2. Data augmentation for spectrogram
        if args.specaug is not None:
            specaug_class = specaug_choices.get_class(args.specaug)
            specaug = specaug_class(**args.specaug_conf)
        else:
            specaug = None

        # 3. Normalization layer
        if args.normalize is not None:
            normalize_class = normalize_choices.get_class(args.normalize)
            normalize = normalize_class(**args.normalize_conf)
        else:
            normalize = None

        # 4. Pre-encoder input block
        # NOTE(kan-bayashi): Use getattr to keep the compatibility
        if getattr(args, "preencoder", None) is not None:
            preencoder_class = preencoder_choices.get_class(args.preencoder)
            preencoder = preencoder_class(**args.preencoder_conf)
            input_size = preencoder.output_size()
        else:
            preencoder = None

        # 4. Encoder
        encoder_class = encoder_choices.get_class(args.encoder)
        encoder = encoder_class(input_size=input_size, **args.encoder_conf)

        # 5. Post-encoder block
        # NOTE(kan-bayashi): Use getattr to keep the compatibility
        encoder_output_size = encoder.output_size()
        if getattr(args, "postencoder", None) is not None:
            postencoder_class = postencoder_choices.get_class(args.postencoder)
            postencoder = postencoder_class(
                input_size=encoder_output_size, **args.postencoder_conf
            )
            encoder_output_size = postencoder.output_size()
        else:
            postencoder = None

        # 5. Decoder
        if getattr(args, "decoder", None) is not None:
            decoder_class = decoder_choices.get_class(args.decoder)

            if args.decoder == "transducer":
                decoder = decoder_class(
                    vocab_size,
                    embed_pad=0,
                    **args.decoder_conf,
                )

                joint_network = JointNetwork(
                    vocab_size,
                    encoder.output_size(),
                    decoder.dunits,
                    **args.joint_net_conf,
                )
            else:
                decoder = decoder_class(
                    vocab_size=vocab_size,
                    encoder_output_size=encoder_output_size,
                    **args.decoder_conf,
                )
                joint_network = None
        else:
            decoder = None
            joint_network = None

        # 6. LID modules
        if getattr(args, "preencoder_lid", None) is not None:
            lid_layer_selections = args.frontend_conf.get("lid_layer_selections", None)
            preencoder_lid_nums = len(lid_layer_selections) if lid_layer_selections is not None else 1
            
            if preencoder_lid_nums > 1:
                preencoder_lid_class = preencoder_choices.get_class(args.preencoder_lid)
                preencoder_lid = torch.nn.ModuleList([preencoder_lid_class(**args.preencoder_lid_conf) for i in range(preencoder_lid_nums)])
                lid_input_size = preencoder_lid[0].output_size()
            else:
                preencoder_lid_class = preencoder_choices.get_class(args.preencoder_lid)
                preencoder_lid = preencoder_lid_class(**args.preencoder_lid_conf)
                lid_input_size = preencoder_lid.output_size()
        else:
            preencoder_lid = None
            if args.input_size is None:
                lid_input_size = frontend.output_size()
            else:
                lid_input_size = args.input_size

        if getattr(args, "encoder_lid", None) is not None:
            if args.model_conf.get("separate_lid_modules", False):
                encoder_lid_class = encoder_lid_choices.get_class(args.encoder_lid)
                encoder_lid = torch.nn.ModuleList([encoder_lid_class(input_size=lid_input_size, **args.encoder_lid_conf) for i in range(2)])
                encoder_lid_output_size = encoder_lid[0].output_size()
            else: 
                encoder_lid_class = encoder_lid_choices.get_class(args.encoder_lid)
                encoder_lid = encoder_lid_class(input_size=lid_input_size, **args.encoder_lid_conf)
                encoder_lid_output_size = encoder_lid.output_size()


        if getattr(args, "postencoder_lid", None) is not None:
            if args.model_conf.get("separate_lid_modules", False):
                postencoder_lid_class = postencoder_lid_choices.get_class(args.postencoder_lid)
                postencoder_lid = torch.nn.ModuleList([postencoder_lid_class(input_size=encoder_lid_output_size, **args.postencoder_lid_conf) for i in range(2)])
                encoder_lid_output_size = postencoder_lid[0].output_size()
            else: 
                postencoder_lid_class = postencoder_lid_choices.get_class(args.postencoder_lid)
                postencoder_lid = postencoder_lid_class(
                    input_size=encoder_lid_output_size, **args.postencoder_lid_conf
                )
                encoder_lid_output_size = postencoder_lid.output_size()

        if getattr(args, "projector_lid", None) is not None:
            if args.model_conf.get("separate_lid_modules", False):
                projector_lid_class = projector_lid_choices.get_class(args.projector_lid)
                projector_lid = torch.nn.ModuleList([projector_lid_class(**args.projector_lid_conf) for i in range(2)])
            else: 
                projector_lid_class = projector_lid_choices.get_class(args.projector_lid)
                projector_lid = projector_lid_class(**args.projector_lid_conf)


        if getattr(args, "loss_lid", None) is not None:
            if args.model_conf.get("separate_lid_modules", False):
                loss_lid_class = loss_lid_choices.get_class(args.loss_lid)
                loss_lid = torch.nn.ModuleList([loss_lid_class(**args.loss_lid_conf) for i in range(2)])
            else: 
                loss_lid_class = loss_lid_choices.get_class(args.loss_lid)
                loss_lid = loss_lid_class(**args.loss_lid_conf)

        # 7. SV modules
        if getattr(args, "preencoder_spk", None) is not None:
            sv_layer_selections = args.frontend_conf.get("sv_layer_selections", None)
            preencoder_sv_nums = len(sv_layer_selections) if sv_layer_selections is not None else 1

            if preencoder_sv_nums > 1:
                preencoder_spk_class = preencoder_choices.get_class(args.preencoder_spk)
                preencoder_spk = torch.nn.ModuleList([preencoder_spk_class(**args.preencoder_spk_conf) for i in range(preencoder_sv_nums)])

                spk_input_size = preencoder_spk[0].output_size()
            else:
                preencoder_spk_class = preencoder_choices.get_class(args.preencoder_spk)
                preencoder_spk = preencoder_spk_class(**args.preencoder_spk_conf)
                spk_input_size = preencoder_spk.output_size()
        else:
            preencoder_spk = None
            if args.input_size is None:
                spk_input_size = frontend.output_size()
            else:
                spk_input_size = args.input_size

        if getattr(args, "encoder_spk", None) is not None:
            if args.model_conf.get("separate_sv_modules", False):
                encoder_spk_class = encoder_spk_choices.get_class(args.encoder_spk)
                encoder_spk = torch.nn.ModuleList([encoder_spk_class(input_size=spk_input_size, **args.encoder_spk_conf) for i in range(2)])
                encoder_spk_output_size = encoder_spk[0].output_size()
            else:
                encoder_spk_class = encoder_spk_choices.get_class(args.encoder_spk)
                encoder_spk = encoder_spk_class(input_size=spk_input_size, **args.encoder_spk_conf)
                encoder_spk_output_size = encoder_spk.output_size()
        else:
            encoder_spk = None
            
    
        if getattr(args, "pooling_spk", None) is not None:
            if args.model_conf.get("separate_sv_modules", False):
                pooling_spk_class = pooling_spk_choices.get_class(args.pooling_spk)
                pooling_spk = torch.nn.ModuleList([pooling_spk_class(input_size=encoder_spk_output_size, **args.pooling_spk_conf) for i in range(2)])
                pooling_spk_output_size = pooling_spk[0].output_size()
            else:
                pooling_spk_class = pooling_spk_choices.get_class(args.pooling_spk)
                pooling_spk = pooling_spk_class(
                    input_size=encoder_spk_output_size, **args.pooling_spk_conf
                )
                pooling_spk_output_size = pooling_spk.output_size()
        else:
            pooling_spk = None

        # import pdb; pdb.set_trace()
        if getattr(args, "projector_spk", None) is not None:
            if args.model_conf.get("separate_sv_modules", False):
                projector_spk_class = projector_spk_choices.get_class(args.projector_spk)
                projector_spk = torch.nn.ModuleList([projector_spk_class(input_size=pooling_spk_output_size, **args.projector_spk_conf) for i in range(2)])
                projector_spk_output_size = projector_spk[0].output_size()
            else:    
                projector_spk_class = projector_spk_choices.get_class(args.projector_spk)
                projector_spk = projector_spk_class(input_size=pooling_spk_output_size, **args.projector_spk_conf)
                projector_spk_output_size = projector_spk.output_size()
        else:
            projector_spk = None

        if getattr(args, "loss_spk", None) is not None:
            if args.model_conf.get("separate_sv_modules", False):
                loss_spk_class = loss_spk_choices.get_class(args.loss_spk)
                loss_spk = torch.nn.ModuleList([loss_spk_class(nout=projector_spk_output_size, nclasses=args.spk_num, **args.loss_spk_conf) for i in range(2)])
            else:
                loss_spk_class = loss_spk_choices.get_class(args.loss_spk)
                loss_spk = loss_spk_class(nout=projector_spk_output_size, nclasses=args.spk_num, **args.loss_spk_conf)
        else:
            loss_spk = None


        # 6. CTC
        ctc = CTC(
            odim=vocab_size, encoder_output_size=encoder_output_size, **args.ctc_conf
        )

        # 7. Build model
        try:
            model_class = model_choices.get_class(args.model)
        except AttributeError:
            model_class = model_choices.get_class("espnet")
        
        if model_class == ESPnetCASSLRModel:
            model = model_class(
                vocab_size=vocab_size,
                frontend=frontend,
                specaug=specaug,
                normalize=normalize,
                preencoder=preencoder,
                encoder=encoder,
                postencoder=postencoder,
                preencoder_lid=preencoder_lid,
                encoder_lid=encoder_lid,
                postencoder_lid=postencoder_lid,
                projector_lid=projector_lid,
                loss_lid=loss_lid,
                decoder=decoder,
                ctc=ctc,
                joint_network=joint_network,
                token_list=token_list,
                lid_tokens=lid_tokens,
                langs_num=langs_num,
                preencoder_spk=preencoder_spk,
                encoder_spk=encoder_spk,
                pooling_spk=pooling_spk,
                projector_spk=projector_spk,
                loss_spk=loss_spk,
                asr_layer_selections=args.frontend_conf.get("asr_layer_selections", None),
                lid_layer_selections=args.frontend_conf.get("lid_layer_selections", None),
                sv_layer_selections=args.frontend_conf.get("sv_layer_selections", None),
                pretrained_asr_path=args.pretrained_models.get("asr_path", None),
                pretrained_lid_path=args.pretrained_models.get("lid_path", None),
                pretrained_sv_path=args.pretrained_models.get("sv_path", None),
                **args.model_conf,
            )
        elif model_class == ESPnetLIDModel:

            projector_lid_class = projector_lid_choices.get_class(args.projector_lid)
            projector_lid = projector_lid_class(**args.projector_lid_conf)

            loss_lid_class = loss_lid_choices.get_class(args.loss_lid)
            loss_lid = loss_lid_class(**args.loss_lid_conf)

            model = model_class(
                vocab_size=vocab_size,
                frontend=frontend,
                specaug=specaug,
                normalize=normalize,
                preencoder=preencoder,
                encoder=encoder,
                postencoder=postencoder,
                projector_lid=projector_lid,
                loss_lid=loss_lid,
                decoder=decoder,
                ctc=ctc,
                joint_network=joint_network,
                token_list=token_list,
                lid_tokens=lid_tokens,
                langs_num=langs_num,
                **args.model_conf,
            )

        else:
            model = model_class(
                vocab_size=vocab_size,
                frontend=frontend,
                specaug=specaug,
                normalize=normalize,
                preencoder=preencoder,
                encoder=encoder,
                postencoder=postencoder,
                decoder=decoder,
                ctc=ctc,
                joint_network=joint_network,
                token_list=token_list,
                **args.model_conf,
            )

        # FIXME(kamo): Should be done in model?
        # 8. Initialize
        if args.init is not None:
            initialize(model, args.init)

        return model
