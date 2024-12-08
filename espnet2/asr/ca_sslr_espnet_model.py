import logging
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from packaging.version import parse as V
from typeguard import typechecked

from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.espnet_model import ESPnetASRModel
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.postencoder.abs_postencoder import AbsPostEncoder
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.asr.transducer.error_calculator import ErrorCalculatorTransducer
from espnet2.asr_transducer.utils import get_transducer_task_io
from espnet2.spk.pooling.abs_pooling import AbsPooling
from espnet2.spk.loss.abs_loss import AbsLoss
from espnet2.spk.projector.abs_projector import AbsProjector
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet.nets.e2e_asr_common import ErrorCalculator
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (  # noqa: H301
    LabelSmoothingLoss,
)

if V(torch.__version__) >= V("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield

def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True, export=False):
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)


class ESPnetCASSLRModel(ESPnetASRModel):
    """CTC-attention hybrid Encoder-Decoder model"""

    @typechecked
    def __init__(
        self,
        vocab_size: int,
        token_list: Union[Tuple[str, ...], List[str]],
        frontend: Optional[AbsFrontend],
        specaug: Optional[AbsSpecAug],
        normalize: Optional[AbsNormalize],
        preencoder: Optional[AbsPreEncoder],
        encoder: AbsEncoder,
        postencoder: Optional[AbsPostEncoder],
        decoder: Optional[AbsDecoder],
        encoder_lid: Union[AbsEncoder, torch.nn.modules.container.ModuleList],
        preencoder_lid: Union[AbsPreEncoder, torch.nn.modules.container.ModuleList],
        postencoder_lid: Union[AbsPooling, AbsPostEncoder, torch.nn.modules.container.ModuleList],
        projector_lid: Union[AbsProjector, torch.nn.modules.container.ModuleList],
        loss_lid: Union[AbsLoss, torch.nn.modules.container.ModuleList],
        preencoder_spk: Optional[Union[AbsPreEncoder,torch.nn.modules.container.ModuleList]],
        encoder_spk: Optional[Union[AbsEncoder,torch.nn.modules.container.ModuleList]],
        pooling_spk: Optional[Union[AbsPooling,torch.nn.modules.container.ModuleList]],
        projector_spk: Optional[Union[AbsProjector,torch.nn.modules.container.ModuleList]],
        loss_spk: Optional[Union[AbsLoss,torch.nn.modules.container.ModuleList]],
        ctc: CTC,
        joint_network: Optional[torch.nn.Module],
        lid_tokens: Union[Tuple[str, ...], List[str]] = None,
        langs_num: int = 0,
        embed_condition_size: int = 0,
        lid_condition_feature: str = "soft",
        lid_condition_activate: Optional[str] = None,
        droprate: float = 0.3,
        aux_ctc: Optional[dict] = None,
        ctc_weight: float = 0.5,
        lid_weight: float = 1.0,
        spk_weight: float = 1.0,
        lid_ave_weight: float = 1.0,
        spk_ave_weight: float = 1.0,
        lid_audio_length: int = 0,
        spk_audio_length: int = 0,
        interctc_weight: float = 0.0,
        ignore_id: int = -1,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        report_cer: bool = True,
        report_wer: bool = True,
        sym_space: str = "<space>",
        sym_blank: str = "<blank>",
        transducer_multi_blank_durations: List = [],
        transducer_multi_blank_sigma: float = 0.05,
        # In a regular ESPnet recipe, <sos> and <eos> are both "<sos/eos>"
        # Pretrained HF Tokenizer needs custom sym_sos and sym_eos
        sym_sos: str = "<sos/eos>",
        sym_eos: str = "<sos/eos>",
        extract_feats_in_collect_stats: bool = True,
        lang_token_id: int = -1,
        asr_layer_selections:  Optional[list] = None,
        lid_layer_selections:  Optional[list] = None,
        sv_layer_selections:  Optional[list] = None,
        separate_lid_modules: bool = True,
        separate_sv_modules: bool = False,
        skip_asr: bool = False,
        twice_ssl_ablation: bool = False,
        double_condition: bool = False,
        pretrained_asr_path: Optional[str] = None,
        pretrained_lid_path: Optional[str] = None,
        pretrained_sv_path: Optional[str] = None,
    ):
        assert 0.0 <= ctc_weight <= 1.0, ctc_weight
        assert 0.0 <= interctc_weight < 1.0, interctc_weight

        super().__init__(
            vocab_size=vocab_size,
            token_list=token_list,
            frontend=frontend,
            specaug=specaug,
            normalize=normalize,
            preencoder=preencoder,
            encoder=encoder,
            postencoder=postencoder,
            decoder=decoder,
            ctc=ctc,
            joint_network=joint_network,
            aux_ctc=aux_ctc,
            ctc_weight=ctc_weight,
            interctc_weight=interctc_weight,
            ignore_id=ignore_id,
            lsm_weight=lsm_weight,
            length_normalized_loss=length_normalized_loss,
            report_cer=report_cer,
            report_wer=report_wer,
            sym_space=sym_space,
            sym_blank=sym_blank,
            transducer_multi_blank_durations=transducer_multi_blank_durations,
            transducer_multi_blank_sigma=transducer_multi_blank_sigma,
            sym_sos=sym_sos,
            sym_eos=sym_eos,
            extract_feats_in_collect_stats=extract_feats_in_collect_stats,
            lang_token_id=lang_token_id,

        )
        self.separate_lid_modules = separate_lid_modules
        self.separate_sv_modules = separate_sv_modules
        self.skip_asr = skip_asr
        self.twice_ssl_ablation = twice_ssl_ablation
        self.double_condition = double_condition

        if twice_ssl_ablation:
            assert(len(lid_layer_selections) == 2)
        
        self.preencoder_lid = preencoder_lid
        if not self.separate_lid_modules:
            self.encoder_lid = encoder_lid
            self.postencoder_lid = postencoder_lid
            self.projector_lid = projector_lid
            self.loss_lid = loss_lid
        else:
            self.encoder_lid, self.encoder_lid_last = encoder_lid
            self.postencoder_lid, self.postencoder_lid_last = postencoder_lid
            self.projector_lid, self.projector_lid_last = projector_lid
            self.loss_lid, self.loss_lid_last = loss_lid

        self.preencoder_spk = preencoder_spk
        if not self.separate_sv_modules:
            self.encoder_spk = encoder_spk
            self.pooling_spk = pooling_spk
            self.projector_spk = projector_spk
            self.loss_spk = loss_spk
        else:
            self.encoder_spk, self.encoder_spk_last = encoder_spk
            self.pooling_spk, self.pooling_spk_last = pooling_spk
            self.projector_spk, self.projector_spk_last = projector_spk
            self.loss_spk, self.loss_spk_last = loss_spk

        self.spk_weight = spk_weight
        self.spk_audio_length = spk_audio_length


        self.lid_weight = lid_weight
        self.lid_ave_weight = lid_ave_weight
        self.spk_ave_weight = spk_ave_weight
        self.lid_audio_length = lid_audio_length
        self.lid_condition_feature = lid_condition_feature
        self.lid_condition_activate = lid_condition_activate
        
        self.asr_layer_selections = asr_layer_selections
        self.lid_layer_selections = lid_layer_selections
        self.sv_layer_selections = sv_layer_selections
        
    
        # 256, 192 are the size of the lid/speaker embeddings
        lid_cond_num = len(self.lid_layer_selections) - 1
        self.lang_embeddings = torch.nn.ModuleList([torch.nn.Linear(256, embed_condition_size) for i in range(lid_cond_num)])
        self.lns = torch.nn.ModuleList([LayerNorm(embed_condition_size, export=False) for i in range(lid_cond_num)])
        self.activation_fns = torch.nn.ModuleList([torch.nn.PReLU() for i in range(lid_cond_num)])
        self.dropouts = torch.nn.ModuleList([torch.nn.Dropout(p=droprate) for i in range(lid_cond_num)])

        if self.double_condition:
            spk_cond_num = len(self.sv_layer_selections) - 1
            self.spk_embeddings = torch.nn.ModuleList([torch.nn.Linear(192, embed_condition_size) for i in range(spk_cond_num)])
            self.lns_spk = torch.nn.ModuleList([LayerNorm(embed_condition_size, export=False) for i in range(spk_cond_num)])
            self.activation_fns_spk = torch.nn.ModuleList([torch.nn.PReLU() for i in range(spk_cond_num)])
            self.dropouts_spk = torch.nn.ModuleList([torch.nn.Dropout(p=droprate) for i in range(spk_cond_num)])

        self.load_model_parameters(pretrained_asr_path, pretrained_lid_path, pretrained_sv_path)


    def load_model_parameters(self, pretrained_asr_path, pretrained_lid_path, pretrained_sv_path):
        consistent_parameters = []
        inconsistent_parameters = []
        self = self.to("cuda")

        if pretrained_asr_path is not None:
            asr_state_dict = torch.load(pretrained_asr_path, map_location="cuda")
            logging.info("loading pretrained ASR model from {}".format(pretrained_asr_path))
            for name, param in asr_state_dict.items():
                name = name.replace("featurizer.", "asr_featurizers.0.")
                if name in self.state_dict() and param.shape == self.state_dict()[name].shape:
                    if torch.allclose(self.state_dict()[name], param):
                        consistent_parameters.append(name)
                    else:
                        inconsistent_parameters.append(name)
                    self.state_dict()[name].copy_(param)
                else:
                    logging.warning(f"ASR parameter {name} not found or shape mismatch in joint model")

        if pretrained_lid_path is not None:
            lid_state_dict = torch.load(pretrained_lid_path, map_location="cuda")
            logging.info("loading pretrained LID model from {}".format(pretrained_lid_path))
            for name, param in lid_state_dict.items():
                if "frontend.upstream.upstream.model." not in name:
                    name = name.replace("preencoder.", "preencoder_lid.")
                    name = name.replace("encoder.", "encoder_lid.")
                    name = name.replace("projector.", "projector_lid.")
                    name = name.replace("loss.", "loss_lid.")

                if name in self.state_dict() and param.shape == self.state_dict()[name].shape:
                    if torch.allclose(self.state_dict()[name], param):
                        consistent_parameters.append(name)
                    else:
                        inconsistent_parameters.append(name)
                    self.state_dict()[name].copy_(param)

                    if self.separate_lid_modules:
                        if "encoder_lid." in name or "postencoder_lid." in name or "projector_lid." in name or "loss_lid" in name:
                            new_name = name.replace("encoder_lid.", "encoder_lid_last.")
                            new_name = new_name.replace("postencoder_lid.", "postencoder_lid_last.")
                            new_name = new_name.replace("projector_lid.", "projector_lid_last.")
                            new_name = new_name.replace("loss_lid.", "loss_lid_last.")

                            if new_name in self.state_dict() and param.shape == self.state_dict()[new_name].shape:
                                self.state_dict()[new_name].copy_(param)
                                inconsistent_parameters.append(new_name)
                            else:
                                logging.warning(f"Parameter {new_name} in LID model not found in joint model")



                elif "featurizer" in name:
                    new_name = name.replace("featurizer", f"lid_featurizers.{len(self.preencoder_lid)-1}")
                    if new_name in self.state_dict() and param.shape == self.state_dict()[new_name].shape:
                        if torch.allclose(self.state_dict()[new_name], param):
                            consistent_parameters.append(new_name)
                        else:
                            inconsistent_parameters.append(new_name)
                        self.state_dict()[new_name].copy_(param)
                    else:
                        logging.warning(f"Featurizer parameter {name} not found or shape mismatch in joint model")
                elif "preencoder_lid" in name:
                    for i in range(len(self.preencoder_lid)):
                        new_name = name.replace("preencoder_lid", f"preencoder_lid.{i}")
                        if new_name in self.state_dict() and param.shape == self.state_dict()[new_name].shape:
                            if torch.allclose(self.state_dict()[new_name], param):
                                consistent_parameters.append(new_name)
                            else:
                                inconsistent_parameters.append(new_name)
                            self.state_dict()[new_name].copy_(param)
                        else:
                            logging.warning(f"Preencoder parameter {name} not found or shape mismatch in joint model")
                else:
                    logging.warning(f"LID parameter {name} not found or shape mismatch in joint model")

        if pretrained_sv_path is not None:
            sv_state_dict = torch.load(pretrained_sv_path, map_location="cuda")
            logging.info("loading pretrained SV model from {}".format(pretrained_sv_path))
            for name, param in sv_state_dict.items():
                name = name.replace("preencoder.", "preencoder_spk.")
                name = name.replace("encoder.", "encoder_spk.")
                name = name.replace("pooling.", "pooling_spk.")
                name = name.replace("projector.", "projector_spk.")
                name = name.replace("loss.", "loss_spk.")

                if name in self.state_dict() and param.shape == self.state_dict()[name].shape:
                    if torch.allclose(self.state_dict()[name], param):
                        consistent_parameters.append(name)
                    else:
                        inconsistent_parameters.append(name)
                    self.state_dict()[name].copy_(param)

                    if self.separate_sv_modules:
                        if "encoder_spk." in name or "projector_spk." in name or "pooling_spk." in name or "loss_spk." in name:
                            new_name = name.replace("encoder_spk.", "encoder_spk_last.")
                            new_name = name.replace("projector_spk.", "projector_spk_last.")
                            new_name = name.replace("pooling_spk.", "pooling_spk_last.")
                            new_name = name.replace("loss_spk.", "loss_spk_last.")
                            
                            if new_name in self.state_dict() and param.shape == self.state_dict()[new_name].shape:
                                self.state_dict()[new_name].copy_(param)
                                inconsistent_parameters.append(new_name)
                            else:
                                logging.warning(f"Parameter {new_name} in SV model not found in joint model")
                            


                elif "spk_featurizers.0" in name:
                    new_name = name.replace("spk_featurizers.0", f"spk_featurizers.{len(self.preencoder_spk)-1}")
                    if new_name in self.state_dict() and param.shape == self.state_dict()[new_name].shape:
                        if torch.allclose(self.state_dict()[new_name], param):
                            consistent_parameters.append(new_name)
                        else:
                            inconsistent_parameters.append(new_name)
                        self.state_dict()[new_name].copy_(param)
                    else:
                        logging.warning(f"SV parameter {name} not found or shape mismatch in joint model")
                elif "preencoder_spk" in name:
                    for i in range(len(self.preencoder_spk)):
                        new_name = name.replace("preencoder_spk", f"preencoder_spk.{i}")
                        if new_name in self.state_dict() and param.shape == self.state_dict()[new_name].shape:
                            if torch.allclose(self.state_dict()[new_name], param):
                                consistent_parameters.append(new_name)
                            else:
                                inconsistent_parameters.append(new_name)
                            self.state_dict()[new_name].copy_(param)
                        else:
                            logging.warning(f"SV parameter {name} not found or shape mismatch in joint model")
                else:
                    logging.warning(f"SV parameter {name} not found or shape mismatch in joint model")

        logging.info("Pretrained model parameters loaded successfully.")
        logging.info(f"Consistent parameters: {consistent_parameters}")
        logging.info(f"Inconsistent parameters: {inconsistent_parameters}")

        
    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        langs: torch.Tensor = None,
        spk_labels: torch.Tensor = None,
        extract_spk_embd: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            text: (Batch, Length)
            text_lengths: (Batch,)
            kwargs: "utt_id" is among the input.
        """
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (
            speech.shape[0]
            == speech_lengths.shape[0]
            == text.shape[0]
            == text_lengths.shape[0]
        ), (speech.shape, speech_lengths.shape, text.shape, text_lengths.shape)
        batch_size = speech.shape[0]

        text[text == -1] = self.ignore_id

        if not extract_spk_embd:
            asr_valid_indices = [i for i, label in enumerate(text) if label[0] != -1]
            langs_valid_indices = [i for i, label in enumerate(langs) if label[0] != -1]
            
            if self.sv_layer_selections is not None:
                spk_valid_indices = [i for i, label in enumerate(spk_labels) if label[0] != -1]
                # logging.info("spk_valid_indices: {}".format(spk_valid_indices))
            # logging.info("asr_valid_indices: {}".format(asr_valid_indices))
            # logging.info("langs_valid_indices: {}".format(langs_valid_indices))

        # for data-parallel
            text = text[:, : text_lengths.max()]

        # 1. Encoder
        encoder_out, encoder_out_lens, lid_embd_list, encoder_lid_out_lens, spk_embd_list =  self.encode(speech, speech_lengths, langs)
        
        if extract_spk_embd:
            if len(spk_embd_list) == 1:
                spk_embd_list = spk_embd_list[0]
            return spk_embd_list
        
        intermediate_outs = None
        if isinstance(encoder_out, tuple):
            intermediate_outs = encoder_out[1]
            encoder_out = encoder_out[0]


        loss_lid_list = [] #[self.loss_lid(lid_embd, langs) for lid_embd in lid_embd_list]
        acc_lid_list = [] # None
        # calculate accuracy lid for each lid layer, and use 1 best hypothesis
        # do not use error_calculator

        loss_lid_ave = None
        acc_lid_ave = None

        loss_lid = torch.tensor(0.0)
        acc_lid = torch.tensor(0.0)
        if len(self.lid_layer_selections) > 1:
            loss_lid_ave = torch.tensor(0.0)
            acc_lid_ave = torch.tensor(0.0)

        if len(langs_valid_indices) > 0:
            lid_embd_list = [lid_embd[langs_valid_indices] for lid_embd in lid_embd_list]
            langs = langs[langs_valid_indices]

            for i, lid_embd in enumerate(lid_embd_list):
                if i == len(lid_embd_list) - 1 and self.separate_lid_modules:
                    loss_lid = self.loss_lid_last(lid_embd, langs)
                    cosine = F.linear(F.normalize(lid_embd), F.normalize(self.loss_lid_last.weight))
                else:
                    loss_lid = self.loss_lid(lid_embd, langs)
                    cosine = F.linear(F.normalize(lid_embd), F.normalize(self.loss_lid.weight))
                loss_lid_list.append(loss_lid)
                cosine_similarity = torch.max(cosine, dim=1)
                langs_token = torch.argmax(cosine, dim=1)

                acc_lid = (langs.squeeze(1) == langs_token).float().mean().item()
                acc_lid_list.append(acc_lid)

            if len(self.lid_layer_selections) > 1:
                loss_lid_ave = sum(loss_lid_list[:-1]) / (len(loss_lid_list) - 1)
                acc_lid_ave = sum(acc_lid_list[:-1]) / (len(loss_lid_list) - 1)
            loss_lid = loss_lid_list[-1]
            acc_lid = acc_lid_list[-1]

        stats = dict()

        stats["loss_lid"] = loss_lid.detach() if loss_lid is not None else None
        stats["loss_lid_ave"] = loss_lid_ave.detach() if loss_lid_ave is not None else None
        stats["acc_lid"] = acc_lid
        stats["acc_lid_ave"] = acc_lid_ave
        
        loss_spk_list = []
        if self.sv_layer_selections is not None:
            loss_spk = torch.tensor(0.0)
            if len(self.sv_layer_selections) > 1:
                loss_spk_ave = torch.tensor(0.0)
            else:
                loss_spk_ave = None

            if len(spk_valid_indices) > 0:
                spk_embd_list = [spk_embd[spk_valid_indices] for spk_embd in spk_embd_list]
                spk_labels = spk_labels[spk_valid_indices]
                # loss_spk = self.loss_spk(spk_embd, spk_labels)

                for i, spk_embd in enumerate(spk_embd_list):
                    if i == len(spk_embd_list) - 1 and self.separate_sv_modules:
                        loss_spk = self.loss_spk_last(spk_embd, spk_labels)
                        cosine = F.linear(F.normalize(spk_embd), F.normalize(self.loss_spk_last.weight))
                    else:
                        loss_spk = self.loss_spk(spk_embd, spk_labels)
                        cosine = F.linear(F.normalize(spk_embd), F.normalize(self.loss_spk.weight))
                    loss_spk_list.append(loss_spk)

                if len(self.sv_layer_selections) > 1:
                    loss_spk_ave = sum(loss_spk_list[:-1]) / (len(loss_spk_list) - 1)
                loss_spk = loss_spk_list[-1]
            
            stats["loss_spk"] = loss_spk.detach()
            stats["loss_spk_ave"] = loss_spk_ave.detach() if loss_spk_ave is not None else None
        else:
            loss_spk = None
            loss_spk_ave = None


        if self.skip_asr:
            if loss_lid_ave is not None:
                loss = self.lid_ave_weight * loss_lid_ave.to(speech.device) + self.lid_weight * loss_lid.to(speech.device)
            else:
                loss = self.lid_weight
                
            if loss_spk is not None:
                loss += self.spk_weight * loss_spk.to(speech.device)
                
            stats["loss"] = loss.detach()

            # force_gatherable: to-device and to-tensor if scalar for DataParallel
            loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
            return loss, stats, weight
            # return {"loss":loss, "loss_lid":loss_lid, "loss_asr":loss_asr}, stats, weight


        loss_att, acc_att, cer_att, wer_att = None, None, None, None
        loss_ctc, cer_ctc = None, None
        loss_transducer, cer_transducer, wer_transducer = None, None, None

        # 1. CTC branch
        if len(asr_valid_indices) > 0:
            if self.ctc_weight != 0.0:
                # filter text labels
                
                text = text[asr_valid_indices]
                text_lengths = text_lengths[asr_valid_indices]
                encoder_out = encoder_out[asr_valid_indices]
                encoder_out_lens = encoder_out_lens[asr_valid_indices]

                loss_ctc, cer_ctc = self._calc_ctc_loss(
                    encoder_out, encoder_out_lens, text, text_lengths
                )

                # Collect CTC branch stats
                stats["loss_ctc"] = loss_ctc.detach() if loss_ctc is not None else None
                stats["cer_ctc"] = cer_ctc

            # Intermediate CTC (optional)
            loss_interctc = 0.0
            if self.interctc_weight != 0.0 and intermediate_outs is not None:
                for layer_idx, intermediate_out in intermediate_outs:
                    # we assume intermediate_out has the same length & padding
                    # as those of encoder_out

                    # use auxillary ctc data if specified
                    loss_ic = None
                    if self.aux_ctc is not None:
                        idx_key = str(layer_idx)
                        if idx_key in self.aux_ctc:
                            aux_data_key = self.aux_ctc[idx_key]
                            aux_data_tensor = kwargs.get(aux_data_key, None)
                            aux_data_lengths = kwargs.get(aux_data_key + "_lengths", None)

                            if aux_data_tensor is not None and aux_data_lengths is not None:
                                loss_ic, cer_ic = self._calc_ctc_loss(
                                    intermediate_out,
                                    encoder_out_lens,
                                    aux_data_tensor,
                                    aux_data_lengths,
                                )
                            else:
                                raise Exception(
                                    "Aux. CTC tasks were specified but no data was found"
                                )
                    if loss_ic is None:
                        loss_ic, cer_ic = self._calc_ctc_loss(
                            intermediate_out, encoder_out_lens, text, text_lengths
                        )
                    loss_interctc = loss_interctc + loss_ic

                    # Collect Intermedaite CTC stats
                    stats["loss_interctc_layer{}".format(layer_idx)] = (
                        loss_ic.detach() if loss_ic is not None else None
                    )
                    stats["cer_interctc_layer{}".format(layer_idx)] = cer_ic

                loss_interctc = loss_interctc / len(intermediate_outs)

                # calculate whole encoder loss
                loss_ctc = (
                    1 - self.interctc_weight
                ) * loss_ctc + self.interctc_weight * loss_interctc

            if self.use_transducer_decoder:
                # 2a. Transducer decoder branch
                (
                    loss_transducer,
                    cer_transducer,
                    wer_transducer,
                ) = self._calc_transducer_loss(
                    encoder_out,
                    encoder_out_lens,
                    text,
                )

                if loss_ctc is not None:
                    loss_asr = loss_transducer + (self.ctc_weight * loss_ctc)
                else:
                    loss_asr = loss_transducer

                # Collect Transducer branch stats
                stats["loss_transducer"] = (
                    loss_transducer.detach() if loss_transducer is not None else None
                )
                stats["cer_transducer"] = cer_transducer
                stats["wer_transducer"] = wer_transducer

            else:
                # logging.info("text",text)
                # logging.info("text_lengths",text_lengths)
                # 2b. Attention decoder branch
                if self.ctc_weight != 1.0:
                    loss_att, acc_att, cer_att, wer_att = self._calc_att_loss(
                        encoder_out, encoder_out_lens, text, text_lengths
                    )

                # 3. CTC-Att loss definition
                if self.ctc_weight == 0.0:
                    loss_asr = loss_att
                elif self.ctc_weight == 1.0:
                    loss_asr = loss_ctc
                else:
                    loss_asr = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att

                # Collect Attn branch stats
                stats["loss_att"] = loss_att.detach() if loss_att is not None else None
                stats["acc"] = acc_att
                stats["cer"] = cer_att
                stats["wer"] = wer_att
        else:
            # logging.info("no text data in this batch")
            loss_asr = torch.tensor(0.0)
            # self.zero_grad_asr()

            if self.ctc_weight != 0.0:
                stats["loss_ctc"] = 0.0

                if self.training or self.error_calculator is None:
                    stats["cer_ctc"] = None
                else:
                    stats["cer_ctc"] = 0.0

            # Intermediate CTC (optional)
            if self.interctc_weight != 0.0 and intermediate_outs is not None:
                for layer_idx, intermediate_out in intermediate_outs:
                    # Collect Intermedaite CTC stats
                    stats["loss_interctc_layer{}".format(layer_idx)] = 0.0
                    
                    stats["cer_interctc_layer{}".format(layer_idx)] = 0.0

            if self.use_transducer_decoder:
                stats["loss_transducer"] = 0.0
                stats["cer_transducer"] =  0.0
                stats["wer_transducer"] =  0.0

            else:
                # Collect Attn branch stats
                stats["loss_att"] = None
                stats["acc"] = None
                stats["cer"] = None
                stats["wer"] = None


        # Collect total loss stats
        stats["loss_asr"] = loss_asr.detach()

        loss = loss_asr.to(speech.device) + self.lid_weight * loss_lid.to(speech.device)
        if loss_lid_ave is not None:
            loss += self.lid_ave_weight * loss_lid_ave.to(speech.device)
        if loss_spk is not None:
            loss += self.spk_weight * loss_spk.to(speech.device)
        if loss_spk_ave is not None:
            loss += self.spk_ave_weight * loss_spk_ave.to(speech.device)
            
        stats["loss"] = loss.detach()

        # logging.info("loss_asr: {}".format(loss_asr))
        # logging.info("loss_lid: {}".format(loss_lid))
        # logging.info("loss_lid_ave: {}".format(loss_lid_ave))
        # logging.info("loss_spk: {}".format(loss_spk))
        # logging.info("stats: {}".format(stats))
        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight
        # return {"loss":loss, "loss_lid":loss_lid, "loss_asr":loss_asr}, stats, weight


    def project_lid_embd(self, utt_level_feat: torch.Tensor) -> torch.Tensor:
        if self.projector_lid is not None:
            lid_embd = self.projector_lid(utt_level_feat)
        else:
            lid_embd = utt_level_feat

        return lid_embd

    def project_lid_embd_last(self, utt_level_feat: torch.Tensor) -> torch.Tensor:
        if self.projector_lid_last is not None:
            lid_embd = self.projector_lid_last(utt_level_feat)
        else:
            lid_embd = utt_level_feat

        return lid_embd

    def project_spk_embd(self, utt_level_feat: torch.Tensor) -> torch.Tensor:
        if self.projector_spk is not None:
            lid_embd = self.projector_spk(utt_level_feat)
        else:
            lid_embd = utt_level_feat

        return lid_embd

    def project_spk_embd_last(self, utt_level_feat: torch.Tensor) -> torch.Tensor:
        if self.projector_spk is not None:
            lid_embd = self.projector_spk_last(utt_level_feat)
        else:
            lid_embd = utt_level_feat

        return lid_embd





    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        langs: torch.Tensor = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        
        condition_features = None
        # import pdb; pdb.set_trace()
        
        # condition_features = self.lang_embedding(langs)
            
        feats, feats_lengths, feats_layers, feats_lengths_layers  = self._extract_feats(speech, speech_lengths, condition_features)
        return {"feats": feats, "feats_lengths": feats_lengths}

    def encode(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor, langs: torch.Tensor = None, extract_spk_embd: bool = False, inference: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder. Note that this method is used by asr_inference.py

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
        """


        def my_hook_asr(module, input, output):
            # This will be executed upon the forward pass of the hooked layer.
            # 'module' is the layer the hook is attached to,
            # 'input' is the input to the layer,
            # 'output' is the output of the layer.
            # Here you can do things with the output, for instance:
            x, (attn, layer_result, self_attn_padding_mask) = output  # Storing it in the instance for later use
            self.intermediate_outputs_asr = x
            self.mask_asr = self_attn_padding_mask


        def my_hook_lid(module, input, output):
            # This will be executed upon the forward pass of the hooked layer.
            # 'module' is the layer the hook is attached to,
            # 'input' is the input to the layer,
            # 'output' is the output of the layer.
            # Here you can do things with the output, for instance:
            x, (attn, layer_result, self_attn_padding_mask) = output  # Storing it in the instance for later use
            self.intermediate_outputs_lid = x
            self.mask_lid = self_attn_padding_mask

        
        speech = speech[:, : speech_lengths.max()]

        lid_embd_list = []
        spk_embd_list = []
        padding_mask = None
        condition_features = None
        condition_features_lid = None
        condition_features_spk = None

        chunk_features = not inference and self.lid_audio_length > 0 and speech_lengths.min() > self.lid_audio_length
        full_features = ((not self.skip_asr) and (not extract_spk_embd)) or not chunk_features
        
        if chunk_features:
            # logging.info("chunking")
            chunk_start = torch.randint(0, speech_lengths.min() - self.lid_audio_length + 1, (1,)).item()
            speech_chunk = speech[:, chunk_start: chunk_start + self.lid_audio_length]
            speech_chunk_lengths = speech_chunk.new_full(speech_lengths.shape, self.lid_audio_length, dtype=int)
        # if full_features:
        #     logging.info("full feature")

        for index in range(len(self.lid_layer_selections)):
            if index == 0 or self.twice_ssl_ablation:
                if full_features:
                    hook_handle = self.frontend.upstream.upstream.model.encoder.layers[self.lid_layer_selections[0]-1].register_forward_hook(my_hook_asr)
                    feats_layers, feats_lengths_layers = self.frontend(
                                                            speech, speech_lengths, 
                                                            condition_features=condition_features, 
                                                            layer=self.lid_layer_selections[index]
                                                            )
                    hook_handle.remove()
                
                if chunk_features:
                    hook_handle = self.frontend.upstream.upstream.model.encoder.layers[self.lid_layer_selections[0]-1].register_forward_hook(my_hook_lid)
                    feats_layers_chunk, feats_lengths_layers_chunk = self.frontend(
                                                            speech_chunk, speech_chunk_lengths, 
                                                            condition_features=condition_features, 
                                                            layer=self.lid_layer_selections[index]
                                                            )
                    hook_handle.remove()

            else:
                if full_features:
                    self.intermediate_outputs_asr, feats_layers, feats_lengths_layers = self.frontend.encode_layers(
                                                            speech, speech_lengths, 
                                                            condition_features=condition_features,
                                                            last_layer_result=self.intermediate_outputs_asr, 
                                                            layers=(self.lid_layer_selections[index-1], self.lid_layer_selections[index]),
                                                            feats_layers=feats_layers,
                                                            feats_lengths_layers=feats_lengths_layers,
                                                            padding_mask=self.mask_asr
                                                            )
                if chunk_features:
                    self.intermediate_outputs_lid, feats_layers_chunk, feats_lengths_layers_chunk = self.frontend.encode_layers(
                                                            speech_chunk, speech_chunk_lengths, 
                                                            condition_features=condition_features,
                                                            last_layer_result=self.intermediate_outputs_lid, 
                                                            layers=(self.lid_layer_selections[index-1], self.lid_layer_selections[index]),
                                                            feats_layers=feats_layers_chunk,
                                                            feats_lengths_layers=feats_lengths_layers_chunk,
                                                            padding_mask=self.mask_lid
                                                            )

            if not chunk_features:
                feats_layers_chunk = feats_layers
                feats_lengths_layers_chunk = feats_lengths_layers

            feats_lid, feats_lid_lengths = self.frontend.lid_featurizers[index](feats_layers_chunk, feats_lengths_layers_chunk)

            # 2. Data augmentation
            if self.specaug is not None and self.training:
                feats_lid, feats_lid_lengths = self.specaug(feats_lid, feats_lid_lengths)

            # 3. Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
            if self.normalize is not None:
                feats_lid, feats_lid_lengths = self.normalize(feats_lid, feats_lid_lengths)

            # 4. Forward encoder for LID
            if len(self.lid_layer_selections) > 1:
                feats_lid, feats_lid_lengths = self.preencoder_lid[index](feats_lid, feats_lid_lengths)
            else:
                feats_lid, feats_lid_lengths = self.preencoder_lid(feats_lid, feats_lid_lengths)

            if not self.separate_lid_modules or index < (len(self.lid_layer_selections) - 1):
                encoder_lid_out, encoder_lid_out_lens, _ = self.encoder_lid(feats_lid, feats_lid_lengths)

                if self.postencoder_lid is not None:
                    encoder_lid_out = self.postencoder_lid(
                        encoder_lid_out
                    )

                lid_embd = self.project_lid_embd(encoder_lid_out)
            else:
                encoder_lid_out, encoder_lid_out_lens, _ = self.encoder_lid_last(feats_lid, feats_lid_lengths)

                if self.postencoder_lid_last is not None:
                    encoder_lid_out = self.postencoder_lid_last(
                        encoder_lid_out
                    )

                lid_embd = self.project_lid_embd_last(encoder_lid_out)
                
            lid_embd_list.append(lid_embd)


            if self.sv_layer_selections is not None and self.lid_layer_selections[index] in self.sv_layer_selections:
                # get index in sv
                sv_index = self.sv_layer_selections.index(self.lid_layer_selections[index])
                task_tokens = None
                batch_size = speech.shape[0]

                feats_spk, feats_spk_lengths = self.frontend.spk_featurizers[sv_index](feats_layers_chunk, feats_lengths_layers_chunk)

                if len(self.sv_layer_selections) > 1:
                    feats_spk, feats_spk_lengths = self.preencoder_spk[sv_index](feats_spk, feats_spk_lengths)
                else:
                    feats_spk, feats_spk_lengths = self.preencoder_spk(feats_spk, feats_spk_lengths)

                if not self.separate_sv_modules or sv_index < (len(self.sv_layer_selections) - 1):
                    frame_level_feats = self.encoder_spk(feats_spk)
                    utt_level_feat = self.pooling_spk(frame_level_feats, task_tokens)
                    spk_embd = self.project_spk_embd(utt_level_feat)
                else:
                    frame_level_feats = self.encoder_spk_last(feats_spk)
                    utt_level_feat = self.pooling_spk_last(frame_level_feats, task_tokens)
                    spk_embd = self.project_spk_embd_last(utt_level_feat)
                    
                spk_embd_list.append(spk_embd)

            # does not need to generate condition features from the last layer
            if index == len(self.lid_layer_selections) - 1: # not self.last_condition and 
                break

            # 5. generate lid condition features
            if self.lid_condition_feature == "hard":
                # Compute cosine similarity
                if not self.separate_lid_modules or index < (len(self.lid_layer_selections) - 1):
                    cosine = F.linear(F.normalize(lid_embd), F.normalize(self.loss_lid.weight))
                else:
                    cosine = F.linear(F.normalize(lid_embd), F.normalize(self.loss_lid_last.weight))

                # Get the predicted speaker index
                cosine_similarity = torch.max(cosine, dim=1)
                langs_token = torch.argmax(cosine, dim=1)
                condition_features = self.lang_embedding(langs_token).unsqueeze(1)

            elif self.lid_condition_feature == "GT":
                condition_features = self.lang_embedding(langs)

            elif self.lid_condition_feature == "soft":
                if not self.double_condition:
                    condition_features = self.lang_embeddings[index](lid_embd)
                    condition_features = self.lns[index](condition_features)
                    condition_features = condition_features.unsqueeze(1)
                    condition_features = self.activation_fns[index](condition_features)
                    condition_features = self.dropouts[index](condition_features)
                else:
                    condition_features_lid = self.lang_embeddings[index](lid_embd)
                    condition_features_lid = self.lns[index](condition_features_lid)
                    condition_features_lid = condition_features_lid.unsqueeze(1)
                    condition_features_lid = self.activation_fns[index](condition_features_lid)
                    condition_features_lid = self.dropouts[index](condition_features_lid)

                    if self.lid_layer_selections[index] in self.sv_layer_selections:
                        condition_features_spk = self.spk_embeddings[sv_index](spk_embd)
                        condition_features_spk = self.lns_spk[sv_index](condition_features_spk)
                        condition_features_spk = condition_features_spk.unsqueeze(1)
                        condition_features_spk = self.activation_fns_spk[sv_index](condition_features_spk)
                        condition_features_spk = self.dropouts_spk[sv_index](condition_features_spk)

                    condition_features = [condition_features_lid, condition_features_spk]
                    
            # # for sanity check
            # condition_features = None

        # # 7. Forward encoder for SV
        # if self.sv_layer_selections is not None:
        #     task_tokens = None
        #     batch_size = speech.shape[0]

        #     feats, feats_lengths = self.frontend.spk_featurizers[-1](feats_layers_chunk, feats_lengths_layers_chunk)
        #     if self.preencoder_spk is not None:
        #         feats, feats_lengths = self.preencoder_spk(feats, feats_lengths)
        #     frame_level_feats = self.encoder_spk(feats)
        #     utt_level_feat = self.pooling_spk(frame_level_feats, task_tokens)
        #     spk_embd = self.project_spk_embd(utt_level_feat)
        # else:
        #     spk_embd = None

        if self.skip_asr or extract_spk_embd:
            return _, _, lid_embd_list, encoder_lid_out_lens, spk_embd_list


        # 6. Forward encoder for ASR
        with autocast(False):
            # not using for now
            if self.lid_layer_selections[-1] < self.asr_layer_selections[0]:
                self.intermediate_outputs, feats_layers, feats_lengths_layers = self.frontend.encode_layers(
                                                        speech, speech_lengths, 
                                                        condition_features=condition_features,
                                                        last_layer_result=self.intermediate_outputs, 
                                                        layer=(self.lid_layer_selections[-1], self.asr_layer_selections[0]),
                                                        feats_layers=feats_layers,
                                                        feats_lengths_layers=feats_lengths_layers
                                                        )

                
            # ori_feats_layers, ori_feats_lengths_layers = self.frontend.upstream(speech, speech_lengths, condition_features)

            # ori_feats_layers_chunk, ori_feats_lengths_layers_chunk = self.frontend.upstream(speech_chunk, speech_chunk_lengths, condition_features)

            # for i in range(25):
            #     # logging.info("ori_feats_layers {}: {}".format(i, ori_feats_layers[i]))
            #     # logging.info("feats_layers {}: {}".format(i, feats_layers[i]))
            #     same = torch.all(torch.eq(ori_feats_layers[i],feats_layers[i]))
            #     same_lid = torch.all(torch.eq(ori_feats_layers_chunk[i],feats_layers_chunk[i]))
            #     logging.info("feats_layers {} is the same as ori_feats_layers or not: {}".format(i, same))
            #     logging.info("feats_layers_chunk {} is the same as ori_feats_layers_chunk or not: {}".format(i, same_lid))
            #     if not same:
            #         # logging.info("ori_feats_layers: {}".format(ori_feats_layers[i]))
            #         # logging.info("feats_layers: {}".format(feats_layers[i]))
            #         logging.info("same part of feats_layers {} is: {}".format(i, torch.eq(ori_feats_layers[i],feats_layers[i])))
            #         # import pdb; pdb.set_trace()
            #         # logging.info("same part of feats_layers {} with {} is: {}".format(i, i+1, torch.eq(ori_feats_layers[i],feats_layers[i+1])))
            #     # logging.info("feats_layers {} is the same as ori_feats_layers or not: {}".format(i, torch.all(torch.eq(ori_feats_layers[i],feats_layers[i]))))
            #     if not same_lid:
            #         # logging.info("ori_feats_layers: {}".format(ori_feats_layers[i]))
            #         # logging.info("feats_layers: {}".format(feats_layers[i]))
            #         logging.info("same part of feats_layers_chunk {} is: {}".format(i, torch.eq(ori_feats_layers_chunk[i],feats_layers_chunk[i])))
            #         # import pdb; pdb.set_trace()
            #         # logging.info("same part of feats_layers {} with {} is: {}".format(i, i+1, torch.eq(ori_feats_layers[i],feats_layers[i+1])))
            # exit()            

            feats, feats_lengths = self.frontend.asr_featurizers[-1](feats_layers, feats_lengths_layers)


            # 2. Data augmentation
            if self.specaug is not None and self.training:
                feats, feats_lengths = self.specaug(feats, feats_lengths)

            # 3. Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
            if self.normalize is not None:
                feats, feats_lengths = self.normalize(feats, feats_lengths)

        # Pre-encoder, e.g. used for raw input data
        if self.preencoder is not None:
            feats, feats_lengths = self.preencoder(feats, feats_lengths)

        # 4. Forward encoder
        # feats: (Batch, Length, Dim)
        # -> encoder_out: (Batch, Length2, Dim2)
        if self.encoder.interctc_use_conditioning:
            encoder_out, encoder_out_lens, _ = self.encoder(
                feats, feats_lengths, ctc=self.ctc
            )
        else:
            encoder_out, encoder_out_lens, _ = self.encoder(feats, feats_lengths, condition_features=condition_features)
        intermediate_outs = None
        if isinstance(encoder_out, tuple):
            intermediate_outs = encoder_out[1]
            encoder_out = encoder_out[0]

        # Post-encoder, e.g. NLU
        if self.postencoder is not None:
            encoder_out, encoder_out_lens = self.postencoder(
                encoder_out, encoder_out_lens
            )

        assert encoder_out.size(0) == speech.size(0), (
            encoder_out.size(),
            speech.size(0),
        )
        if (
            getattr(self.encoder, "selfattention_layer_type", None) != "lf_selfattn"
            and not self.is_encoder_whisper
        ):
            assert encoder_out.size(-2) <= encoder_out_lens.max(), (
                encoder_out.size(),
                encoder_out_lens.max(),
            )


        if intermediate_outs is not None:
            return (encoder_out, intermediate_outs), encoder_out_lens, spk_embd_list

        return encoder_out, encoder_out_lens, lid_embd_list, encoder_lid_out_lens, spk_embd_list

    def _extract_feats(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor, condition_features: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert speech_lengths.dim() == 1, speech_lengths.shape

        # for data-parallel
        speech = speech[:, : speech_lengths.max()]

        feats_layers, feats_lengths_layers = self.frontend(speech, speech_lengths, condition_features=condition_features)
        feats, feats_lengths = self.frontend.asr_featurizers[-1](feats_layers, feats_lengths_layers)

        return feats, feats_lengths, feats_layers, feats_lengths_layers

    def nll(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Compute negative log likelihood(nll) from transformer-decoder

        Normally, this function is called in batchify_nll.

        Args:
            encoder_out: (Batch, Length, Dim)
            encoder_out_lens: (Batch,)
            ys_pad: (Batch, Length)
            ys_pad_lens: (Batch,)
        """
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        # 1. Forward decoder
        decoder_out, _ = self.decoder(
            encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens
        )  # [batch, seqlen, dim]
        batch_size = decoder_out.size(0)
        decoder_num_class = decoder_out.size(2)
        # nll: negative log-likelihood
        nll = torch.nn.functional.cross_entropy(
            decoder_out.view(-1, decoder_num_class),
            ys_out_pad.view(-1),
            ignore_index=self.ignore_id,
            reduction="none",
        )
        nll = nll.view(batch_size, -1)
        nll = nll.sum(dim=1)
        assert nll.size(0) == batch_size
        return nll

    def batchify_nll(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
        batch_size: int = 100,
    ):
        """Compute negative log likelihood(nll) from transformer-decoder

        To avoid OOM, this fuction seperate the input into batches.
        Then call nll for each batch and combine and return results.
        Args:
            encoder_out: (Batch, Length, Dim)
            encoder_out_lens: (Batch,)
            ys_pad: (Batch, Length)
            ys_pad_lens: (Batch,)
            batch_size: int, samples each batch contain when computing nll,
                        you may change this to avoid OOM or increase
                        GPU memory usage
        """
        total_num = encoder_out.size(0)
        if total_num <= batch_size:
            nll = self.nll(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)
        else:
            nll = []
            start_idx = 0
            while True:
                end_idx = min(start_idx + batch_size, total_num)
                batch_encoder_out = encoder_out[start_idx:end_idx, :, :]
                batch_encoder_out_lens = encoder_out_lens[start_idx:end_idx]
                batch_ys_pad = ys_pad[start_idx:end_idx, :]
                batch_ys_pad_lens = ys_pad_lens[start_idx:end_idx]
                batch_nll = self.nll(
                    batch_encoder_out,
                    batch_encoder_out_lens,
                    batch_ys_pad,
                    batch_ys_pad_lens,
                )
                nll.append(batch_nll)
                start_idx = end_idx
                if start_idx == total_num:
                    break
            nll = torch.cat(nll)
        assert nll.size(0) == total_num
        return nll

    def _calc_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        if hasattr(self, "lang_token_id") and self.lang_token_id is not None:
            ys_pad = torch.cat(
                [
                    self.lang_token_id.repeat(ys_pad.size(0), 1).to(ys_pad.device),
                    ys_pad,
                ],
                dim=1,
            )
            ys_pad_lens += 1

        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        # 1. Forward decoder
        decoder_out, _ = self.decoder(
            encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens
        )

        # 2. Compute attention loss
        loss_att = self.criterion_att(decoder_out, ys_out_pad)
        acc_att = th_accuracy(
            decoder_out.view(-1, self.vocab_size),
            ys_out_pad,
            ignore_label=self.ignore_id,
        )

        # Compute cer/wer using attention-decoder
        if self.training or self.error_calculator is None:
            cer_att, wer_att = None, None
        else:
            ys_hat = decoder_out.argmax(dim=-1)
            cer_att, wer_att = self.error_calculator(ys_hat.cpu(), ys_pad.cpu())

        return loss_att, acc_att, cer_att, wer_att

    def _calc_ctc_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        # Calc CTC loss
        loss_ctc = self.ctc(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)

        # Calc CER using CTC
        cer_ctc = None
        if not self.training and self.error_calculator is not None:
            ys_hat = self.ctc.argmax(encoder_out).data
            cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
        return loss_ctc, cer_ctc

    def _calc_transducer_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        labels: torch.Tensor,
    ):
        """Compute Transducer loss.

        Args:
            encoder_out: Encoder output sequences. (B, T, D_enc)
            encoder_out_lens: Encoder output sequences lengths. (B,)
            labels: Label ID sequences. (B, L)

        Return:
            loss_transducer: Transducer loss value.
            cer_transducer: Character error rate for Transducer.
            wer_transducer: Word Error Rate for Transducer.

        """
        decoder_in, target, t_len, u_len = get_transducer_task_io(
            labels,
            encoder_out_lens,
            ignore_id=self.ignore_id,
            blank_id=self.blank_id,
        )

        self.decoder.set_device(encoder_out.device)
        decoder_out = self.decoder(decoder_in)

        joint_out = self.joint_network(
            encoder_out.unsqueeze(2), decoder_out.unsqueeze(1)
        )

        loss_transducer = self.criterion_transducer(
            joint_out,
            target,
            t_len,
            u_len,
        )

        cer_transducer, wer_transducer = None, None
        if not self.training and self.error_calculator_trans is not None:
            cer_transducer, wer_transducer = self.error_calculator_trans(
                encoder_out, target
            )

        return loss_transducer, cer_transducer, wer_transducer

    def _calc_batch_ctc_loss(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        ctc: CTC,
    ):
        if ctc is None:
            return
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (
            speech.shape[0]
            == speech_lengths.shape[0]
            == text.shape[0]
            == text_lengths.shape[0]
        ), (speech.shape, speech_lengths.shape, text.shape, text_lengths.shape)

        # for data-parallel
        text = text[:, : text_lengths.max()]

        # 1. Encoder
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]

        # Calc CTC loss
        do_reduce = ctc.reduce
        ctc.reduce = False
        loss_ctc = ctc(encoder_out, encoder_out_lens, text, text_lengths)
        ctc.reduce = do_reduce
        return loss_ctc
