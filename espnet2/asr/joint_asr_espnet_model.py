import logging
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from packaging.version import parse as V
from typeguard import check_argument_types

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


class ESPnetJointASRModel(ESPnetASRModel):
    """CTC-attention hybrid Encoder-Decoder model"""

    def __init__(
        self,
        vocab_size: int,
        token_list: Union[Tuple[str, ...], List[str]],
        frontend: Optional[AbsFrontend],
        specaug: Optional[AbsSpecAug],
        normalize: Optional[AbsNormalize],
        preencoder: Optional[AbsPreEncoder],
        preencoder_lid: Optional[AbsPreEncoder],
        encoder: AbsEncoder,
        encoder_lid: AbsEncoder,
        postencoder: Optional[AbsPostEncoder],
        decoder: Optional[AbsDecoder],
        postencoder_lid: Optional[AbsPostEncoder],
        projector_lid: Optional[AbsProjector],
        loss_lid: Optional[AbsLoss],
        ctc: CTC,
        joint_network: Optional[torch.nn.Module],
        lid_tokens: Union[Tuple[str, ...], List[str]] = None,
        langs_num: int = 0,
        embed_condition: bool = False,
        embed_condition_size: int = 0,
        lid_condition_feature: str = "soft",
        lid_condition_activate: Optional[str] = None,
        droprate: float = 0.3,
        aux_ctc: dict = None,
        ctc_weight: float = 0.5,
        lid_weight: float = 1.0,
        lid_audio_length: int = 0,
        lid_start_begin: bool = False,
        separate_forward: bool = True,
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
    ):
        assert check_argument_types()
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
            lid_tokens=lid_tokens,
            langs_num=langs_num,
            embed_condition=embed_condition,
            embed_condition_size=embed_condition_size,
            lid_condition_feature=lid_condition_feature,
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
        self.preencoder_lid = preencoder_lid
        self.encoder_lid = encoder_lid
        self.postencoder_lid = postencoder_lid
        self.projector_lid = projector_lid
        self.loss_lid = loss_lid
        self.lid_weight = lid_weight
        self.lid_audio_length = lid_audio_length
        self.lid_start_begin = lid_start_begin
        self.separate_forward = separate_forward
        self.lid_condition_feature = lid_condition_feature
        self.lid_condition_activate = lid_condition_activate
        # if 

        if embed_condition and (lid_condition_feature == "weighted_cosine"):
            self.langs_num = langs_num
            self.lang_embedding = torch.nn.Linear(langs_num, embed_condition_size)
            # self.lang_embedding = torch.nn.Embedding(langs_num, embed_condition_size)

        if self.embed_condition and lid_condition_feature == "soft":
            self.lang_embedding = torch.nn.Linear(256, embed_condition_size)

        if self.lid_condition_activate == "bndrop":
            self.ln = LayerNorm(embed_condition_size, export=False)
            self.activation_fn = torch.nn.PReLU()
            self.dropout = torch.nn.Dropout(p=droprate)

        
    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        langs: torch.Tensor = None,
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

        # for data-parallel
        text = text[:, : text_lengths.max()]

        # 1. Encoder
        encoder_out, encoder_out_lens, lid_embd, encoder_lid_out_lens =  self.encode(speech, speech_lengths, langs)
        intermediate_outs = None
        if isinstance(encoder_out, tuple):
            intermediate_outs = encoder_out[1]
            encoder_out = encoder_out[0]


        loss_lid =self.loss_lid(lid_embd, langs)
        # acc_lid = 

        loss_att, acc_att, cer_att, wer_att = None, None, None, None
        loss_ctc, cer_ctc = None, None
        loss_transducer, cer_transducer, wer_transducer = None, None, None
        stats = dict()

        # 1. CTC branch
        if self.ctc_weight != 0.0:
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

        # Collect total loss stats
        stats["loss_asr"] = loss_asr.detach()
        stats["loss_lid"] = loss_lid.detach()

        loss = loss_asr + self.lid_weight * loss_lid
        stats["loss"] = loss.detach()

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
        if self.embed_condition:
            condition_features = self.lang_embedding(langs)
            
        feats, feats_lengths, feats_layers, feats_lengths_layers  = self._extract_feats(speech, speech_lengths, condition_features)
        return {"feats": feats, "feats_lengths": feats_lengths}

    def encode(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor, langs: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder. Note that this method is used by asr_inference.py

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
        """

        def my_hook(module, input, output):
            # This will be executed upon the forward pass of the hooked layer.
            # 'module' is the layer the hook is attached to,
            # 'input' is the input to the layer,
            # 'output' is the output of the layer.
            # Here you can do things with the output, for instance:
            x, (attn, layer_result) = output  # Storing it in the instance for later use
            self.intermediate_outputs = x


        # set hook for the self.frontend.upstream.upstream.model.encoder.layers[self.frontend.upstream.upstream.sep1_layer]
        if self.separate_forward:
            hook_handle = self.frontend.upstream.upstream.model.encoder.layers[self.frontend.upstream.upstream.sep1_layer-1].register_forward_hook(my_hook)

        with autocast(False):
            condition_features = None
            # 1. Extract feats
            lid_speech = speech
            lid_speech_lengths = speech_lengths
            if self.lid_audio_length > 0 and not self.separate_forward:
                # Random cropping
                if speech_lengths.min() > self.lid_audio_length:
                    if self.lid_start_begin:
                        speech = speech[:, : self.lid_audio_length]
                        speech_lengths = speech.new_full(speech_lengths.shape, self.lid_audio_length, dtype=int)
                    else:
                        lid_start = torch.randint(0, speech_lengths.min() - self.lid_audio_length + 1, (1,)).item()
                        lid_speech = speech[:, lid_start: lid_start + self.lid_audio_length]
                        lid_speech_lengths = lid_speech.new_full(lid_speech_lengths.shape, self.lid_audio_length, dtype=int)   
            
            feats_lid, feats_lengths, feats_layers, feats_lengths_layers = self._extract_feats(lid_speech, lid_speech_lengths, condition_features)
            
            # import pdb; pdb.set_trace()

            # 2. Data augmentation
            if self.specaug is not None and self.training:
                feats_lid, feats_lengths = self.specaug(feats_lid, feats_lengths)

            # 3. Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
            if self.normalize is not None:
                feats_lid, feats_lengths = self.normalize(feats_lid, feats_lengths)

        
        if self.separate_forward:
            hook_handle.remove()

        # Add lid forward and _extract_feats 2

        # Pre-encoder, e.g. used for raw input data
        if self.preencoder_lid is not None:
            feats_lid, feats_lengths = self.preencoder_lid(feats_lid, feats_lengths)

        encoder_lid_out, encoder_lid_out_lens, _ = self.encoder_lid(feats_lid, feats_lengths)

        # Post-encoder, e.g. NLU
        if self.postencoder_lid is not None:
            encoder_lid_out = self.postencoder_lid(
                encoder_lid_out
            )


        # 3. (optionally) go through further projection(s)
        lid_embd = self.project_lid_embd(encoder_lid_out)


        # ys_hat = self.ctc_lid.argmax(encoder_lid_out).data
        # first_non_zero_indices = ys_hat.argmax(1)
        # first_non_zero_values = ys_hat[torch.arange(ys_hat.size(0)), first_non_zero_indices].unsqueeze(1)
        # import pdb; pdb.set_trace()
        if self.embed_condition:
            if self.lid_condition_feature == "hard":
                # Compute cosine similarity
                cosine = F.linear(F.normalize(lid_embd), F.normalize(self.loss_lid.weight))
                
                # Get the predicted speaker index
                cosine_similarity = torch.max(cosine, dim=1)
                langs_token = torch.argmax(cosine, dim=1)

            # condition_features = self.lang_embedding(langs)
                condition_features = self.lang_embedding(langs_token).unsqueeze(1)
            elif self.lid_condition_feature == "weighted_cosine":
                # Compute cosine similarity
                cosine = F.linear(F.normalize(lid_embd), F.normalize(self.loss_lid.weight))
                cosine_probabilities = F.softmax(cosine, dim=-1)
                if self.lid_condition_activate == "LeakyReLU":
                    condition_features = self.lang_embedding(cosine).unsqueeze(1)
                    condition_features = torch.nn.LeakyReLU()(condition_features)
                if self.lid_condition_activate == "bndrop":
                    condition_features = self.lang_embedding(cosine)
                    condition_features = self.ln(condition_features)
                    condition_features = condition_features.unsqueeze(1)
                    condition_features = self.activation_fn(condition_features)
                    condition_features = self.dropout(condition_features)
            elif self.lid_condition_feature == "GT":
                condition_features = self.lang_embedding(langs)
            elif self.lid_condition_feature == "soft":
                if self.lid_condition_activate == "LeakyReLU":
                    # import pdb; pdb.set_trace()
                    condition_features = self.lang_embedding(lid_embd).unsqueeze(1)
                    condition_features = torch.nn.LeakyReLU()(condition_features)
                elif self.lid_condition_activate == "bndrop":
                    # import pdb; pdb.set_trace()
                    condition_features = self.lang_embedding(lid_embd)
                    condition_features = self.ln(condition_features)
                    condition_features = condition_features.unsqueeze(1)
                    condition_features = self.activation_fn(condition_features)
                    condition_features = self.dropout(condition_features)


            # logging.info("lid_embd shape:{}".format(lid_embd.shape))
          
            # # 1. Extract feats
            # feats, feats_lengths, feats_layers, feats_lengths_layers  = self._extract_feats(speech, speech_lengths, condition_features)

        with autocast(False):
            if self.separate_forward:
                feats_layers_new, feats_lengths_new = self.frontend.upstream(speech, speech_lengths, condition_features, split_forward=True, second_forward=True, last_layer_result=self.intermediate_outputs)
                feats_layers = feats_layers[:-1]
                feats_layers.extend(feats_layers_new)
                feats_lengths_layers.extend(feats_lengths_new)
            else:
                feats_layers, feats_lengths_layers = self.frontend.upstream(speech, speech_lengths, condition_features)
           
            feats, feats_lengths = self.frontend.featurizer2(feats_layers, feats_lengths_layers)


            # feats, feats_lengths = self.frontend.featurizer2(feats_layers, feats_lengths_layers)

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
            return (encoder_out, intermediate_outs), encoder_out_lens

        return encoder_out, encoder_out_lens, lid_embd, encoder_lid_out_lens

    def _extract_feats(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor, condition_features: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert speech_lengths.dim() == 1, speech_lengths.shape

        # for data-parallel
        speech = speech[:, : speech_lengths.max()]

        if self.frontend is not None:
            # Frontend
            #  e.g. STFT and Feature extract
            #       data_loader may send time-domain signal in this case
            # speech (Batch, NSamples) -> feats: (Batch, NFrames, Dim)
            # logging.info("langs:{}".format(langs))
            # if self.embed_condition:
            feats, feats_lengths, feats_layers, feats_lengths_layers = self.frontend(speech, speech_lengths, condition_features=condition_features, split_forward=self.separate_forward)
            # else:
            #     feats, feats_lengths, feats_layers, feats_lengths_layers  = self.frontend(speech, speech_lengths)
        else:
            # No frontend and no feature extract
            feats, feats_lengths = speech, speech_lengths
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
