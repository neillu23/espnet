import logging
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Union

import torch
from packaging.version import parse as V
from typeguard import check_argument_types

from espnet2.asr.ctc import CTC
from espnet2.spk.loss.abs_loss import AbsLoss
from espnet2.spk.projector.abs_projector import AbsProjector
from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.espnet_model import ESPnetASRModel
from espnet2.asr.postencoder.abs_postencoder import AbsPostEncoder
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.asr.transducer.error_calculator import ErrorCalculatorTransducer
from espnet2.asr_transducer.utils import get_transducer_task_io
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


class ESPnetLIDModel(ESPnetASRModel):
    """CTC-attention hybrid Encoder-Decoder model"""

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
        projector: Optional[AbsProjector],
        loss: Optional[AbsLoss],
        ctc: CTC,
        joint_network: Optional[torch.nn.Module],
        lid_tokens: Union[Tuple[str, ...], List[str]] = None,
        langs_num: int = 0,
        embed_condition: bool = False,
        embed_condition_size: int = 0,
        aux_ctc: dict = None,
        ctc_weight: float = 0.5,
        lid_audio_length: int = 0,
        lid_start_begin: bool = False,
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
    ):

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
        self.projector = projector
        self.loss = loss
        self.lid_audio_length = lid_audio_length
        self.lid_start_begin = lid_start_begin

        # assert check_argument_types()
        # assert 0.0 <= ctc_weight <= 1.0, ctc_weight
        # assert 0.0 <= interctc_weight < 1.0, interctc_weight

        # super().__init__()
        # # NOTE (Shih-Lun): else case is for OpenAI Whisper ASR model,
        # #                  which doesn't use <blank> token
        # if sym_blank in token_list:
        #     self.blank_id = token_list.index(sym_blank)
        # else:
        #     self.blank_id = 0
        # if sym_sos in token_list:
        #     self.sos = token_list.index(sym_sos)
        # else:
        #     self.sos = vocab_size - 1
        # if sym_eos in token_list:
        #     self.eos = token_list.index(sym_eos)
        # else:
        #     self.eos = vocab_size - 1
        # self.vocab_size = vocab_size
        # self.ignore_id = ignore_id
        # self.ctc_weight = ctc_weight
        # self.interctc_weight = interctc_weight
        # self.aux_ctc = aux_ctc
        # self.token_list = token_list.copy()
        # if lid_tokens is not None:
        #     self.lid_tokens = lid_tokens.copy()
        # else:
        #     self.lid_tokens = None

        # self.frontend = frontend
        # self.specaug = specaug
        # self.normalize = normalize
        # self.preencoder = preencoder
        # self.postencoder = postencoder
        # self.encoder = encoder
        # self.embed_condition = embed_condition
        # self.embed_condition_size = embed_condition_size

        # if embed_condition:
        #     self.langs_num = langs_num
        #     self.lang_embedding = torch.nn.Embedding(langs_num, embed_condition_size)

        # if not hasattr(self.encoder, "interctc_use_conditioning"):
        #     self.encoder.interctc_use_conditioning = False
        # if self.encoder.interctc_use_conditioning:
        #     self.encoder.conditioning_layer = torch.nn.Linear(
        #         vocab_size, self.encoder.output_size()
        #     )

        # self.use_transducer_decoder = joint_network is not None

        # self.error_calculator = None

        # if self.use_transducer_decoder:
        #     self.decoder = decoder
        #     self.joint_network = joint_network

        #     if not transducer_multi_blank_durations:
        #         from warprnnt_pytorch import RNNTLoss

        #         self.criterion_transducer = RNNTLoss(
        #             blank=self.blank_id,
        #             fastemit_lambda=0.0,
        #         )
        #     else:
        #         from espnet2.asr.transducer.rnnt_multi_blank.rnnt_multi_blank import (
        #             MultiblankRNNTLossNumba,
        #         )

        #         self.criterion_transducer = MultiblankRNNTLossNumba(
        #             blank=self.blank_id,
        #             big_blank_durations=transducer_multi_blank_durations,
        #             sigma=transducer_multi_blank_sigma,
        #             reduction="mean",
        #             fastemit_lambda=0.0,
        #         )
        #         self.transducer_multi_blank_durations = transducer_multi_blank_durations

        #     if report_cer or report_wer:
        #         self.error_calculator_trans = ErrorCalculatorTransducer(
        #             decoder,
        #             joint_network,
        #             token_list,
        #             sym_space,
        #             sym_blank,
        #             report_cer=report_cer,
        #             report_wer=report_wer,
        #         )
        #     else:
        #         self.error_calculator_trans = None

        #         if self.ctc_weight != 0:
        #             self.error_calculator = ErrorCalculator(
        #                 token_list, sym_space, sym_blank, report_cer, report_wer
        #             )
        # else:
        #     # we set self.decoder = None in the CTC mode since
        #     # self.decoder parameters were never used and PyTorch complained
        #     # and threw an Exception in the multi-GPU experiment.
        #     # thanks Jeff Farris for pointing out the issue.
        #     if ctc_weight < 1.0:
        #         assert (
        #             decoder is not None
        #         ), "decoder should not be None when attention is used"
        #     else:
        #         decoder = None
        #         logging.warning("Set decoder to none as ctc_weight==1.0")

        #     self.decoder = decoder

        #     self.criterion_att = LabelSmoothingLoss(
        #         size=vocab_size,
        #         padding_idx=ignore_id,
        #         smoothing=lsm_weight,
        #         normalize_length=length_normalized_loss,
        #     )

        #     if report_cer or report_wer:
        #         self.error_calculator = ErrorCalculator(
        #             token_list, sym_space, sym_blank, report_cer, report_wer
        #         )

        # # if ctc_weight == 0.0:
        # #     self.ctc = None
        # # else:
        # #     self.ctc = ctc

        # self.extract_feats_in_collect_stats = extract_feats_in_collect_stats

        # self.is_encoder_whisper = "Whisper" in type(self.encoder).__name__

        # if self.is_encoder_whisper:
        #     assert (
        #         self.frontend is None
        #     ), "frontend should be None when using full Whisper model"

        # if lang_token_id != -1:
        #     self.lang_token_id = torch.tensor([[lang_token_id]])
        # else:
        #     self.lang_token_id = None

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
        # logging.info("speech length:{}".format(speech_lengths))

        # for data-parallel
        text = text[:, : text_lengths.max()]

        # 1. Encoder
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths, langs)
        intermediate_outs = None
        if isinstance(encoder_out, tuple):
            intermediate_outs = encoder_out[1]
            encoder_out = encoder_out[0]


        # if extract_embd:
        #     return lid_embd

        # 4. calculate loss
        loss = self.loss(encoder_out, text)

        stats = dict(loss=loss.detach())

        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

        # # LID loss calculation
        # loss = nn.CrossEntropyLoss()
        # # Collect total loss stats
        # stats["loss"] = loss(encoder_out, text).mean()

        # # force_gatherable: to-device and to-tensor if scalar for DataParallel
        # loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        # return loss, stats, weight


        # ys_hat = self.ctc.argmax(encoder_out).data

        # logging.info("langs:{}".format(langs))
        # logging.info("ys_hat:{}".format(ys_hat.shape))
        # logging.info("ys_hat value:{}".format(ys_hat))





    def project_lid_embd(self, utt_level_feat: torch.Tensor) -> torch.Tensor:
        if self.projector is not None:
            lid_embd = self.projector(utt_level_feat)
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
            
        feats, feats_lengths = self._extract_feats(speech, speech_lengths, condition_features)
        return {"feats": feats, "feats_lengths": feats_lengths}

    def encode(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor, langs: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder. Note that this method is used by asr_inference.py

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
        """
        with autocast(False):
            condition_features = None
            if self.embed_condition:
                condition_features = self.lang_embedding(langs)

            if self.lid_audio_length > 0:
                if speech_lengths.min() > self.lid_audio_length:
                    if self.lid_start_begin:
                        speech = speech[:, : self.lid_audio_length]
                        speech_lengths = speech.new_full(speech_lengths.shape, self.lid_audio_length, dtype=int)
                    # Random cropping
                    else:
                        lid_start = torch.randint(0, speech_lengths.min() - self.lid_audio_length + 1, (1,)).item()
                        speech = speech[:, lid_start: lid_start + self.lid_audio_length]
                        speech_lengths = speech.new_full(speech_lengths.shape, self.lid_audio_length, dtype=int)   
            
            # 1. Extract feats
            feats, feats_lengths = self._extract_feats(speech, speech_lengths, condition_features)

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
        # logging.info("encoder_out:{}".format(encoder_out.shape))

        # Post-encoder, e.g. NLU
        if self.postencoder is not None:
            encoder_out = self.postencoder(
                encoder_out
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

        # 3. (optionally) go through further projection(s)
        lid_embd = self.project_lid_embd(encoder_out)


        return lid_embd, encoder_out_lens

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
            if self.embed_condition:
                feats, feats_lengths = self.frontend(speech, speech_lengths, condition_features=condition_features)
            else:
                feats, feats_lengths = self.frontend(speech, speech_lengths)
        else:
            # No frontend and no feature extract
            feats, feats_lengths = speech, speech_lengths
        return feats, feats_lengths

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
    ):
        if self.ctc is None:
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
        do_reduce = self.ctc.reduce
        self.ctc.reduce = False
        loss_ctc = self.ctc(encoder_out, encoder_out_lens, text, text_lengths)
        self.ctc.reduce = do_reduce
        return loss_ctc
