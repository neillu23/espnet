from contextlib import contextmanager
from distutils.version import LooseVersion
import logging
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch
from typeguard import check_argument_types

from espnet.nets.e2e_asr_common import ErrorCalculator
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (
    LabelSmoothingLoss,  # noqa: H301
)
# from espnet2.asr.ctc import CTC
# from espnet2.asr.decoder.abs_decoder import AbsDecoder
# from espnet2.asr.encoder.abs_encoder import AbsEncoder
# from espnet2.asr.frontend.abs_frontend import AbsFrontend
# from espnet2.asr.postencoder.abs_postencoder import AbsPostEncoder
# from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
# from espnet2.asr.specaug.abs_specaug import AbsSpecAug
# from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


class ESPnetASRModel(AbsESPnetModel):
    """CTC-attention hybrid Encoder-Decoder model"""

    def __init__(
        self,
        vocab_size: int,
        token_list: Union[Tuple[str, ...], List[str]],
        ctc_weight: float = 1.0,
        ignore_id: int = -1,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        report_cer: bool = True,
        report_wer: bool = True,
        sym_space: str = "<space>",
        sym_blank: str = "<blank>",
        extract_feats_in_collect_stats: bool = True,
    ):
        assert check_argument_types()
        assert 0.0 <= ctc_weight <= 1.0, ctc_weight
        # assert rnnt_decoder is None, "Not implemented"

        super().__init__()
        # note that eos is the same as sos (equivalent ID)
        self.sos = 1
        self.eos = 2
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.ctc_weight = ctc_weight
        self.token_list = token_list.copy()

        self.frontend = None
        self.specaug = None
        self.normalize = None
        self.preencoder = None
        self.postencoder = None
        self.encoder = None
        # we set self.decoder = None in the CTC mode since
        # self.decoder parameters were never used and PyTorch complained
        # and threw an Exception in the multi-GPU experiment.
        # thanks Jeff Farris for pointing out the issue.
        assert ctc_weight == 1.0
        self.decoder = None
        self.rnnt_decoder = None
        self.criterion_att = None

        self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        self.ctc_loss_fn = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

        if report_cer or report_wer:
            self.error_calculator = ErrorCalculator(
                token_list, sym_space, sym_blank, report_cer, report_wer
            )
        else:
            self.error_calculator = None

        self.extract_feats_in_collect_stats = extract_feats_in_collect_stats

    def _get_output_length(
        self,
        speech_lengths: torch.Tensor,
    ) -> torch.Tensor:
        speech_lengths = (speech_lengths - 10) // 5 + 1
        speech_lengths = (speech_lengths - 3) // 2 + 1
        speech_lengths = (speech_lengths - 3) // 2 + 1
        speech_lengths = (speech_lengths - 3) // 2 + 1
        speech_lengths = (speech_lengths - 3) // 2 + 1
        speech_lengths = (speech_lengths - 2) // 2 + 1
        speech_lengths = (speech_lengths - 2) // 2 + 1
        return speech_lengths

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            text: (Batch, Length)
            text_lengths: (Batch,)
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

        # for data-parallel
        text = text[:, : text_lengths.max()]

        encoder_out_lens = self._get_output_length(speech_lengths)
        logits = self.model(speech).logits
        loss_ctc, cer_ctc = self._calc_ctc_loss(
            logits, encoder_out_lens, text, text_lengths
        )
        loss = loss_ctc
        loss_att, acc_att, cer_att, wer_att = None, None, None, None

        stats = dict(
            loss=loss.detach(),
            loss_att=loss_att.detach() if loss_att is not None else None,
            loss_ctc=loss_ctc.detach() if loss_ctc is not None else None,
            acc=acc_att,
            cer=cer_att,
            wer=wer_att,
            cer_ctc=cer_ctc,
        )

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        # Generate dummy stats if extract_feats_in_collect_stats is False
        logging.warning(
            "Generating dummy stats for feats and feats_lengths, "
            "because encoder_conf.extract_feats_in_collect_stats is "
            f"{self.extract_feats_in_collect_stats}"
        )
        feats, feats_lengths = speech, speech_lengths
        return {"feats": feats, "feats_lengths": feats_lengths}

    def encode(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder. Note that this method is used by asr_inference.py

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
        """
        encoder_out = self.model(speech).logits
        encoder_out_lens = self._get_output_length(speech_lengths)

        assert encoder_out.size(0) == speech.size(0), (
            encoder_out.size(),
            speech.size(0),
        )
        assert encoder_out.size(1) <= encoder_out_lens.max(), (
            encoder_out.size(),
            encoder_out_lens.max(),
        )

        return encoder_out, encoder_out_lens

    def _extract_feats(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert speech_lengths.dim() == 1, speech_lengths.shape

        # for data-parallel
        speech = speech[:, : speech_lengths.max()]

        # No frontend and no feature extract
        feats, feats_lengths = speech, speech_lengths
        return feats, feats_lengths

    def _calc_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        raise NotImplementedError

    def _calc_ctc_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        # Calc CTC loss
        loss_ctc = self.ctc_loss_fn(
            torch.nn.functional.log_softmax(encoder_out, dim=-1).transpose(1, 0),
            ys_pad,
            encoder_out_lens,
            ys_pad_lens,
        )

        # Calc CER using CTC
        cer_ctc = None
        if not self.training and self.error_calculator is not None:
            ys_hat = torch.argmax(encoder_out, dim=-1).data
            cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
        return loss_ctc, cer_ctc

    def _calc_rnnt_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        raise NotImplementedError
