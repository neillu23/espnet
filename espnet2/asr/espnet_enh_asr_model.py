from contextlib import contextmanager
from distutils.version import LooseVersion
import logging
import random
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch
from typeguard import check_argument_types

from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet2.asr.espnet_model import ESPnetASRModel
from espnet2.enh.espnet_model import ESPnetEnhancementModel
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel

if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


class ESPnetEnhASRModel(AbsESPnetModel):
    """CTC-attention hybrid Encoder-Decoder model"""

    def __init__(
        self,
        enh_model: ESPnetEnhancementModel,
        asr_model: ESPnetASRModel,
        enh_loss_weight: float = 0.1,
        permutation_by_enh: bool = True,
        enh_loss_prob: float = 0,   # 1 means calculating enh loss for all data
        bypass_enh_prob: float = 0,  # 0 means do not bypass enhancement for all data
    ):
        assert check_argument_types()

        super().__init__()
        self.enh_model = enh_model
        self.asr_model = asr_model

        self.enh_loss_weight = enh_loss_weight
        self.enh_loss_prob = enh_loss_prob
        self.bypass_enh_prob = bypass_enh_prob

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

        # clean speech signal
        speech_ref = None
        if self.enh_loss_weight > 0:
            assert "speech_ref1" in kwargs
            speech_ref = [
                kwargs["speech_ref1"]
            ]  # [(Batch, samples)] x num_spkr

        # for data-parallel
        text = text[:, : text_lengths.max()]

        utt_id = kwargs.get("utt_id")

        # Calculating enhancement loss
        enh_loss_flag = True
        if utt_id[0].endswith("SIMU"):  # For simulated single-/multi-speaker data: feed it to Enhancement and calculate loss_enh
            enh_loss_flag = True
        elif utt_id[0].endswith("REAL"):  # For single-speaker real data: feed it to Enhancement but without calculating loss_enh with some probability
            if random.random() < self.enh_loss_prob:
                enh_loss_flag = True
            else:
                enh_loss_flag = False
        else:       # For clean data, feed it to Enhancement, without calculating loss_enh
            enh_loss_flag = False

        # Bypass the enhancement module
        bypass_enh_flag = False
        if not enh_loss_flag:
            if random.random() <= self.bypass_enh_prob:
                bypass_enh_flag = True

        # 1. Enhancement
        # model forward
        if not bypass_enh_flag:
            speech_pre, feature_mix, feature_pre, others = self.enh_model.forward_enhance(speech, speech_lengths)
            # loss computation
            if enh_loss_flag and self.enh_loss_weight > 0:
                loss_enh, _, _ = self.enh_model.forward_loss(
                    speech_pre,
                    speech_lengths,
                    feature_mix,
                    feature_pre,
                    others,
                    speech_ref,
                )
                loss_enh = loss_enh[0]
            else:
                loss_enh = None
        else:
            speech_pre = [speech]

        # 2. ASR
        loss_asr, stats, weight = self.asr_model(speech_pre[0], speech_lengths, text, text_lengths)

        if loss_enh is not None:
            # loss = self.enh_loss_weight * loss_enh + (1 - self.enh_loss_weight) * loss_asr
            loss = self.enh_loss_weight * loss_enh + loss_asr
        else:
            loss = loss_asr

        stats["loss_joint"] = loss.detach() if loss is not None else None
        stats["loss_enh"] = loss_enh.detach() if loss_enh is not None else None

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        enh_stats = self.enh_model.collect_feats(speech, speech_lengths)
        asr_stats = self.asr_model.collect_feats(enh_stats["feats"], enh_stats["feats_lengths"], text, text_lengths)

        return asr_stats

    def encode(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder. Note that this method is used by asr_inference.py

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
        """
        speech_pre, feature_mix, feature_pre, others = self.enh_model.forward_enhance(speech, speech_lengths)
        encoder_out, encoder_out_lens = self.asr_model.encode(speech_pre[0], speech_lengths)

        return speech_pre, encoder_out, encoder_out_lens


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
