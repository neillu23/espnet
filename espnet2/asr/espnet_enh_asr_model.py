from itertools import chain
from math import ceil
import random
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from typeguard import check_argument_types

from espnet.nets.pytorch_backend.nets_utils import pad_list
from espnet2.asr.espnet_model import ESPnetASRModel
from espnet2.enh.espnet_model import ESPnetEnhancementModel
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel


CLEAN_DATA = 1
REAL_DATA = 2
OTHER_DATA = 0


def fliter_attrs(a, b):
    a_attr = [attr for attr in a if attr[:2] != "__" and attr[-2:] != "__"]
    b_attr = [attr for attr in b if attr[:2] != "__" and attr[-2:] != "__"]
    a_attr_ = [i for i in a_attr if i not in b_attr]
    b_attr_ = [i for i in b_attr if i not in a_attr]
    return a_attr_, b_attr_


class ESPnetEnhASRModel(AbsESPnetModel):
    """Enhancement frontend with CTC-attention hybrid Encoder-Decoder model."""

    def __init__(
        self,
        enh_model: Optional[ESPnetEnhancementModel],
        asr_model: Optional[ESPnetASRModel],
        enh_weight: float = 0.5,
        cal_enh_loss: bool = True,
        end2end_train: bool = True,
        total_loss_scale: float = 1,
        tBPTT: bool = False,
        truncate_length: int = 32000,
        truncate_win_len: int = 10,
        truncate_win_shift: int = 10,
        truncate_center_mode: bool = False,
        truncate_slice_mode: str = "sample",
        enh_real_prob: float = 1.0,
    ):
        assert check_argument_types()
        assert 0.0 <= asr_model.ctc_weight <= 1.0, asr_model.ctc_weight
        assert 0.0 <= enh_weight <= 1.0, asr_model.ctc_weight
        assert asr_model.rnnt_decoder is None, "Not implemented"
        super().__init__()

        self.enh_subclass = enh_model
        self.asr_subclass = asr_model
        self.enh_weight = enh_weight
        self.cal_enh_loss = cal_enh_loss
        self.total_loss_scale = total_loss_scale

        # TODO(Jing): find out the -1 or 0 here
        # self.idx_blank = token_list.index(sym_blank) # 0
        self.idx_blank = -1
        self.num_spk = enh_model.num_spk
        if self.num_spk > 1:
            assert (
                asr_model.ctc_weight != 0.0 or cal_enh_loss
            )  # need at least one to cal PIT permutation

        # self.end2end_train = False
        self.end2end_train = end2end_train
        self.enh_attr = dir(enh_model)
        self.asr_attr = dir(asr_model)

        # Note(Jing): self delegation from the enh and asr sub-modules
        # fliter the specific attr for each subclass
        self.enh_attr, self.asr_attr = fliter_attrs(self.enh_attr, self.asr_attr)
        for arr in self.enh_attr:
            setattr(self, arr, getattr(self.enh_subclass, arr))
        for arr in self.asr_attr:
            setattr(self, arr, getattr(self.asr_subclass, arr))

        # (approximated) truncated back-propagation through time
        self.tBPTT = tBPTT
        self.truncate_length = truncate_length
        # assumed window length used in enh_model.encoder
        self.truncate_win_len = truncate_win_len
        # assumed window shift used in enh_model.encoder
        self.truncate_win_shift = truncate_win_shift
        # Whether to use center=True mode in enh_model.encoder
        self.truncate_center_mode = truncate_center_mode
        # "sample" for sample-level truncation; "frame" for encoder-level truncation
        self.truncate_slice_mode = truncate_slice_mode
        assert (
            self.truncate_slice_mode == "sample"
        ), "Currently only supporting sample-level truncation"

        self.truncate_win_overlap = self.truncate_win_len - self.truncate_win_shift
        self.truncated_frames = (
            self.truncate_length - self.truncate_win_overlap
        ) // self.truncate_win_shift

        # for multi-condition training with real data
        # if < 1.0, feed the real data only to the backend with probability
        self.enh_real_prob = enh_real_prob

    @staticmethod
    def get_pad_slice(
        start_sample, end_sample, full_length, win_length, hop_length, center=False
    ):
        """Calculate the required padding on both sides and the corresponding slice
        on the resultant encoded representation.

        Args:
            start_sample (int): index of the start sample
            end_sample (int): index of the end sample
            full_length (int): total length of the input sequence
            win_length (int): size of the sliding window
            hop_length (int): hop size of the sliding window
            center (bool): whether to apply the center mode with the sliding window
        Returns:
            pad_left (int): number of zeros to be padded to the left
            pad_right (int): number of zeros to be padded to the right
            pad_slice (slice): slice to be applied to the resultant encoded representation
        """
        overlap_length = win_length - hop_length
        if center:
            # num_frames = (len(x) + win_length - overlap_length) // hop_length
            start_index, pad_left = divmod(start_sample, hop_length)
            end_index, pad_right = divmod(end_sample + hop_length, hop_length)

            if start_sample >= win_length // 2:
                # pad some zeros (up to `n_pad` frames) on the left before STFT
                # later, STFT will pad (win_length // 2) zeros in both sides
                n_pad = ceil(win_length // 2 / hop_length)
                pad_left += n_pad * hop_length
                left_slice = n_pad
            else:
                # pad `start_sample` zeros to align with iSTFT(spec)
                pad_left = start_sample
                left_slice = start_index

            if (full_length - end_sample) >= win_length // 2:
                # pad some zeros (up to `n_pad` frames) on the right before STFT
                # later, STFT will pad (win_length // 2) zeros in both sides
                n_pad = ceil(win_length // 2 / hop_length)
                pad_right += n_pad * hop_length
            else:
                # pad `full_length - end_sample` zeros to align with iSTFT(spec)
                pad_right = full_length - end_sample
            right_slice = left_slice + end_index - start_index

            pad_slice = slice(left_slice, right_slice)
        else:
            # num_frames = (len(x) - overlap_length) // hop_length
            start_index, pad_left = divmod(start_sample, hop_length)
            end_index, pad_right = divmod(end_sample - overlap_length, hop_length)
            pad_slice = slice(0, end_index - start_index)
        return pad_left, pad_right, pad_slice

    def _truncate_speech(
        self,
        speech_mix: torch.Tensor,
        ilens: torch.Tensor,
        center: bool = False,
        slice_mode: str = "sample",
    ):
        """Truncate the input speech with given length and window info.

        Note: this can be used for truncated BPTT.

        Args:
            speech_mix (torch.Tensor): input speech (Batch, samples [, channels])
            ilens (torch.Tensor): input speech lengths (Batch,)
            truncate_length (int): length of the truncated chunk with gradient enabled
                NOTE: This may be adjusted slightly to match the exact length of X frames.
            truncate_win_len (int): assumed window length (in samples)
                                    used in the downstream encoder
                NOTE: This should be the final window length including padding.
            truncate_win_shift (int): assumed window shift (in samples)
                                    used in the downstream encoder
            center (bool): whether to apply center mode in the downstream encoder
                        (e.g. STFT, conv)
            slice_mode (str): one of ("sample", "frame")

        Returns:
            speech_truncated (torch.Tensor): truncated speech
            olens (torch.Tensor): truncated speech lengths
            truncate_length (int): the finally used truncate_length
            out_slice (List[slice]): the slice to be applied to the processed output
            offsets (torch.Tensor): expected output offsets to align the truncated
                speech with full-length speech after the downstream encoder
        """
        if self.truncate_length <= 0:
            raise ValueError(
                f"Invalid data length ({ilens}) or truncate_length "
                f"({self.truncate_length})"
            )
        assert slice_mode in ("sample", "frame"), slice_mode
        min_length = torch.min(ilens)
        truncate_win_overlap = self.truncate_win_len - self.truncate_win_shift
        to_add = self.truncate_win_len if center else 0
        if min_length < self.truncate_length:
            print(
                f"WARNING: sample is shorter than {self.truncate_length}, "
                f"fall back to min sample length ({min_length})",
                flush=True,
            )
            truncate_length = min_length
        else:
            truncate_length = self.truncate_length
        truncated_frames = (
            truncate_length - truncate_win_overlap + to_add
        ) // self.truncate_win_shift
        truncate_length = (
            truncated_frames * self.truncate_win_shift + truncate_win_overlap - to_add
        )
        speech_truncated = []
        olens = []
        offsets = ilens.new_zeros(ilens.size())
        out_slice = [slice(None) for _ in range(speech_mix.size(0))]
        for i, length in enumerate(ilens):
            if length == truncate_length:
                offsets[i] = 0
                speech_truncated.append(speech_mix[i, :length])
            else:
                total_frames = (
                    length - truncate_win_overlap + to_add
                ) // self.truncate_win_shift
                assert total_frames > truncated_frames, (total_frames, truncated_frames)
                offset = torch.randint(
                    size=(), low=0, high=total_frames - truncated_frames
                )
                idx = offset.item() * self.truncate_win_shift
                offsets[i] = offset if slice_mode == "frame" else idx
                pad_left, pad_right, pad_slice = self.get_pad_slice(
                    idx,
                    idx + truncate_length,
                    length,
                    self.truncate_win_len,
                    self.truncate_win_shift,
                    center=center,
                )
                start = idx - pad_left
                end = idx + truncate_length + pad_right
                assert idx - pad_left >= 0, (idx, pad_left)
                assert idx + truncate_length + pad_right <= length, (
                    idx + truncate_length,
                    pad_right,
                    length,
                )
                speech_truncated.append(speech_mix[i, start:end])
                olens.append(end - start)
                if center:
                    out_slice[i] = (
                        pad_slice
                        if slice_mode == "frame"
                        else slice(pad_left, pad_left + truncate_length)
                    )
        olens = torch.as_tensor(olens, dtype=torch.long, device=ilens.device)
        speech_truncated = pad_list(speech_truncated, 0).to(speech_mix.device)
        return speech_truncated, olens, truncate_length, out_slice, offsets

    def forward(
        self,
        speech_mix: torch.Tensor,
        speech_mix_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Enhancement + Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            text: (Batch, Length)
            text_lengths: (Batch,)
        """
        speech_ref = [
            kwargs["speech_ref{}".format(spk + 1)]
            if f"speech_ref{spk + 1}" in kwargs
            else None
            for spk in range(self.num_spk)
        ]
        text_ref = [kwargs["text_ref{}".format(spk + 1)] for spk in range(self.num_spk)]
        text_ref_lengths = [
            kwargs["text_ref{}_lengths".format(spk + 1)] for spk in range(self.num_spk)
        ]

        assert all(ref_lengths.dim() == 1 for ref_lengths in text_ref_lengths), (
            ref_lengths.shape for ref_lengths in text_ref_lengths
        )
        # Check that batch_size is unified
        batch_size = speech_mix.shape[0]
        assert batch_size == speech_mix_lengths.shape[0], (
            speech_mix.shape,
            speech_mix_lengths.shape,
        )
        assert all(
            it.shape[0] == batch_size for it in chain(text_ref, text_ref_lengths)
        ), (
            speech_mix.shape,
            (ref.shape for ref in text_ref),
            (ref_lengths.shape for ref_lengths in text_ref_lengths),
        )

        if "noise_ref1" in kwargs:
            # noise signal (optional, required when using
            # frontend models with beamformering)
            noise_ref = [
                kwargs["noise_ref{}".format(n + 1)] for n in range(self.num_noise_type)
            ]
        else:
            noise_ref = None

        # dereverberated noisy signal
        # (optional, only used for frontend models with WPE)
        if "dereverb_ref1" in kwargs:
            # noise signal (optional, required when using
            # frontend models with beamformering)
            dereverb_speech_ref = [
                kwargs["dereverb_ref{}".format(n + 1)]
                for n in range(self.num_spk)
                if "dereverb_ref{}".format(n + 1) in kwargs
            ]
            assert len(dereverb_speech_ref) in (1, self.num_spk), len(
                dereverb_speech_ref
            )
        else:
            dereverb_speech_ref = None

        if "utt2category" in kwargs:
            utt2category = kwargs["utt2category"][0]
        else:
            utt2category = 0

        # for data-parallel
        text_length_max = max(ref_lengths.max() for ref_lengths in text_ref_lengths)
        text_ref = [
            torch.cat(
                [
                    ref,
                    torch.ones(batch_size, text_length_max, dtype=ref.dtype).to(
                        ref.device
                    )
                    * self.idx_blank,
                ],
                dim=1,
            )[:, :text_length_max]
            for ref in text_ref
        ]

        # 0. Enhancement
        if utt2category == CLEAN_DATA:
            # clean data that should only pass through the speech recognition backend
            # TODO(Jing): find a better way to locate single-spk set
            # single-speaker case
            speech_pre_all = (
                speech_mix
                if speech_mix.dim() == 2
                else speech_mix[..., self.ref_channel]
            )
            speech_pre_lengths = speech_mix_lengths
            text_ref_all, text_ref_lengths = text_ref[0], text_ref_lengths[0]
            perm = True
            loss_enh = None
            n_speaker_asr = 1
        elif utt2category == REAL_DATA:
            if random.random() > self.enh_real_prob:
                # only feed real data to the backend and bypass the
                # speech enhancement frontend
                speech_pre_all = (
                    speech_mix
                    if speech_mix.dim() == 2
                    else speech_mix[..., self.ref_channel]
                )
                speech_pre_lengths = speech_mix_lengths
                text_ref_all, text_ref_lengths = text_ref[0], text_ref_lengths[0]
                perm = True
                loss_enh = None
                n_speaker_asr = 1
            else:
                # real data that should pass through the speech enhancement frontend
                # but without loss
                loss_enh, perm, speech_pre, speech_pre_lengths = self.forward_enh(
                    speech_mix,
                    speech_mix_lengths,
                    speech_ref=speech_ref,
                    noise_ref=noise_ref,
                    dereverb_speech_ref=dereverb_speech_ref,
                    cal_enh_loss=False,
                )
                # speech_pre: List[bs,T] --> (bs,num_spk,T)
                speech_pre = torch.stack(speech_pre, dim=1)
                if speech_pre[:, 0].dim() == speech_mix.dim():
                    # single-channel input
                    assert speech_pre[:, 0].shape == speech_mix.shape, (
                        speech_pre[:, 0].shape,
                        speech_mix.shape,
                    )
                    speech_frame_length = speech_mix.size(-1)
                else:
                    # multi-channel input
                    assert speech_pre[:, 0].shape == speech_mix[..., 0].shape, (
                        speech_pre[:, 0].shape,
                        speech_mix.shape,
                    )
                    speech_frame_length = speech_mix.size(-2)

                # Pack the separated speakers into the ASR part.
                speech_pre_all = (
                    speech_pre.transpose(0, 1)
                    .contiguous()
                    .view(-1, speech_frame_length)
                )  # (N_spk*B, T)
                speech_pre_lengths = torch.stack(
                    [speech_mix_lengths for _ in range(speech_pre.size(1))], dim=0
                ).view(-1)
                text_ref_all = torch.stack(text_ref, dim=1).view(
                    batch_size * len(text_ref), -1
                )
                text_ref_lengths = torch.stack(text_ref_lengths, dim=1).view(-1)
                n_speaker_asr = self.num_spk
                perm = None

        else:
            loss_enh, perm, speech_pre, speech_pre_lengths = self.forward_enh(
                speech_mix,
                speech_mix_lengths,
                speech_ref=speech_ref,
                noise_ref=noise_ref,
                dereverb_speech_ref=dereverb_speech_ref,
            )
            # speech_pre: List[bs,T] --> (bs,num_spk,T)
            speech_pre = torch.stack(speech_pre, dim=1)
            if speech_pre[:, 0].dim() == speech_mix.dim():
                # single-channel input
                assert speech_pre[:, 0].shape == speech_mix.shape, (
                    speech_pre[:, 0].shape,
                    speech_mix.shape,
                )
                speech_frame_length = speech_mix.size(-1)
            else:
                # multi-channel input
                assert speech_pre[:, 0].shape == speech_mix[..., 0].shape, (
                    speech_pre[:, 0].shape,
                    speech_mix.shape,
                )
                speech_frame_length = speech_mix.size(-2)

            if not self.end2end_train:
                # if the FrontEnd and ASR are trained independetly
                # use the speech_ref to train asr
                speech_pre = torch.stack(speech_ref, dim=1)

            # Pack the separated speakers into the ASR part.
            speech_pre_all = (
                speech_pre.transpose(0, 1).contiguous().view(-1, speech_frame_length)
            )  # (N_spk*B, T)
            speech_pre_lengths = torch.stack(
                [speech_mix_lengths for _ in range(speech_pre.size(1))], dim=0
            ).view(-1)
            text_ref_all = torch.stack(text_ref, dim=1).view(
                batch_size * len(text_ref), -1
            )
            text_ref_lengths = torch.stack(text_ref_lengths, dim=1).view(-1)
            n_speaker_asr = 1 if self.cal_enh_loss else self.num_spk

        # 1. Encoder
        encoder_out, encoder_out_lens = self.encode(speech_pre_all, speech_pre_lengths)

        # 2a. CTC branch
        if self.ctc_weight == 0.0:
            loss_ctc, cer_ctc = None, None
        else:
            if n_speaker_asr == 1 or (
                self.cal_enh_loss and perm is not None
            ):  # No permutation is required
                assert n_speaker_asr == 1
                loss_ctc, cer_ctc, _, _, = self._calc_ctc_loss_with_spk(
                    encoder_out,
                    encoder_out_lens,
                    text_ref_all,
                    text_ref_lengths,
                    n_speakers=n_speaker_asr,
                )
            else:  # Permutation is determined by CTC
                assert n_speaker_asr > 1
                (
                    loss_ctc,
                    cer_ctc,
                    encoder_out,
                    encoder_out_lens,
                ) = self._calc_ctc_loss_with_spk(
                    encoder_out,
                    encoder_out_lens,
                    text_ref_all,
                    text_ref_lengths,
                    n_speakers=self.num_spk,
                )

        # 2b. Attention-decoder branch
        if self.ctc_weight == 1.0:
            loss_att, acc_att, cer_att, wer_att = None, None, None, None
        else:
            loss_att, acc_att, cer_att, wer_att = self._calc_att_loss(
                encoder_out, encoder_out_lens, text_ref_all, text_ref_lengths
            )

        # 2c. RNN-T branch
        if self.rnnt_decoder is not None:
            _ = self._calc_rnnt_loss(
                encoder_out, encoder_out_lens, text_ref_all, text_ref_lengths
            )

        if self.ctc_weight == 0.0:
            loss_asr = loss_att
        elif self.ctc_weight == 1.0:
            loss_asr = loss_ctc
        else:
            loss_asr = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att

        if self.enh_weight == 0.0 or not self.cal_enh_loss or loss_enh is None:
            loss_enh = None
            loss = loss_asr
        else:
            loss = self.total_loss_scale * (loss_asr + self.enh_weight * loss_enh)

        stats = dict(
            loss=loss.detach(),
            loss_att=loss_att.detach() if loss_att is not None else None,
            loss_ctc=loss_ctc.detach() if loss_ctc is not None else None,
            loss_enh=loss_enh.detach() if loss_enh is not None else None,
            acc=acc_att,
            cer=cer_att,
            wer=wer_att,
            cer_ctc=cer_ctc,
        )

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def collect_feats(
        self, speech_mix: torch.Tensor, speech_mix_lengths: torch.Tensor, **kwargs
    ) -> Dict[str, torch.Tensor]:
        # for data-parallel
        speech_mix = speech_mix[:, : speech_mix_lengths.max()]
        feats, feats_lengths = speech_mix, speech_mix_lengths
        return {"feats": feats, "feats_lengths": feats_lengths}

    # Enhancement related, basicly from the espnet2/enh/espnet_model.py
    def forward_enh(
        self,
        speech_mix: torch.Tensor,
        speech_mix_lengths: torch.Tensor = None,
        speech_ref: List[torch.Tensor] = None,
        noise_ref: List[torch.Tensor] = None,
        dereverb_speech_ref: List[torch.Tensor] = None,
        resort_pre: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech_mix: (Batch, samples) or (Batch, samples, channels)
            speech_ref: num_speaker * (Batch, samples)
                        or num_speaker * (Batch, samples, channels)
            speech_mix_lengths: (Batch,), default None for chunk interator,
                            because the chunk-iterator does not have the
                            speech_lengths returned. see in
                            espnet2/iterators/chunk_iter_factory.py
        """
        # (Batch, num_speaker, samples) or (Batch, num_speaker, samples, channels)
        if speech_ref is not None and speech_ref[0] is not None:
            speech_ref = torch.stack(speech_ref, dim=1)
        else:
            speech_ref = None
        if noise_ref is not None and noise_ref[0] is not None:
            noise_ref = torch.stack(noise_ref, dim=1)
        else:
            noise_ref = None
        if dereverb_speech_ref is not None and dereverb_speech_ref[0] is not None:
            dereverb_speech_ref = torch.stack(dereverb_speech_ref, dim=1)
        else:
            dereverb_speech_ref = None

        cal_enh_loss = self.cal_enh_loss and kwargs.get("cal_enh_loss", True)
        if speech_ref is None and noise_ref is None and dereverb_speech_ref is None:
            # There is no ref provided, avoid the enh loss
            assert not cal_enh_loss, (
                "There is no reference,"
                "cal_enh_loss must be false, but {} given.".format(cal_enh_loss)
            )

        batch_size = speech_mix.shape[0]
        speech_lengths = (
            speech_mix_lengths
            if speech_mix_lengths is not None
            else torch.ones(batch_size).int() * speech_mix.shape[1]
        )
        assert speech_lengths.dim() == 1, speech_lengths.shape
        # Check that batch_size is unified
        if speech_ref is not None:
            assert (
                speech_mix.shape[0] == speech_ref.shape[0] == speech_lengths.shape[0]
            ), (
                speech_mix.shape,
                speech_ref.shape,
                speech_lengths.shape,
            )

        # for data-parallel
        speech_ref = (
            speech_ref[:, :, : speech_lengths.max()] if speech_ref is not None else None
        )
        speech_mix = speech_mix[:, : speech_lengths.max()]

        if self.tBPTT:
            with torch.no_grad():
                (
                    loss,
                    speech_pre,
                    others,
                    out_lengths,
                    perm,
                    stats_ret,
                ) = self.enh_subclass._compute_loss(
                    speech_mix,
                    speech_lengths,
                    speech_ref,
                    dereverb_speech_ref=dereverb_speech_ref,
                    noise_ref=noise_ref,
                    cal_loss=cal_enh_loss,
                )
            (
                speech_mix_truncate,
                speech_olengths,
                truncate_length,
                out_slice,
                offsets,
            ) = self._truncate_speech(
                speech_mix,
                speech_lengths,
                center=self.truncate_center_mode,
                slice_mode=self.truncate_slice_mode,
            )
            sample_offsets = offsets * self.truncate_win_shift
            if speech_ref is not None:
                speech_ref2 = speech_ref.new_empty(
                    [*speech_ref.shape[:2], truncate_length, *speech_ref.shape[3:]]
                )
                for b in range(len(speech_ref2)):
                    speech_ref2[b] = speech_ref[
                        b, :, sample_offsets[b] : sample_offsets[b] + truncate_length
                    ]
            else:
                speech_ref2 = None
            if dereverb_speech_ref is not None:
                dereverb_speech_ref2 = dereverb_speech_ref.new_empty(
                    [
                        *dereverb_speech_ref.shape[:2],
                        truncate_length,
                        *dereverb_speech_ref.shape[3:],
                    ]
                )
                for b in range(len(dereverb_speech_ref2)):
                    dereverb_speech_ref2[b] = dereverb_speech_ref[
                        b, :, sample_offsets[b] : sample_offsets[b] + truncate_length
                    ]
            else:
                dereverb_speech_ref2 = None
            if noise_ref is not None:
                noise_ref2 = noise_ref.new_empty(
                    [*noise_ref.shape[:2], truncate_length, *noise_ref.shape[3:]]
                )
                for b in range(len(noise_ref2)):
                    noise_ref2[b] = noise_ref[
                        b, :, sample_offsets[b] : sample_offsets[b] + truncate_length
                    ]
            else:
                noise_ref2 = None
            loss, speech_pre_truncate, *_ = self.enh_subclass._compute_loss(
                speech_mix_truncate,
                speech_olengths,
                speech_ref2,
                dereverb_speech_ref=dereverb_speech_ref2,
                noise_ref=noise_ref2,
                cal_loss=cal_enh_loss,
            )
            speech_pre = torch.stack(speech_pre, dim=1)
            speech_pre_truncate = torch.stack(speech_pre_truncate, dim=1)
            # replace corresponding region with trauncated prediction
            # (with backward graph)
            for b in range(len(speech_pre)):
                speech_pre[
                    b, :, sample_offsets[b] : sample_offsets[b] + truncate_length
                ] = speech_pre_truncate[b, :, out_slice[b]]
            speech_pre = speech_pre.unbind(dim=1)
        else:
            (
                loss,
                speech_pre,
                others,
                out_lengths,
                perm,
            ) = self.enh_subclass._compute_loss(
                speech_mix,
                speech_lengths,
                speech_ref,
                dereverb_speech_ref=dereverb_speech_ref,
                noise_ref=noise_ref,
                cal_loss=cal_enh_loss,
            )
        if self.enh_subclass.loss_type not in ("snr", "ci_sdr", "si_snr"):
            speech_pre = [
                self.enh_subclass.decoder(ps, speech_lengths)[0] for ps in speech_pre
            ]

        if resort_pre and perm is not None:
            # resort the prediction wav with the perm from enh_loss
            # speech_pre : List[(BS, ...)] of spk
            # perm : List[(num_spk)] of batch
            speech_pre_list = []
            for batch_idx, p in enumerate(perm):
                batch_list = []
                for spk_idx in p:
                    batch_list.append(speech_pre[spk_idx][batch_idx])  # spk,...
                speech_pre_list.append(torch.stack(batch_list, dim=0))

            speech_pre = torch.stack(speech_pre_list, dim=0)  # bs,num_spk,...
            speech_pre = torch.unbind(speech_pre, dim=1)  # list[(bs,...)] of spk
        else:
            # speech_pre = torch.stack(speech_pre, dim=1)  # bs,num_spk,...
            pass

        return loss, perm, speech_pre, out_lengths

    def _calc_ctc_loss_with_spk(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
        n_speakers: int = 1,
    ):
        # Calc CTC loss
        if n_speakers == 1:
            loss_ctc = self.ctc(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)
            # loss_ctc = loss_ctc.masked_fill(torch.isinf(loss_ctc), 0)
            loss_ctc = loss_ctc.mean()
        else:
            encoder_out, encoder_out_lens, ys_pad, ys_pad_lens = (
                torch.chunk(encoder_out, n_speakers, dim=0),
                torch.chunk(encoder_out_lens, n_speakers, dim=0),
                torch.chunk(ys_pad, n_speakers, dim=0),
                torch.chunk(ys_pad_lens, n_speakers, dim=0),
            )
            batch_size = encoder_out[0].size(0)
            loss_ctc = torch.stack(
                [
                    torch.stack(
                        [
                            self.ctc(
                                encoder_out[h],
                                encoder_out_lens[h],
                                ys_pad[r],
                                ys_pad_lens[r],
                            )
                            for r in range(n_speakers)
                        ],
                        dim=1,
                    )
                    for h in range(n_speakers)
                ],
                dim=2,
            )  # (B, n_ref, n_hyp)
            # loss_ctc = loss_ctc.masked_fill(torch.isinf(loss_ctc), 0)
            perm_detail, min_loss_ctc = self.permutation_invariant_training(loss_ctc)
            loss_ctc = min_loss_ctc.mean()
            # permutate the encoder_out
            encoder_out, encoder_out_lens = (
                torch.stack(encoder_out, dim=1),
                torch.stack(encoder_out_lens, dim=1),
            )  # (B, n_spk, T, D)
            for b in range(batch_size):
                encoder_out[b] = encoder_out[b, perm_detail[b]]
                encoder_out_lens[b] = encoder_out_lens[b, perm_detail[b]]
            encoder_out = torch.cat(
                [encoder_out[:, i] for i in range(n_speakers)], dim=0
            )
            encoder_out_lens = torch.cat(
                [encoder_out_lens[:, i] for i in range(n_speakers)], dim=0
            )
            ys_pad = torch.cat(ys_pad, dim=0)

        # Calc CER using CTC
        cer_ctc = None
        if not self.training and self.error_calculator is not None:
            ys_hat = self.ctc.argmax(encoder_out).data
            cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
        return loss_ctc, cer_ctc, encoder_out, encoder_out_lens

    def _permutation_loss(self, ref, inf, criterion, perm=None):
        """The basic permutation loss function.

        Args:
            ref (List[torch.Tensor]): [(batch, ...), ...] x n_spk
            inf (List[torch.Tensor]): [(batch, ...), ...]
            criterion (function): Loss function
            perm: (batch)
        Returns:
            loss: torch.Tensor: (batch)
            perm: list[(num_spk)]
        """
        num_spk = len(ref)

        losses = torch.stack(
            [
                torch.stack([criterion(ref[r], inf[h]) for r in range(num_spk)], dim=1)
                for h in range(num_spk)
            ],
            dim=2,
        )  # (B, n_ref, n_hyp)
        perm_detail, min_loss = self.permutation_invariant_training(losses)

        return min_loss.mean(), perm_detail

    def permutation_invariant_training(self, losses: torch.Tensor):
        """Compute  PIT loss.

        Args:
            losses (torch.Tensor): (batch, nref, nhyp)
        Returns:
            perm: list: (batch, n_spk)
            loss: torch.Tensor: (batch)
        """
        hyp_perm, min_perm_loss = [], []
        losses_cpu = losses.data.cpu()
        for b, b_loss in enumerate(losses_cpu):
            # hungarian algorithm
            try:
                row_ind, col_ind = linear_sum_assignment(b_loss)
            except ValueError as err:
                if str(err) == "cost matrix is infeasible":
                    # random assignment since the cost is always inf
                    col_ind = np.array([0, 1])
                    min_perm_loss.append(torch.mean(losses[b, col_ind, col_ind]))
                    hyp_perm.append(col_ind)
                    continue
                else:
                    raise

            min_perm_loss.append(torch.mean(losses[b, row_ind, col_ind]))
            hyp_perm.append(col_ind)

        return hyp_perm, torch.stack(min_perm_loss)
