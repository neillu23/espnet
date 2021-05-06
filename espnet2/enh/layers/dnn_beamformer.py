from distutils.version import LooseVersion
from typing import List
from typing import Tuple
from typing import Union

import logging
import torch
from torch.nn import functional as F
from torch_complex import functional as FC
from torch_complex.tensor import ComplexTensor

from espnet.nets.pytorch_backend.frontends.beamformer import apply_beamforming_vector
from espnet.nets.pytorch_backend.frontends.beamformer import (
    get_power_spectral_density_matrix,  # noqa: H301
)
from espnet2.enh.layers.beamformer import diagonal
from espnet2.enh.layers.beamformer import get_covariances
from espnet2.enh.layers.beamformer import get_mvdr_vector
from espnet2.enh.layers.beamformer import get_mvdr_vector_with_rtf
from espnet2.enh.layers.beamformer import get_WPD_filter_v2
from espnet2.enh.layers.beamformer import get_WPD_filter_with_rtf
from espnet2.enh.layers.beamformer import perform_WPD_filtering
from espnet2.enh.layers.mask_estimator import MaskEstimator

EPS = torch.finfo(torch.double).eps
is_torch_1_2_plus = LooseVersion(torch.__version__) >= LooseVersion("1.2.0")
is_torch_1_3_plus = LooseVersion(torch.__version__) >= LooseVersion("1.3.0")


BEAMFORMER_TYPES = (
    # Minimum Variance Distortionless Response beamformer
    "mvdr",  # RTF-based formula
    "mvdr_souden",  # Souden's solution
    # Minimum Power Distortionless Response beamformer
    "mpdr",  # RTF-based formula
    "mpdr_souden",  # Souden's solution
    # weighted MPDR beamformer
    "wmpdr",  # RTF-based formula
    "wmpdr_souden",  # Souden's solution
    # Weighted Power minimization Distortionless response beamformer
    "wpd",  # RTF-based formula
    "wpd_souden",  # Souden's solution
)


class DNN_Beamformer(torch.nn.Module):
    """DNN mask based Beamformer.

    Citation:
        Multichannel End-to-end Speech Recognition; T. Ochiai et al., 2017;
        http://proceedings.mlr.press/v70/ochiai17a/ochiai17a.pdf

    """

    def __init__(
        self,
        bidim,
        btype: str = "blstmp",
        blayers: int = 3,
        bunits: int = 300,
        bprojs: int = 320,
        num_spk: int = 1,
        use_noise_mask: bool = True,
        nonlinear: str = "sigmoid",
        dropout_rate: float = 0.0,
        badim: int = 320,
        ref_channel: int = -1,
        beamformer_type: str = "mvdr_souden",
        rtf_iterations: int = 2,
        eps: float = 1e-6,
        diagonal_loading: bool = True,
        diag_eps: float = 1e-7,
        mask_flooring: bool = False,
        flooring_thres: float = 1e-6,
        use_torch_solver: bool = True,
        # only for WPD beamformer
        btaps: int = 5,
        bdelay: int = 3,
    ):
        super().__init__()
        bnmask = num_spk + 1 if use_noise_mask else num_spk
        self.mask = MaskEstimator(
            btype,
            bidim,
            blayers,
            bunits,
            bprojs,
            dropout_rate,
            nmask=bnmask,
            nonlinear=nonlinear,
        )
        self.ref = AttentionReference(bidim, badim) if ref_channel < 0 else None
        self.ref_channel = ref_channel

        self.use_noise_mask = use_noise_mask
        assert num_spk >= 1, num_spk
        self.num_spk = num_spk
        self.nmask = bnmask

        if beamformer_type not in BEAMFORMER_TYPES:
            raise ValueError("Not supporting beamformer_type=%s" % beamformer_type)
        if (
            beamformer_type == "mvdr_souden" or not beamformer_type.endswith("_souden")
        ) and not use_noise_mask:
            if num_spk == 1:
                logging.warning(
                    "Initializing %s beamformer without noise mask "
                    "estimator (single-speaker case)" % beamformer_type.upper()
                )
                logging.warning(
                    "(1 - speech_mask) will be used for estimating noise "
                    "PSD in %s beamformer!" % beamformer_type.upper()
                )
            else:
                logging.warning(
                    "Initializing %s beamformer without noise mask "
                    "estimator (multi-speaker case)" % beamformer_type.upper()
                )
                logging.warning(
                    "Interference speech masks will be used for estimating "
                    "noise PSD in %s beamformer!" % beamformer_type.upper()
                )

        self.beamformer_type = beamformer_type
        if not beamformer_type.endswith("_souden"):
            assert rtf_iterations >= 2, rtf_iterations
        # number of iterations in power method for estimating the RTF
        self.rtf_iterations = rtf_iterations

        assert btaps >= 0 and bdelay >= 0, (btaps, bdelay)
        self.btaps = btaps
        self.bdelay = bdelay if self.btaps > 0 else 1
        self.eps = eps
        self.diagonal_loading = diagonal_loading
        self.diag_eps = diag_eps
        self.mask_flooring = mask_flooring
        self.flooring_thres = flooring_thres
        self.use_torch_solver = use_torch_solver
        # for mask binarization during inference
        self.binarize_thres = None

    def forward(
        self,
        data: ComplexTensor,
        ilens: torch.LongTensor,
        powers: Union[List[torch.Tensor], None] = None,
        # for SNR-based channel selection
        snr_selection: bool = False,
        # for utilizing external masks
        speech_masks: Union[List[torch.Tensor], None] = None,
        # for combining external masks and estimated masks
        mask_fusion: bool = False,
        # if both are provided, the reference signals will be used
        # to directly estiamte the beamforming filter (not mask-based)
        speech_ref: Union[ComplexTensor, None] = None,
        noise_ref: Union[ComplexTensor, None] = None,
    ) -> Tuple[ComplexTensor, torch.LongTensor, torch.Tensor]:
        """DNN_Beamformer forward function.

        Notation:
            B: Batch
            C: Channel
            T: Time or Sequence length
            F: Freq

        Args:
            data (ComplexTensor): (B, T, C, F)
            ilens (torch.Tensor): (B,)
            powers (List[torch.Tensor] or None): used for wMPDR or WPD (B, F, T)
            snr_selection (bool): Whether to use dynamic reference channel
                                  selection based on a posteriori SNR
            speech_masks (List[torch.Tensor] or None): given speech masks (B, F, C, T)
            mask_fusion (bool): If True and speech_masks is not None, the estimated
                                masks will be combined with the given speech_masks
            speech_ref (ComplexTensor or None): clean speech spectrum (B, T, C F)
                                                e.g. oracle speech
            noise_ref (ComplexTensor or None): noise spectrum (B, T, C F)
                                                e.g. oracle noise

        Returns:
            enhanced (ComplexTensor): (B, T, F)
            ilens (torch.Tensor): (B,)
            masks (torch.Tensor): (B, T, C, F)
            prior_snrs (torch.Tensor): a priori SNRs List[(B,)]
            post_snrs (torch.Tensor): a posteriori SNRs List[(B,)]
        """

        def apply_beamforming(data, ilens, psd_n, psd_speech, psd_distortion=None):
            """Beamforming with the provided statistics.

            Args:
                data (ComplexTensor): (B, F, C, T)
                ilens (torch.Tensor): (B,)
                psd_n (ComplexTensor):
                    Noise covariance matrix for MVDR (B, F, C, C)
                    Observation covariance matrix for MPDR/wMPDR (B, F, C, C)
                    Stacked observation covariance for WPD (B,F,(btaps+1)*C,(btaps+1)*C)
                psd_speech (ComplexTensor): Speech covariance matrix (B, F, C, C)
                psd_distortion (ComplexTensor): Noise covariance matrix (B, F, C, C)
            Return:
                enhanced (ComplexTensor): (B, F, T)
                ws (ComplexTensor): (B, F) or (B, F, (btaps+1)*C)
                prior_snrs (torch.Tensor): a priori SNRs List[(B,)]
                post_snrs (torch.Tensor): a posteriori SNRs List[(B,)]
            """
            # u: (B, C)
            if self.training or not snr_selection:
                if self.ref_channel < 0:
                    u, _ = self.ref(psd_speech.to(dtype=data.dtype), ilens)
                    u = u.double()
                elif self.beamformer_type.endswith("_souden"):
                    # (optional) Create onehot vector for fixed reference microphone
                    u = torch.zeros(
                        *(data.size()[:-3] + (data.size(-2),)),
                        device=data.device,
                        dtype=torch.double
                    )
                    u[..., self.ref_channel].fill_(1)
                else:
                    # for simplifying computation in RTF-based beamforming
                    u = self.ref_channel

            if self.beamformer_type in ("mvdr", "mpdr", "wmpdr"):
                ws = get_mvdr_vector_with_rtf(
                    psd_n.double(),
                    psd_speech.double(),
                    psd_distortion.double(),
                    iterations=self.rtf_iterations,
                    reference_vector=u,
                    normalize_ref_channel=self.ref_channel,
                    use_torch_solver=self.use_torch_solver,
                    diagonal_loading=self.diagonal_loading,
                    diag_eps=self.diag_eps,
                )
                enhanced = apply_beamforming_vector(ws, data.double())
            elif self.beamformer_type in ("mpdr_souden", "mvdr_souden", "wmpdr_souden"):
                ws = get_mvdr_vector(
                    psd_speech.double(),
                    psd_n.double(),
                    u,
                    use_torch_solver=self.use_torch_solver,
                    diagonal_loading=self.diagonal_loading,
                    diag_eps=self.diag_eps,
                )
                enhanced = apply_beamforming_vector(ws, data.double())
            elif self.beamformer_type == "wpd":
                ws = get_WPD_filter_with_rtf(
                    psd_n.double(),
                    psd_speech.double(),
                    psd_distortion.double(),
                    iterations=self.rtf_iterations,
                    reference_vector=u,
                    normalize_ref_channel=self.ref_channel,
                    use_torch_solver=self.use_torch_solver,
                    diagonal_loading=self.diagonal_loading,
                    diag_eps=self.diag_eps,
                )
                enhanced = perform_WPD_filtering(
                    ws, data.double(), self.bdelay, self.btaps
                )
            elif self.beamformer_type == "wpd_souden":
                ws = get_WPD_filter_v2(
                    psd_speech.double(),
                    psd_n.double(),
                    u,
                    diagonal_loading=self.diagonal_loading,
                    diag_eps=self.diag_eps,
                )
                enhanced = perform_WPD_filtering(
                    ws, data.double(), self.bdelay, self.btaps
                )
            else:
                raise ValueError(
                    "Not supporting beamformer_type={}".format(self.beamformer_type)
                )
            prior_SNR = None
            post_SNR = None
            if psd_distortion is not None and self.beamformer_type not in (
                "wpd",
                "wpd_souden",
            ):
                with torch.no_grad():
                    # (B, F, C, C) -> (B, F, C)
                    numerator = diagonal(psd_speech)
                    denominator = diagonal(psd_distortion)
                    prior_SNR = numerator[..., self.ref_channel].abs().sum(dim=-1) / (
                        denominator[..., self.ref_channel].abs().sum(dim=-1) + EPS
                    )
                    numerator = numerator * ws.conj() * ws
                    denominator = denominator * ws.conj() * ws
                    post_SNR = numerator.abs().sum(dim=(-1, -2)) / (
                        denominator.abs().sum(dim=(-1, -2)) + EPS
                    )
            else:
                # Perform SNR-based reference channel selection
                # Reference:
                #   Improved MVDR beamforming using single-channel mask prediction
                #   networks; Interspeech 2016
                channels = data.size(-2)
                enh, wss = [], []
                prior_SNRs, post_SNRs = [], []
                for c in range(channels):
                    if self.beamformer_type.endswith("_souden"):
                        # (optional) Create onehot vector for fixed reference microphone
                        u = torch.zeros(
                            *(data.size()[:-3] + (data.size(-2),)),
                            device=data.device,
                            dtype=torch.double
                        )
                        u[..., c].fill_(1)
                    else:
                        # for simplifying computation in RTF-based beamforming
                        u = c

                    if self.beamformer_type in ("mvdr", "mpdr", "wmpdr"):
                        # (B, F, C)
                        ws_ = get_mvdr_vector_with_rtf(
                            psd_n.double(),
                            psd_speech.double(),
                            psd_distortion.double(),
                            iterations=self.rtf_iterations,
                            reference_vector=u,
                            normalize_ref_channel=self.ref_channel,
                            use_torch_solver=self.use_torch_solver,
                            diagonal_loading=self.diagonal_loading,
                            diag_eps=self.diag_eps,
                        )
                        enhanced_ = apply_beamforming_vector(ws_, data.double())
                    elif self.beamformer_type in (
                        "mpdr_souden",
                        "mvdr_souden",
                        "wmpdr_souden",
                    ):
                        ws_ = get_mvdr_vector(
                            psd_speech.double(),
                            psd_n.double(),
                            u,
                            use_torch_solver=self.use_torch_solver,
                            diagonal_loading=self.diagonal_loading,
                            diag_eps=self.diag_eps,
                        )
                        enhanced_ = apply_beamforming_vector(ws_, data.double())
                    else:
                        raise ValueError(
                            "Not supporting beamformer_type={}".format(
                                self.beamformer_type
                            )
                        )
                    with torch.no_grad():
                        # (B, F, C, C) -> (B, F, C)
                        numerator = diagonal(psd_speech)
                        denominator = diagonal(psd_distortion)
                        prior_snr_ = numerator[..., self.ref_channel].abs().sum(
                            dim=-1
                        ) / (denominator[..., self.ref_channel].abs().sum(dim=-1) + EPS)
                        numerator = numerator * ws_.conj() * ws_
                        denominator = denominator * ws_.conj() * ws_
                        post_snr_ = numerator.abs().sum(dim=(-1, -2)) / (
                            denominator.abs().sum(dim=(-1, -2)) + EPS
                        )
                    enh.append(enhanced_)
                    wss.append(ws_)
                    prior_SNRs.append(prior_snr_)
                    post_SNRs.append(post_snr_)

                # (B,)
                post_SNR, ref_channels = torch.stack(post_SNRs, dim=1).max(dim=1)
                prior_SNR = torch.stack(
                    [
                        prior_SNRs[ref_channels[b]][b]
                        for b in range(ref_channels.size(0))
                    ],
                    dim=0,
                )
                enhanced = FC.stack(
                    [enh[ref_channels[b]][b] for b in range(ref_channels.size(0))],
                    dim=0,
                )
                ws = FC.stack(
                    [wss[ref_channels[b]][b] for b in range(ref_channels.size(0))],
                    dim=0,
                )

            return (
                enhanced.to(dtype=data.dtype),
                ws.to(dtype=data.dtype),
                prior_SNR,
                post_SNR,
            )

        # data (B, T, C, F) -> (B, F, C, T)
        data = data.permute(0, 3, 2, 1)
        data_d = data.double()

        has_ref_signal = speech_ref is not None and noise_ref is not None
        if has_ref_signal:
            masks = None
            speech_ref = speech_ref.permute(0, 3, 2, 1).double()
            noise_ref = noise_ref.permute(0, 3, 2, 1).double()
            assert self.num_spk == 1, "Only support single-speaker case"
        elif speech_masks is not None:
            # mask: [(B, F, C, T)]
            if mask_fusion:
                masks, _ = self.mask(data, ilens)
                if self.nmask == self.num_spk:
                    assert len(masks) == len(speech_masks), (
                        len(masks),
                        len(speech_masks),
                    )
                else:
                    assert len(masks) == len(speech_masks) + 1, (
                        len(masks),
                        len(speech_masks),
                    )
                if self.binarize_thres is not None:
                    # binarize masks
                    ones = masks[0].new_ones(1)
                    zeros = masks[0].new_zeros(1)
                    masks = [
                        torch.where(m > self.binarize_thres, ones, zeros) for m in masks
                    ]
                    speech_masks = [
                        torch.where(m > self.binarize_thres, ones, zeros)
                        for m in speech_masks
                    ]
                    half_channels = masks[0].shape[2] // 2
                    masks_ = [
                        torch.where(
                            (masks[spk] + speech_masks[spk]).sum(dim=2, keepdim=True)
                            >= half_channels,
                            ones,
                            zeros,
                        )
                        for spk in range(len(masks))
                    ]
                    if self.nmask == self.num_spk:
                        masks = masks_
                    else:
                        masks = masks_ + [masks[-1]]
                else:
                    # taking the maximum value in two masks
                    masks_ = [
                        torch.max(masks[spk], sm) for spk, sm in enumerate(speech_masks)
                    ]
                    if self.nmask == self.num_spk:
                        masks = masks_
                    else:
                        masks = masks_ + [masks[-1]]
            elif self.nmask == self.num_spk:
                masks = speech_masks
            else:
                masks, _ = self.mask(data, ilens)
                masks = list(speech_masks) + [masks[-1]]
        else:
            masks, _ = self.mask(data, ilens)

        if not has_ref_signal:
            assert self.nmask == len(masks), len(masks)
            # floor masks to increase numerical stability
            if self.mask_flooring and self.binarize_thres is None:
                masks = [torch.clamp(m, min=self.flooring_thres) for m in masks]

        if self.num_spk == 1:  # single-speaker case
            if not has_ref_signal:
                if self.use_noise_mask:
                    # (mask_speech, mask_noise)
                    mask_speech, mask_noise = masks
                else:
                    # (mask_speech,)
                    mask_speech = masks[0]
                    mask_noise = 1 - mask_speech

            if self.beamformer_type.startswith(
                "wmpdr"
            ) or self.beamformer_type.startswith("wpd"):
                if powers is None:
                    power_input = data_d.real ** 2 + data_d.imag ** 2
                    # Averaging along the channel axis: (..., C, T) -> (..., T)
                    powers = (power_input * mask_speech.double()).mean(dim=-2)
                else:
                    assert len(powers) == 1, len(powers)
                    powers = powers[0]
                inverse_power = 1 / torch.clamp(powers, min=self.eps)

            if has_ref_signal:
                psd_speech = FC.einsum(
                    "...ct,...et->...ce", [speech_ref, speech_ref.conj()]
                )
            else:
                psd_speech = get_power_spectral_density_matrix(
                    data_d, mask_speech.double()
                )
            if (
                (has_ref_signal or mask_noise is not None)
                and (
                    self.beamformer_type == "mvdr_souden"
                    or not self.beamformer_type.endswith("_souden")
                )
                or not self.training
            ):
                # MVDR or other RTF-based formulas
                if has_ref_signal:
                    psd_noise = FC.einsum(
                        "...ct,...et->...ce", [noise_ref, noise_ref.conj()]
                    )
                else:
                    psd_noise = get_power_spectral_density_matrix(
                        data_d, mask_noise.double()
                    )
            else:
                psd_noise = None
            if self.beamformer_type in ("mvdr", "mvdr_souden"):
                enhanced, ws, prior_snrs, post_snrs = apply_beamforming(
                    data, ilens, psd_noise, psd_speech, psd_distortion=psd_noise
                )
            elif self.beamformer_type in ("mpdr", "mpdr_souden"):
                psd_observed = FC.einsum("...ct,...et->...ce", [data_d, data_d.conj()])
                enhanced, ws, prior_snrs, post_snrs = apply_beamforming(
                    data, ilens, psd_observed, psd_speech, psd_distortion=psd_noise
                )
            elif self.beamformer_type in ("wmpdr", "wmpdr_souden"):
                psd_observed = FC.einsum(
                    "...ct,...et->...ce",
                    [data_d * inverse_power[..., None, :], data_d.conj()],
                )
                enhanced, ws, prior_snrs, post_snrs = apply_beamforming(
                    data, ilens, psd_observed, psd_speech, psd_distortion=psd_noise
                )
            elif self.beamformer_type in ("wpd", "wpd_souden"):
                psd_observed_bar = get_covariances(
                    data_d, inverse_power, self.bdelay, self.btaps, get_vector=False
                )
                enhanced, ws, prior_snrs, post_snrs = apply_beamforming(
                    data, ilens, psd_observed_bar, psd_speech, psd_distortion=psd_noise
                )
            else:
                raise ValueError(
                    "Not supporting beamformer_type={}".format(self.beamformer_type)
                )

            # (..., F, T) -> (..., T, F)
            enhanced = enhanced.transpose(-1, -2)
        else:  # multi-speaker case
            if self.use_noise_mask:
                # (mask_speech1, ..., mask_noise)
                mask_speech = list(masks[:-1])
                mask_noise = masks[-1]
            else:
                # (mask_speech1, ..., mask_speechX)
                mask_speech = list(masks)
                mask_noise = None

            if self.beamformer_type.startswith(
                "wmpdr"
            ) or self.beamformer_type.startswith("wpd"):
                if powers is None:
                    power_input = data_d.real ** 2 + data_d.imag ** 2
                    # Averaging along the channel axis: (..., C, T) -> (..., T)
                    powers = [
                        (power_input * m.double()).mean(dim=-2) for m in mask_speech
                    ]
                else:
                    assert len(powers) == self.num_spk, len(powers)
                inverse_power = [1 / torch.clamp(p, min=self.eps) for p in powers]

            psd_speeches = [
                get_power_spectral_density_matrix(data_d, mask.double())
                for mask in mask_speech
            ]
            if mask_noise is not None and (
                (
                    self.beamformer_type == "mvdr_souden"
                    or not self.beamformer_type.endswith("_souden")
                )
                or not self.training
            ):
                # MVDR or other RTF-based formulas
                psd_noise = get_power_spectral_density_matrix(
                    data_d, mask_noise.double()
                )
            else:
                psd_noise = None
            if self.beamformer_type in ("mpdr", "mpdr_souden"):
                psd_observed = FC.einsum("...ct,...et->...ce", [data_d, data_d.conj()])
            elif self.beamformer_type in ("wmpdr", "wmpdr_souden"):
                psd_observed = [
                    FC.einsum(
                        "...ct,...et->...ce",
                        [data_d * inv_p[..., None, :], data_d.conj()],
                    )
                    for inv_p in inverse_power
                ]
            elif self.beamformer_type in ("wpd", "wpd_souden"):
                psd_observed_bar = [
                    get_covariances(
                        data_d, inv_p, self.bdelay, self.btaps, get_vector=False
                    )
                    for inv_p in inverse_power
                ]

            enhanced, ws, prior_snrs, post_snrs = [], [], [], []
            for i in range(self.num_spk):
                psd_speech = psd_speeches.pop(i)
                if (
                    self.beamformer_type == "mvdr_souden"
                    or not self.beamformer_type.endswith("_souden")
                ) or not self.training:
                    psd_noise_i = (
                        psd_noise + sum(psd_speeches)
                        if mask_noise is not None
                        else sum(psd_speeches)
                    )
                else:
                    psd_noise_i = None
                # treat all other speakers' psd_speech as noises
                if self.beamformer_type in ("mvdr", "mvdr_souden"):
                    enh, w, prior_snr, post_snr = apply_beamforming(
                        data, ilens, psd_noise_i, psd_speech, psd_distortion=psd_noise_i
                    )
                elif self.beamformer_type in ("mpdr", "mpdr_souden"):
                    enh, w, prior_snr, post_snr = apply_beamforming(
                        data,
                        ilens,
                        psd_observed,
                        psd_speech,
                        psd_distortion=psd_noise_i,
                    )
                elif self.beamformer_type in ("wmpdr", "wmpdr_souden"):
                    enh, w, prior_snr, post_snr = apply_beamforming(
                        data,
                        ilens,
                        psd_observed[i],
                        psd_speech,
                        psd_distortion=psd_noise_i,
                    )
                elif self.beamformer_type in ("wpd", "wpd_souden"):
                    enh, w, prior_snr, post_snr = apply_beamforming(
                        data,
                        ilens,
                        psd_observed_bar[i],
                        psd_speech,
                        psd_distortion=psd_noise_i,
                    )
                else:
                    raise ValueError(
                        "Not supporting beamformer_type={}".format(self.beamformer_type)
                    )
                psd_speeches.insert(i, psd_speech)

                # (..., F, T) -> (..., T, F)
                enh = enh.transpose(-1, -2)
                enhanced.append(enh)
                ws.append(w)
                prior_snrs.append(prior_snr)
                post_snrs.append(post_snr)

        # (..., F, C, T) -> (..., T, C, F)
        if not has_ref_signal:
            masks = [m.transpose(-1, -3) for m in masks]
        return enhanced, ilens, masks, prior_snrs, post_snrs

    def predict_mask(
        self, data: ComplexTensor, ilens: torch.LongTensor
    ) -> Tuple[Tuple[torch.Tensor, ...], torch.LongTensor]:
        """Predict masks for beamforming.

        Args:
            data (ComplexTensor): (B, T, C, F), double precision
            ilens (torch.Tensor): (B,)
        Returns:
            masks (torch.Tensor): (B, T, C, F)
            ilens (torch.Tensor): (B,)
        """
        masks, _ = self.mask(data.permute(0, 3, 2, 1).float(), ilens)
        # (B, F, C, T) -> (B, T, C, F)
        masks = [m.transpose(-1, -3) for m in masks]
        return masks, ilens


class AttentionReference(torch.nn.Module):
    def __init__(self, bidim, att_dim):
        super().__init__()
        self.mlp_psd = torch.nn.Linear(bidim, att_dim)
        self.gvec = torch.nn.Linear(att_dim, 1)

    def forward(
        self, psd_in: ComplexTensor, ilens: torch.LongTensor, scaling: float = 2.0
    ) -> Tuple[torch.Tensor, torch.LongTensor]:
        """Attention-based reference forward function.

        Args:
            psd_in (ComplexTensor): (B, F, C, C)
            ilens (torch.Tensor): (B,)
            scaling (float):
        Returns:
            u (torch.Tensor): (B, C)
            ilens (torch.Tensor): (B,)
        """
        B, _, C = psd_in.size()[:3]
        assert psd_in.size(2) == psd_in.size(3), psd_in.size()
        # psd_in: (B, F, C, C)
        datatype = torch.bool if is_torch_1_3_plus else torch.uint8
        datatype2 = torch.bool if is_torch_1_2_plus else torch.uint8
        psd = psd_in.masked_fill(
            torch.eye(C, dtype=datatype, device=psd_in.device).type(datatype2), 0
        )
        # psd: (B, F, C, C) -> (B, C, F)
        psd = (psd.sum(dim=-1) / (C - 1)).transpose(-1, -2)

        # Calculate amplitude
        psd_feat = (psd.real ** 2 + psd.imag ** 2) ** 0.5

        # (B, C, F) -> (B, C, F2)
        mlp_psd = self.mlp_psd(psd_feat)
        # (B, C, F2) -> (B, C, 1) -> (B, C)
        e = self.gvec(torch.tanh(mlp_psd)).squeeze(-1)
        u = F.softmax(scaling * e, dim=-1)
        return u, ilens
