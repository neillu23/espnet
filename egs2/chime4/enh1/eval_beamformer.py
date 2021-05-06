import argparse
from collections import defaultdict

# import IPython
from mir_eval.separation import bss_eval_sources
import numpy as np
from pathlib import Path
import time

# from pb_bss.evaluation import InputMetrics
# from pb_bss.evaluation import OutputMetrics
from pesq import pesq
from pystoi import stoi
import soundfile as sf
import torch
from tqdm import tqdm

from espnet2.enh.layers.dnn_beamformer import BEAMFORMER_TYPES
from espnet2.enh.layers.dnn_wpe import DNN_WPE
from espnet2.tasks.enh import EnhancementTask
from espnet2.utils.types import str2bool

from vad import compute_vad


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_file", type=str, required=True, help="/path/to/model/xxx.pth"
    )
    parser.add_argument(
        "--train_config", type=str, required=True, help="/path/to/model/config.yaml"
    )
    parser.add_argument("--wavscp", type=str, required=True, help="Path to wav.scp")
    parser.add_argument("--spkscp", type=str, default=None, help="Path to spk1.scp")
    parser.add_argument(
        "--test_only",
        type=str2bool,
        default=False,
        help="If False, spkscp is needed for evaluation",
    )

    # post-processing related
    parser.add_argument(
        "--post_enh_model", type=str, default=None, help="/path/to/model/xxx.pth"
    )
    parser.add_argument(
        "--post_enh_train_config",
        type=str,
        default=None,
        help="/path/to/model/config.yaml",
    )
    parser.add_argument(
        "--do_post_processing",
        type=str2bool,
        default=False,
        help="True to use args.post_enh_model to enhance the beamformed signal",
    )
    parser.add_argument(
        "--beam_tasnet",
        type=str2bool,
        default=False,
        help="True to use MC-TasNet output signals for beamformer filter estimation",
    )
    parser.add_argument(
        "--tasnet_vad",
        type=str2bool,
        default=False,
        help="True to use TasNet output signals for VAD-based post-processing "
        "on beamformer output",
    )
    parser.add_argument(
        "--vad_mode",
        type=int,
        default=2,
        choices=(0, 1, 2, 3),
        help="aggressiveness mode for webrtcvad",
    )
    parser.add_argument(
        "--do_pre_masking",
        type=str2bool,
        default=False,
        help="True to use args.post_enh_model to estimate speech masks for beamforming,"
        " instead of using the mask estimator in the beamformer model",
    )
    parser.add_argument(
        "--mask_fusion",
        type=str2bool,
        default=False,
        help="When both this option args.do_pre_masking are True, "
        "use args.post_enh_model to estimate speech masks and combine "
        "them with the estimated masks in beamformer.",
    )
    parser.add_argument(
        "--binarize_thres",
        type=float,
        default=None,
        help="Used to perform mask binarization when doing mask fusion",
    )

    # output related
    parser.add_argument(
        "--outdir", type=str, required=True, help="Directory to store result files"
    )
    parser.add_argument(
        "--write_wavs",
        type=str2bool,
        default=False,
        help="True to store enhanced audios under args.outdir",
    )

    # data related
    parser.add_argument("--device", type=str, default="cpu", help='"cpu" or "cuda"')
    parser.add_argument("--fs", type=int, default=16000, help="sampling rate in Hz")
    parser.add_argument(
        "--verbose", type=str2bool, default=True, help="True to show detailed logs"
    )

    # enhancement related
    parser.add_argument(
        "--beamformer_type",
        type=str,
        default=None,
        help="Type of beamformer for testing",
    )
    parser.add_argument(
        "--mask_type",
        type=str,
        default=None,
        help="Type of reference masks for beamforming",
    )
    parser.add_argument(
        "--beamforming_iter",
        type=int,
        default=1,
        help="Number of beamforming iterations",
    )
    parser.add_argument(
        "--merge_arrays",
        type=str2bool,
        default=False,
        help="True for multi-array beamforming, False for array selection",
    )
    parser.add_argument(
        "--post_beamforming",
        type=str2bool,
        default=False,
        help="Only used when args.merge_arrays is False. "
        "Perform beamforming between sub-array outputs.",
    )
    parser.add_argument(
        "--snr_selection",
        type=str2bool,
        default=False,
        help="Whether to apply SNR-based ref channel selection instead of the "
        "given ref_channel for each beamforming step;\n"
        "Note: When merge_arrays=False, a second SNR-based enhanced "
        "signal selection will also be performed.",
    )
    parser.add_argument(
        "--normalize_wav",
        type=str2bool,
        default=False,
        help="True for normalize the enhanced/reference speech before evaluation"
        " and storing",
    )

    # WPE related
    parser.add_argument(
        "--force_using_wpe",
        type=str2bool,
        default=False,
        help="True to apply Nara-WPE as preprocessing",
    )
    parser.add_argument(
        "--wpe_taps",
        type=int,
        default=5,
        help="Number of filter taps for Nara-WPE",
    )
    parser.add_argument(
        "--wpe_iterations",
        type=int,
        default=3,
        help="Number of iterations for Nara-WPE",
    )
    return parser


def get_wav_dict(scpfile):
    dic = defaultdict(dict)
    with open(scpfile, "r") as f:
        for line in f:
            if not line.strip():
                continue
            uid, wavpath = line.strip().split()
            # match = re.search(r"_\w+$", uid)
            # if not match:
            #    raise ValueError("Invalid uttid: %s" % uid)
            # uttid, array_type = uid[: match.start()], uid[match.start() + 1 :]
            # dic[uttid][array_type] = wavpath
            dic[uid]["0"] = wavpath
    return dic


def get_one_batch(noisy_wavs, ref_wavs, merge_arrays=False, device="cpu"):
    mixwavs = [sf.read(wav, always_2d=True)[0] for wav in noisy_wavs.values()]
    if merge_arrays:
        # (B=1, T, C1+C2+C3+...)
        speech_mix = torch.as_tensor(
            np.concatenate(mixwavs, axis=1)[None, ...],
            device=device,
            dtype=torch.float32,
        )
        if ref_wavs:
            refwavs = [sf.read(wav, always_2d=True)[0] for wav in ref_wavs.values()]
            speech_ref = torch.as_tensor(
                np.concatenate(refwavs, axis=1)[None, ...],
                device=device,
                dtype=torch.float32,
            )
        else:
            speech_ref = None
    else:
        shapes = [wav.shape for wav in mixwavs]
        has_same_shape = all([sh == shapes[0] for sh in shapes])
        if has_same_shape:
            # (B, T, C)
            speech_mix = torch.as_tensor(
                np.stack(mixwavs, axis=0), device=device, dtype=torch.float32
            )
        else:
            speech_mix = [
                torch.as_tensor(wav[None, ...], device=device, dtype=torch.float32)
                for wav in mixwavs
            ]
            lengths = [s.shape[1] for s in speech_mix]
            # data from all arrays must have the same length
            # assert all([leng == lengths[0] for leng in lengths]), lengths
            if not all([leng == lengths[0] for leng in lengths]):
                print(f'WARNING: length mismatch in "{uttid}"', lengths, flush=True)
                idx_maxlen = np.argmax(lengths)
                assert torch.allclose(
                    speech_mix[idx_maxlen][..., -1, :],
                    speech_mix[idx_maxlen].new_zeros(1),
                )
                # drop the last sample
                speech_mix[idx_maxlen] = speech_mix[idx_maxlen][..., :-1, :]
        if ref_wavs:
            refwavs = [sf.read(wav, always_2d=True)[0] for wav in ref_wavs.values()]
            if has_same_shape:
                speech_ref = torch.as_tensor(
                    np.stack(refwavs, axis=0),
                    device=device,
                    dtype=torch.float32,
                )
            else:
                speech_ref = [
                    torch.as_tensor(wav[None, ...], device=device, dtype=torch.float32)
                    for wav in refwavs
                ]
        else:
            speech_ref = None
    if isinstance(speech_mix, list):
        speech_mix_lengths = torch.LongTensor([s.shape[0] for s in speech_mix[0]])
    else:
        speech_mix_lengths = torch.LongTensor([s.shape[0] for s in speech_mix])
    return speech_mix, speech_mix_lengths, speech_ref


def eval_speech(ref, inf, sample_rate, enh_model):
    sdr, sir, sar, perm = bss_eval_sources(ref, inf, compute_permutation=True)
    stoi_score = stoi(ref, inf, fs_sig=sample_rate)
    estoi_score = stoi(ref, inf, fs_sig=sample_rate, extended=True)
    pesq_score = pesq(sample_rate, ref, inf, "wb")
    si_snr_score = -float(
        enh_model.si_snr_loss(
            torch.as_tensor(ref[None, ...]),
            torch.as_tensor(inf[None, ...]),
        )
    )
    return {
        "pesq": np.array([pesq_score]),
        "stoi": np.array([stoi_score]),
        "estoi": np.array([estoi_score]),
        "si_snr": np.array([si_snr_score]),
        "sdr": sdr,
        "sir": sir,
        "sar": sar,
    }


def display_markdown_table(dict_data):
    new_data = [tuple(dict_data.keys())]
    new_data.extend(list(zip(*[[v for v in dict_data[k]] for k in dict_data.keys()])))
    md = ""
    max_length = [[] for row in dict_data]
    for row in new_data:
        for i, col in enumerate(row):
            max_length[i].append(len(str(col)))
    max_length = [max(length) for length in max_length]
    for i, row in enumerate(new_data):
        md += "|"
        for j, field in enumerate(row):
            spacel = (max_length[j] - len(str(field))) // 2
            spacer = (max_length[j] - len(str(field))) - spacel
            md += " " * (spacel + 1) + str(field) + " " * spacer + " |"
        md += "\n"
        if i == 0:
            md += (
                "|:"
                + "-" * max_length[0]
                + "-|"
                + "".join([":" + "-" * length + ":|" for length in max_length[1:]])
                + "\n"
            )
    print(md, flush=True)
    return md


print(time.ctime(), flush=True)
parser = get_parser()
args = parser.parse_args()

device = args.device
enh_train_config = args.train_config
enh_model_file = args.model_file

enh_model, enh_train_args = EnhancementTask.build_model_from_file(
    enh_train_config, enh_model_file, device
)
enh_model.eval()
for param in enh_model.parameters():
    param.requires_grad = False

# True for multi-array beamforming, False for array selection
merge_arrays = args.merge_arrays
if merge_arrays:
    print("[INFO] Apply multi-array beamforming", flush=True)
else:
    print(
        "[INFO] Apply array selection based on the average estimated a posteriori SNRs",
        flush=True,
    )
force_using_wpe = args.force_using_wpe
if force_using_wpe and not getattr(enh_model.separator, "use_wpe", False):
    enh_train_args.use_wpe = True
    enh_train_args.use_dnn_mask_for_wpe = False
    enh_model.separator.use_wpe = True
    enh_model.separator.wpe = DNN_WPE(use_dnn_mask=False, iterations=3)
    print("[INFO] Setting enh_model.separator.use_wpe to True!", flush=True)
    print("[INFO] Setting enh_model.separator.wpe to Nara-WPE!", flush=True)
if getattr(enh_model.separator, "wpe", None):
    enh_model.separator.wpe.taps = args.wpe_taps
    enh_model.separator.wpe.iterations = args.wpe_iterations
    print(
        f"[INFO] Setting enh_model.separator.wpe.taps to {args.wpe_taps}!", flush=True
    )
    print(
        f"[INFO] Setting enh_model.separator.wpe.iterations to {args.wpe_iterations}!",
        flush=True,
    )

fs = args.fs
ref_channel = enh_model.separator.ref_channel
beamformer_type = (
    args.beamformer_type
    if args.beamformer_type
    else enh_model.separator.beamformer.beamformer_type
)
if args.beamformer_type is not None:
    assert args.beamformer_type in BEAMFORMER_TYPES, args.beamformer_type
    enh_model.separator.beamformer.beamformer_type = args.beamformer_type
    print(
        "[INFO] Setting enh_model.separator.beamformer.beamformer_type to %s!"
        % args.beamformer_type,
        flush=True,
    )
else:
    print(
        "[INFO] enh_model.separator.beamformer.beamformer_type: %s"
        % enh_model.separator.beamformer.beamformer_type,
        flush=True,
    )
if args.mask_type is not None:
    print(f"[INFO] Setting enh_model.mask_type to {args.mask_type}", flush=True)
    enh_model.mask_type = args.mask_type

if args.binarize_thres is not None:
    assert 0.0 <= args.binarize_thres <= 1.0, args.binarize_thres
    enh_model.separator.beamformer.binarize_thres = args.binarize_thres
    print(
        "[INFO] Setting enh_model.separator.beamformer.binarize_thres to %f!"
        % args.binarize_thres,
        flush=True,
    )

if (
    args.beam_tasnet
    or args.do_post_processing
    or args.do_pre_masking
    or args.tasnet_vad
):
    assert args.post_enh_model is not None
    if args.beam_tasnet:
        if args.do_pre_masking:
            print(
                "[INFO] Applying TasNet-Masking-Beam with model: "
                f"{args.post_enh_model}",
                flush=True,
            )
        else:
            print(
                "[INFO] Applying Beam-TasNet style preprocessing with model: "
                f"{args.post_enh_model}",
                flush=True,
            )
    elif args.do_pre_masking:
        print(
            f"[INFO] Applying pre-masking with model: {args.post_enh_model}",
            flush=True,
        )
        if args.mask_fusion:
            print("[INFO] Applying mask fusion", flush=True)
    if args.do_post_processing:
        print(
            f"[INFO] Applying post-processing with model: {args.post_enh_model}",
            flush=True,
        )
    if args.tasnet_vad:
        print(
            f"[INFO] Applying TasNet-VAD (mode={args.vad_mode}) post-processing "
            f"with model: {args.post_enh_model}",
            flush=True,
        )
    if args.post_enh_train_config is None:
        args.post_enh_train_config = Path(args.post_enh_model).parent / "config.yaml"
    post_enh_model, post_enh_train_args = EnhancementTask.build_model_from_file(
        args.post_enh_train_config, args.post_enh_model, device
    )
    post_enh_model.eval()
    for param in post_enh_model.parameters():
        param.requires_grad = False

assert args.beamforming_iter > 0, args.beamforming_iter
if args.beamforming_iter > 1:
    if args.beam_tasnet:
        print(
            "[WARNING] NOT applying multi-iteration beamforming "
            "since we are using Beam-TasNet",
            flush=True,
        )
    else:
        print("[INFO] Applying multi-iteration beamforming", flush=True)

if args.post_beamforming and not args.merge_arrays:
    print(
        "[INFO] Applying array-wise beamforming after beamforming on each array",
        flush=True,
    )

if args.normalize_wav:
    print(
        "[INFO] Apply scale normalization to output audios and reference audios",
        flush=True,
    )

lst_noisy_wav = get_wav_dict(args.wavscp)
keys = lst_noisy_wav.keys()
if not args.test_only:
    assert args.spkscp is not None
    lst_ref_wav = get_wav_dict(args.spkscp)
    assert lst_ref_wav.keys() == keys

keys = list(keys)
# from random import shuffle
# shuffle(keys)

# real time factor
RTFs = []
ret_input = {
    "pesq": [],
    "stoi": [],
    "estoi": [],
    "si_snr": [],
    "sdr": [],
    "sir": [],
    "sar": [],
}
ret_output = {}
# for beamformer_type in BEAMFORMER_TYPES:
ret_output[beamformer_type] = {
    "pesq": [],
    "stoi": [],
    "estoi": [],
    "si_snr": [],
    "sdr": [],
    "sir": [],
    "sar": [],
}

ret_output2 = {}
# for beamformer_type in BEAMFORMER_TYPES:
ret_output2[beamformer_type] = {
    "pesq": [],
    "stoi": [],
    "estoi": [],
    "si_snr": [],
    "sdr": [],
    "sir": [],
    "sar": [],
}

if args.write_wavs:
    wavpath = Path(args.outdir) / "wavs"
    wavpath.mkdir(parents=True, exist_ok=True)
    wavpath_dic = {}

for i, uttid in enumerate(tqdm(keys)):
    if args.verbose:
        print(f"\n====== [{i+1}] {uttid} ======", flush=True)
    speech_mix, speech_lengths, speech_ref = get_one_batch(
        lst_noisy_wav[uttid],
        None if args.test_only else lst_ref_wav[uttid],
        merge_arrays=merge_arrays,
        device=device,
    )
    start = time.time()

    if isinstance(speech_mix, list):
        # B * [(1, T', C, F)]
        spectrum_mix = [enh_model.encoder(sp, speech_lengths)[0] for sp in speech_mix]
        flens = torch.LongTensor([spectrum_mix[0].shape[1]])
    else:
        # (B, T', C, F)
        spectrum_mix, flens = enh_model.encoder(speech_mix, speech_lengths)

    if args.beam_tasnet:
        speech_ref_pre = []
        noise_ref_pre = []
        for n, each_array in enumerate(speech_mix):
            if isinstance(speech_mix, list):
                # C * [(B=1, T, C)]
                speech_mix_arr = [
                    each_array if i == 0 else torch.roll(each_array, shifts=-i, dims=-1)
                    for i in range(each_array.size(-1))
                ]
            else:
                # C * [(1, T, C)]
                speech_mix_arr = [
                    each_array.unsqueeze(0)
                    if i == 0
                    else torch.roll(each_array, shifts=-i, dims=-1).unsqueeze(0)
                    for i in range(each_array.size(-1))
                ]
            feats_pre_sp, feats_pre_n = [], []
            for sm in speech_mix_arr:
                feats_pre0, f_lens_pre = post_enh_model.encoder(sm, speech_lengths[:1])
                fp, _, others = post_enh_model.separator(feats_pre0, f_lens_pre)
                feats_pre_sp.append(fp[0])
                feats_pre_n.append(others["noise1"])

            # (B=1, T, C)
            sp_pre = torch.stack(
                [
                    post_enh_model.decoder(f, speech_lengths[:1])[0]
                    for f in feats_pre_sp
                ],
                dim=-1,
            )
            noise_pre = torch.stack(
                [post_enh_model.decoder(f, speech_lengths[:1])[0] for f in feats_pre_n],
                dim=-1,
            )
            # sf.write(f"speech_est_{uttid}.wav", sp_pre[0].cpu().numpy(), fs)
            # sf.write(f"noise_est_{uttid}.wav", noise_pre[0].cpu().numpy(), fs)
            # for ch in range(sp_pre.shape[-1]):
            #    speech_source = speech_ref[0, :, ch].unsqueeze(0).cpu().numpy()
            #    om = eval_speech(
            #        speech_source[0],
            #        sp_pre[0, ..., ch].cpu().numpy(),
            #        sample_rate=fs,
            #        enh_model=enh_model,
            #    )
            #    if args.verbose:
            #        print(f"Output Metrics (CH{ch}):")
            #        display_markdown_table(om)
            speech_ref_pre.append(sp_pre)
            noise_ref_pre.append(sp_pre)

        # #arr * [(B=1, T', C, F)]
        sp_spectrum_ref_pre = [
            enh_model.encoder(sp, speech_lengths[:1])[0] for sp in speech_ref_pre
        ]
        noise_spectrum_ref_pre = [
            enh_model.encoder(np, speech_lengths[:1])[0] for np in noise_ref_pre
        ]
        if args.do_pre_masking:
            mask_ref_pre = [
                # (B=1, T', C, F)
                enh_model._create_mask_label(
                    spectrum_mix[n]
                    if isinstance(speech_mix, list)
                    else spectrum_mix[n : n + 1],
                    [sp_spec_ref],
                    mask_type=enh_model.mask_type,
                )[0].permute(0, 3, 2, 1)
                for n, sp_spec_ref in enumerate(sp_spectrum_ref_pre)
            ]
            if not isinstance(speech_mix, list):
                # (B, F, C, T')
                mask_ref_pre = torch.cat(mask_ref_pre, dim=0)
        if args.tasnet_vad:
            vad_masks = [
                compute_vad(
                    sp[0].cpu().numpy(),
                    mode=args.vad_mode,
                    fs=fs,
                    frame_size=30,
                    ref_channel=ref_channel,
                    strict=True,
                )
                for sp in speech_ref_pre
            ]
    elif args.do_pre_masking:
        mask_ref_pre = []
        if args.tasnet_vad:
            vad_masks = []
        for n, each_array in enumerate(speech_mix):
            if isinstance(speech_mix, list):
                # (B=1, T, C) -> (C, T)
                speech_mix_arr = each_array.squeeze(0).transpose(-1, -2)
                # (B=1, T', C, F)
                spec_mix = spectrum_mix[n]
            else:
                # (T, C) -> (C, T)
                speech_mix_arr = each_array.transpose(-1, -2)
                # (B=1, T', C, F)
                spec_mix = spectrum_mix[n : n + 1]
            feats_pre, f_lens_pre = post_enh_model.encoder(
                speech_mix_arr, speech_lengths[:1]
            )
            feats_pre, _, _ = post_enh_model.separator(feats_pre, f_lens_pre)
            # (B=1, T, C)
            waves_pre = (
                [post_enh_model.decoder(f, speech_lengths[:1])[0] for f in feats_pre][0]
                .transpose(-1, -2)
                .unsqueeze(0)
            )
            if args.tasnet_vad:
                vad_masks.append(
                    compute_vad(
                        waves_pre[0].cpu().numpy(),
                        mode=args.vad_mode,
                        fs=fs,
                        frame_size=30,
                        ref_channel=ref_channel,
                        strict=True,
                    )
                )
            # (B=1, T', C, F)
            spectrum_ref_pre = enh_model.encoder(waves_pre, speech_lengths[:1])[0]
            mask_pre = enh_model._create_mask_label(
                spec_mix, [spectrum_ref_pre], mask_type=enh_model.mask_type
            )[0]
            # (B=1, F, C, T')
            mask_ref_pre.append(mask_pre.permute(0, 3, 2, 1))
        if not isinstance(speech_mix, list):
            # (B, F, C, T')
            mask_ref_pre = torch.cat(mask_ref_pre, dim=0)

    if isinstance(speech_mix, list):
        if enh_model.separator.use_wpe:
            spectrum_mix = [
                enh_model.separator.wpe(sp, flens)[0] for sp in spectrum_mix
            ]

        spectrum_pre, others, prior_snrs, post_snrs = [], [], [], []
        for n, sp in enumerate(spectrum_mix):
            if args.beam_tasnet and not args.do_pre_masking:
                (
                    spec_pre,
                    _,
                    other,
                    prior_snr,
                    post_snr,
                ) = enh_model.separator.beamformer(
                    sp,
                    flens,
                    snr_selection=args.snr_selection,
                    speech_est=sp_spectrum_ref_pre[n],
                    noise_est=noise_spectrum_ref_pre[n],
                )
            else:
                if args.do_pre_masking:
                    speech_masks = [mask_ref_pre[n]]
                else:
                    speech_masks = None

                for iter_i in range(args.beamforming_iter):
                    if iter_i > 0:
                        # (B=1, T', C, F)
                        mask_pre = enh_model._create_mask_label(
                            spec_mix,
                            [spec_pre.unsqueeze(-2)],
                            mask_type=enh_model.mask_type,
                        )[0]
                        # (B=1, F, C, T')
                        speech_masks = [mask_pre.permute(0, 3, 2, 1)]
                    (
                        spec_pre,
                        _,
                        other,
                        prior_snr,
                        post_snr,
                    ) = enh_model.separator.beamformer(
                        sp,
                        flens,
                        snr_selection=args.snr_selection,
                        speech_masks=speech_masks,
                        mask_fusion=args.mask_fusion if iter_i == 0 else False,
                    )

            spectrum_pre.append(spec_pre)
            others.append(other)
            prior_snrs.append(prior_snr.view(-1))
            post_snrs.append(post_snr.view(-1))
        # (B,)
        prior_snrs = torch.cat(prior_snrs, dim=0)
        post_snrs = torch.cat(post_snrs, dim=0)

        if not merge_arrays:
            if args.verbose:
                print(f"a priori SNRs: {prior_snrs.cpu().float()}", flush=True)
                print(f"a posteriori SNRs: {post_snrs.cpu().float()}", flush=True)

            # using oracle array selection
            # speech_source = speech_ref[0, :, ref_channel].unsqueeze(0).cpu().numpy()
            # speech_pre = [
            #     enh_model.decoder(sp, speech_lengths)[0][0].unsqueeze(0)
            #     for sp in spectrum_pre
            # ]
            # oms = [
            #     eval_speech(
            #         speech_source[0],
            #         sp.cpu().numpy()[0],
            #         sample_rate=fs,
            #         enh_model=enh_model,
            #     )["sdr"]
            #     for sp in speech_pre
            # ]
            # ref_array = np.argmax(oms)

            # using maximum-SNR array selection
            # snr_max, ref_array = torch.max(post_snrs, dim=0)

            # using maximum-energy array selection
            power = torch.stack(
                [(sp.real ** 2 + sp.imag ** 2).mean() for sp in spectrum_pre],
                dim=0,
            )
            print(f"powers: {power.cpu()}", flush=True)
            perm = torch.sort(power, dim=0, descending=True)[1]
            ref_array = perm[0]
            if args.post_beamforming:
                # (B=1, T', C', F)
                spectrum_pre = torch.stack(spectrum_pre, dim=0)[perm].permute(
                    1, 2, 0, 3
                )
                (
                    spectrum_pre,
                    _,
                    other,
                    prior_snr,
                    post_snr,
                ) = enh_model.separator.beamformer(
                    spectrum_pre,
                    flens,
                    snr_selection=args.snr_selection,
                    speech_masks=None,
                    mask_fusion=False,
                )
            else:
                spectrum_pre = spectrum_pre[ref_array]
            speech_mix = speech_mix[ref_array]
            if not args.test_only:
                speech_ref = speech_ref[ref_array]
    else:
        if enh_model.separator.use_wpe:
            spectrum_mix, flens, mask_w, powers = enh_model.separator.wpe(
                spectrum_mix, flens
            )

        if args.beam_tasnet and not args.do_pre_masking:
            (
                spectrum_pre,
                flens,
                others,
                prior_snrs,
                post_snrs,
            ) = enh_model.separator.beamformer(
                spectrum_mix,
                flens,
                snr_selection=args.snr_selection,
                speech_est=sp_spectrum_ref_pre[n],
                noise_est=noise_spectrum_ref_pre[n],
            )
        else:
            if args.do_pre_masking:
                speech_masks = [mask_ref_pre]
            else:
                speech_masks = None

            for iter_i in range(args.beamforming_iter):
                if iter_i > 0:
                    # (B, T', C, F)
                    mask_pre = enh_model._create_mask_label(
                        spec_mix,
                        [spectrum_pre.unsqueeze(-2)],
                        mask_type=enh_model.mask_type,
                    )[0]
                    # (B, F, C, T')
                    speech_masks = [mask_pre.permute(0, 3, 2, 1)]
                (
                    spectrum_pre,
                    flens,
                    others,
                    prior_snrs,
                    post_snrs,
                ) = enh_model.separator.beamformer(
                    spectrum_mix,
                    flens,
                    snr_selection=args.snr_selection,
                    speech_masks=speech_masks,
                    mask_fusion=args.mask_fusion if iter_i == 0 else False,
                )
        if not merge_arrays:
            if args.verbose:
                print(f"a priori SNRs: {prior_snrs.cpu().float()}", flush=True)
                print(f"a posteriori SNRs: {post_snrs.cpu().float()}", flush=True)

            # using oracle array selection
            # speech_source = speech_ref[0, :, ref_channel].unsqueeze(0).cpu().numpy()
            # speech_pre = enh_model.decoder(spectrum_pre, speech_lengths)[0]
            # oms = [
            #     eval_speech(
            #         speech_source[0],
            #         sp.cpu().numpy(),
            #         sample_rate=fs,
            #         enh_model=enh_model,
            #     )["sdr"]
            #     for sp in speech_pre
            # ]
            # ref_array = np.argmax(oms)

            # using maximum-SNR array selection
            # snr_max, ref_array = torch.max(post_snrs, dim=0)

            # using maximum-energy array selection
            power = (spectrum_pre.real ** 2 + spectrum_pre.imag ** 2).mean(dim=(-1, -2))
            print(f"powers: {power.cpu()}", flush=True)
            perm = torch.sort(power, dim=0, descending=True)[1]
            ref_array = perm[0]
            if args.post_beamforming:
                # (B=1, T', C', F)
                spectrum_pre = spectrum_pre[perm].permute(1, 0, 2).unsqueeze(0)
                assert spectrum_pre.shape[-2] > 1, spectrum_pre.shape
                (
                    spectrum_pre,
                    _,
                    other,
                    prior_snr,
                    post_snr,
                ) = enh_model.separator.beamformer(
                    spectrum_pre,
                    flens[:1],
                    snr_selection=args.snr_selection,
                    speech_masks=None,
                    mask_fusion=False,
                )
            else:
                spectrum_pre = spectrum_pre[ref_array : ref_array + 1]
            flens = flens[ref_array : ref_array + 1]
            speech_lengths = speech_lengths[ref_array : ref_array + 1]
            speech_mix = speech_mix[ref_array : ref_array + 1]
            if not args.test_only:
                speech_ref = speech_ref[ref_array : ref_array + 1]

    # (1, T)
    # speech_pre = torch.stack(
    #     [enh_model.decoder(sp, speech_lengths)[0][0] for sp in spectrum_pre],
    #     dim=0
    # )
    speech_pre = enh_model.decoder(spectrum_pre, speech_lengths)[0][0].unsqueeze(0)
    if args.do_post_processing:
        feats, f_lens = post_enh_model.encoder(speech_pre, speech_lengths)
        feats, _, _ = post_enh_model.separator(feats, f_lens)
        waves = [post_enh_model.decoder(f, speech_lengths)[0] for f in feats]
        speech_pre = waves[0]
    elif args.tasnet_vad:
        # (B=1, T)
        vad_masks = torch.as_tensor(
            np.median(vad_masks, axis=0), device=device
        ).unsqueeze(0)
        speech_pre = speech_pre * vad_masks

    if args.normalize_wav:
        speech_pre = speech_pre / abs(speech_pre).max(dim=1, keepdim=True)[0] * 0.9

    stop = time.time()
    rtf = (stop - start) * fs / len(speech_pre.cpu().numpy()[0])
    if args.verbose:
        print(f"RTF = {rtf}", flush=True)
    RTFs.append(rtf)

    if not args.test_only:
        # (1, T)
        observation = (
            speech_mix[0, :, ref_channel : ref_channel + 1]
            .cpu()
            .transpose(-1, -2)
            .numpy()
        )
        # (1, T)
        speech_source = speech_ref[0, :, ref_channel].unsqueeze(0).cpu().numpy()
        if args.normalize_wav:
            speech_source = (
                speech_source / abs(speech_source).max(axis=1, keepdims=True) * 0.9
            )
            observation = (
                observation / abs(observation).max(axis=1, keepdims=True) * 0.9
            )
        # im = InputMetrics(
        #     observation, speech_source, sample_rate=fs
        # ).as_dict()
        im = eval_speech(
            speech_source[0], observation[0], sample_rate=fs, enh_model=enh_model
        )
        if args.verbose:
            print("Input Metrics:")
            display_markdown_table(im)
        for k in ret_input.keys():
            ret_input[k].append((uttid, float(im[k].mean())))

        # om = OutputMetrics(
        #     speech_pre.cpu().numpy(), speech_source, sample_rate=fs
        # ).as_dict()
        om = eval_speech(
            speech_source[0],
            speech_pre.cpu().numpy()[0],
            sample_rate=fs,
            enh_model=enh_model,
        )
        if args.verbose:
            print("Output Metrics:")
            display_markdown_table(om)
        # beamformer_type = enh_model.separator.beamformer.beamformer_type
        for k in ret_output[beamformer_type].keys():
            ret_output[beamformer_type][k].append((uttid, float(om[k].mean())))

        # om2 = eval_speech(
        #    observation[0],
        #    speech_pre.cpu().numpy()[0],
        #    sample_rate=fs,
        #    enh_model=enh_model,
        # )
        # if args.verbose:
        #    print("Output-vs-Obervation Metrics:")
        #    display_markdown_table(om2)
        # for k in ret_output2[beamformer_type].keys():
        #    ret_output2[beamformer_type][k].append((uttid, float(om2[k].mean())))

    if args.write_wavs:
        sf.write(wavpath / (uttid + ".wav"), speech_pre.cpu().numpy()[0], fs)
        wavpath_dic[uttid] = (wavpath / (uttid + ".wav")).absolute()

print("Average real time factor (RTF): ", np.mean(RTFs))
if not args.test_only:
    print("Input Metrics:")
    for k in ret_input.keys():
        print("{}: {}".format(k, np.mean(list(zip(*ret_input[k]))[1])))

    # bt = enh_model.separator.beamformer.beamformer_type
    bt = beamformer_type
    print("\nOutput Metrics:")
    # for bt in ret_output.keys():
    print(bt)
    for k in ret_output[bt].keys():
        print("{}: {}".format(k, np.mean(list(zip(*ret_output[bt][k]))[1])))

    # print("\nOutput-vs-Observation Metrics:")
    # for k in ret_output2[bt].keys():
    #    print("{}: {}".format(k, np.mean(list(zip(*ret_output2[bt][k]))[1])))
    print("----------------------------")

    for metric in ret_input.keys():
        with open(Path(args.outdir) / f"input_{metric.upper()}_spk1", "w") as f:
            for uttid, value in ret_input[metric]:
                f.write("{} {}\n".format(uttid, value))

    for metric in ret_output[bt].keys():
        with open(Path(args.outdir) / f"enhanced_{metric.upper()}_spk1", "w") as f:
            for uttid, value in ret_output[bt][metric]:
                f.write("{} {}\n".format(uttid, value))

if args.write_wavs:
    with open(Path(args.outdir) / "spk1.scp", "w") as f:
        for uttid, value in wavpath_dic.items():
            f.write("{} {}\n".format(uttid, value))

print(time.ctime(), flush=True)
# IPython.embed()
