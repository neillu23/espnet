#!/usr/bin/env python3
import argparse
import logging
import sys
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import humanfriendly
from mir_eval.separation import bss_eval_sources
from pystoi import stoi
import torch
from typeguard import check_argument_types

from espnet.utils.cli_utils import get_commandline_args
from espnet2.enh.espnet_model import ESPnetEnhancementModel
from espnet2.fileio.datadir_writer import DatadirWriter
from espnet2.fileio.sound_scp import SoundScpWriter
from espnet2.tasks.enh_asr import EnhASRTask
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.utils import config_argparse
from espnet2.utils.types import str2bool
from espnet2.utils.types import str2triple_str
from espnet2.utils.types import str_or_none


def humanfriendly_or_none(value: str):
    if value in ("none", "None", "NONE"):
        return None
    return humanfriendly.parse_size(value)


def inference(
    output_dir: str,
    batch_size: int,
    dtype: str,
    fs: int,
    ngpu: int,
    seed: int,
    num_workers: int,
    log_level: Union[int, str],
    data_path_and_name_and_type: Sequence[Tuple[str, str, str]],
    key_file: Optional[str],
    joint_train_config: str,
    joint_model_file: str,
    allow_variable_data_keys: bool,
    normalize_output_wav: bool,
    ref_channel: int,
):
    assert check_argument_types()
    if batch_size > 1:
        raise NotImplementedError("batch decoding is not implemented")
    if ngpu > 1:
        raise NotImplementedError("only single GPU decoding is supported")

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    if ngpu >= 1:
        device = "cuda"
    else:
        device = "cpu"

    # 1. Set random-seed
    set_all_random_seed(seed)

    # 2. Build joint model
    joint_model, joint_train_args = EnhASRTask.build_model_from_file(
        joint_train_config, joint_model_file, device
    )
    joint_model.eval()
    enh_model = joint_model.enh_subclass

    num_spk = enh_model.num_spk

    # 3. Build data-iterator
    loader = EnhASRTask.build_streaming_iterator(
        data_path_and_name_and_type,
        dtype=dtype,
        batch_size=batch_size,
        key_file=key_file,
        num_workers=num_workers,
        preprocess_fn=EnhASRTask.build_preprocess_fn(joint_train_args, False),
        collate_fn=EnhASRTask.build_collate_fn(joint_train_args, False),
        allow_variable_data_keys=allow_variable_data_keys,
        inference=True,
    )

    writers = []
    for i in range(num_spk):
        writers.append(
            SoundScpWriter(f"{output_dir}/wavs/{i + 1}", f"{output_dir}/spk{i + 1}.scp")
        )

    # FIXME(kamo): The output format should be discussed about
    with DatadirWriter(output_dir) as writer:
        for keys, batch in loader:
            assert isinstance(batch, dict), type(batch)
            assert all(isinstance(s, str) for s in keys), keys
            _bs = len(next(iter(batch.values())))
            assert len(keys) == _bs, f"{len(keys)} != {_bs}"

            logging.info(f"keys: {keys}")
            key = keys[0]
            with torch.no_grad():
                # a. To device
                batch = to_device(batch, device)
                # b. Forward Enhancement Frontend
                feature_mix, flens = enh_model.encoder( batch["speech_mix"], batch["speech_mix_lengths"])
                feature_pre, flens, others = enh_model.separator(feature_mix, flens)
                speech_pre = [enh_model.decoder(ps, batch["speech_mix_lengths"])[0] for ps in feature_pre]
            # nspk,T
            speech_ref = [
                batch["speech_ref{}".format(spk + 1)] for spk in range(num_spk)
            ]
            ref = torch.stack(speech_ref, dim=1).squeeze(dim=0).numpy()
            inf = torch.stack(speech_pre, dim=1).squeeze(dim=0).numpy()
            if inf.ndim == 1:
                inf = inf[:, None]
            if ref.ndim == 1:
                ref = ref[:, None]
            if ref.ndim > inf.ndim:
                # multi-channel reference and single-channel output
                ref = ref[..., ref_channel]
                assert ref.shape == inf.shape, (ref.shape, inf.shape)
            elif ref.ndim < inf.ndim:
                # single-channel reference and multi-channel output
                raise ValueError(
                    "Reference must be multi-channel when the \
                    network output is multi-channel."
                )
            elif ref.ndim == inf.ndim == 3:
                # multi-channel reference and output
                ref = ref[..., ref_channel]
                inf = inf[..., ref_channel]

            sdr, sir, sar, perm = bss_eval_sources(ref, inf, compute_permutation=True)

            for i in range(num_spk):
                stoi_score = stoi(ref[i], inf[int(perm[i])], fs_sig=fs)
                estoi_score = stoi(ref[i], inf[int(perm[i])], fs_sig=fs, extended=True)
                si_snr_score = -float(
                    ESPnetEnhancementModel.si_snr_loss(
                        torch.from_numpy(ref[i][None, ...]),
                        torch.from_numpy(inf[int(perm[i])][None, ...]),
                    )
                )
                writer[f"STOI_spk{i + 1}"][key] = str(stoi_score)
                writer[f"ESTOI_spk{i + 1}"][key] = str(estoi_score)
                writer[f"SI_SNR_spk{i + 1}"][key] = str(si_snr_score)
                writer[f"SDR_spk{i + 1}"][key] = str(sdr[i])
                writer[f"SAR_spk{i + 1}"][key] = str(sar[i])
                writer[f"SIR_spk{i + 1}"][key] = str(sir[i])

            # FIXME(Chenda): will be incorrect when
            #  batch size is not 1 or multi-channel case
            if normalize_output_wav:
                speech_pre = [
                    (w / abs(w).max(dim=1, keepdim=True)[0] * 0.9).T.cpu().numpy()
                    for w in speech_pre
                ]  # list[(sample,batch)]
            else:
                speech_pre = [w.T.cpu().numpy() for w in speech_pre]
            # save permutation assigned wav file
            for i in range(num_spk):
                writers[i][key] = fs, speech_pre[perm[i]]

    for wr in writers:
        wr.close()


def get_parser():
    parser = config_argparse.ArgumentParser(
        description="Enh_ASR Inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Note(kamo): Use '_' instead of '-' as separator.
    # '-' is confusing if written in yaml.
    parser.add_argument(
        "--log_level",
        type=lambda x: x.upper(),
        default="INFO",
        choices=("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"),
        help="The verbose level of logging",
    )

    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--ngpu",
        type=int,
        default=0,
        help="The number of gpus. 0 indicates CPU mode",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float16", "float32", "float64"],
        help="Data type",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="The number of workers used for DataLoader",
    )

    parser.add_argument(
        "--fs", type=humanfriendly_or_none, default=8000, help="Sampling rate"
    )
    group = parser.add_argument_group("Input data related")
    group.add_argument(
        "--data_path_and_name_and_type",
        type=str2triple_str,
        required=True,
        action="append",
    )
    group.add_argument("--key_file", type=str_or_none)
    group.add_argument("--ref_channel", type=int, default=0)
    group.add_argument("--allow_variable_data_keys", type=str2bool, default=False)

    group = parser.add_argument_group("The model configuration related")
    group.add_argument("--joint_train_config", type=str, required=True)
    group.add_argument("--joint_model_file", type=str, required=True)

    group.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="The batch size for inference",
    )

    group = parser.add_argument_group("Output wav related")
    group.add_argument(
        "--normalize_output_wav",
        type=str2bool,
        default=False,
        help="Whether to normalize the predicted wav to [-1~1]",
    )

    return parser


def main(cmd=None):
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    kwargs.pop("config", None)
    inference(**kwargs)


if __name__ == "__main__":
    main()
