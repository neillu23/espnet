"""
Trainer module for speaker recognition.
In speaker recognition (embedding extractor training/inference),
calculating validation loss in closed set is not informative since
generalization in unseen utterances from known speakers are good in most cases.
Thus, we measure open set equal error rate (EER) using unknown speakers by
overriding validate_one_epoch.
"""

import argparse
import dataclasses
import logging
from contextlib import contextmanager
from dataclasses import is_dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import humanfriendly
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim
from torch.nn.parallel import DistributedDataParallel as DDP

from packaging.version import parse as V
from typeguard import check_argument_types

from espnet2.iterators.abs_iter_factory import AbsIterFactory
from espnet2.main_funcs.average_nbest_models import average_nbest_models
from espnet2.main_funcs.calculate_all_attentions import calculate_all_attentions
from espnet2.schedulers.abs_scheduler import (
    AbsBatchStepScheduler,
    AbsEpochStepScheduler,
    AbsScheduler,
    AbsValEpochStepScheduler,
)
from espnet2.torch_utils.add_gradient_noise import add_gradient_noise
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.recursive_op import recursive_average
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet2.train.distributed_utils import DistributedOption
from espnet2.train.reporter import Reporter, SubReporter
from espnet2.train.trainer import Trainer, TrainerOptions
from espnet2.utils.build_dataclass import build_dataclass
from espnet2.utils.eer import ComputeErrorRates, ComputeMinDcf, tuneThresholdfromScore
from espnet2.utils.kwargs2args import kwargs2args

if torch.distributed.is_available():
    from torch.distributed import ReduceOp


class JointTrainer(Trainer):
    """
    Trainer.
    Designed for speaker recognition.
    Training will be done as closed set classification.
    Validation will be open set EER calculation.

    """

    def __init__(self):
        raise RuntimeError("This class can't be instantiated.")

    @classmethod
    @torch.no_grad()
    def validate_one_epoch_spk(
        cls,
        model: torch.nn.Module,
        iterator: Iterable[Dict[str, torch.Tensor]],
        reporter: SubReporter,
        options: TrainerOptions,
        distributed_option: DistributedOption,
    ) -> None:
        assert check_argument_types()
        ngpu = options.ngpu
        no_forward_run = options.no_forward_run
        distributed = distributed_option.distributed

        model.eval()

        embed_condition = False
        if hasattr(model, "embed_condition"):
            embed_condition = model.embed_condition
        else:
            embed_condition = model.module.embed_condition
        logging.info(f"embed_condition: {embed_condition}")

        scores = []
        labels = []
        spk_embd_dic = {}
        bs = 0

        # [For distributed] Because iteration counts are not always equals between
        # processes, send stop-flag to the other processes if iterator is finished
        iterator_stop = torch.tensor(0).to("cuda" if ngpu > 0 else "cpu")

        # fill dictionary with speech samples
        utt_id_list = []
        speech_list = []
        lang_list = []
        task_token = None
        for utt_id, batch in iterator:
            bs = max(bs, len(utt_id))
            if "task_tokens" in batch:
                task_token = batch["task_tokens"][0]

            assert isinstance(batch, dict), type(batch)
            
            for _utt_id, _speech, _speech2 in zip(
                utt_id, batch["speech"], batch["speech2"]
            ):
                _utt_id_1, _utt_id_2 = _utt_id.split("*")
                if _utt_id_1 not in utt_id_list:
                    utt_id_list.append(_utt_id_1)
                    speech_list.append(
                        to_device(_speech, "cuda" if ngpu > 0 else "cpu")
                    )
                if _utt_id_2 not in utt_id_list:
                    utt_id_list.append(_utt_id_2)
                    speech_list.append(
                        to_device(_speech2, "cuda" if ngpu > 0 else "cpu")
                    )
        # import pdb; pdb.set_trace()

        # extract speaker embeddings.
        n_utt = len(utt_id_list)
        for ii in range(0, n_utt, bs):
            _utt_ids = utt_id_list[ii : ii + bs]
            _speechs = speech_list[ii : ii + bs]
            num_eval = _speechs[0].shape[0] # torch.Size([5, 48000])
            _speechs = torch.stack(_speechs, dim=0)
            org_shape = (_speechs.size(0), _speechs.size(1))
            _speechs = _speechs.flatten(0, 1)
            _speechs = to_device(_speechs, "cuda" if ngpu > 0 else "cpu")

            if task_token is None:
                task_tokens = None
            else:
                task_tokens = to_device(
                    task_token.repeat(_speechs.size(0)), "cuda" if ngpu > 0 else "cpu"
                ).unsqueeze(1)
                
                
            # if isinstance(model, DDP):
            #     spk_embds = model.module.encode(
            #         speech=_speechs,
            #         speech_lengths=torch.tensor([sp.size(0) for sp in _speechs]).to("cuda" if ngpu > 0 else "cpu"),
            #         encode_spk=True,
            #     )
            # else:
            #     spk_embds = model.encode(
            #         speech=_speechs,
            #         speech_lengths=torch.tensor([sp.size(0) for sp in _speechs]).to("cuda" if ngpu > 0 else "cpu"),
            #         encode_spk=True,
            #     )
            # check if spk_embds is tuple

            spk_embds = model(
                speech=_speechs,
                speech_lengths=torch.tensor([sp.size(0) for sp in _speechs]).to("cuda" if ngpu > 0 else "cpu"),
                text=torch.tensor([-1 for sp in _speechs]).to("cuda" if ngpu > 0 else "cpu"),
                text_lengths=torch.tensor([1 for sp in _speechs]).to("cuda" if ngpu > 0 else "cpu"),
                extract_spk_embd=True,
            )

            if isinstance(spk_embds, tuple):
                spk_embds = spk_embds[-1]
                    
            spk_embds = F.normalize(spk_embds, p=2, dim=1)
            spk_embds = spk_embds.view(org_shape[0], org_shape[1], -1)

            for _utt_id, _spk_embd in zip(_utt_ids, spk_embds):
                spk_embd_dic[_utt_id] = _spk_embd

        del utt_id_list
        del speech_list

        # calculate similarity scores
        for utt_id, batch in iterator:
            batch["spk_labels"] = to_device(
                batch["spk_labels"], "cuda" if ngpu > 0 else "cpu"
            )

            if distributed:
                torch.distributed.all_reduce(iterator_stop, ReduceOp.SUM)
                if iterator_stop > 0:
                    break

            for _utt_id in utt_id:
                _utt_id_1, _utt_id_2 = _utt_id.split("*")
                score = torch.cdist(spk_embd_dic[_utt_id_1], spk_embd_dic[_utt_id_2])
                score = -1.0 * torch.mean(score)
                scores.append(score.view(1))  # 0-dim to 1-dim tensor for cat
            labels.append(batch["spk_labels"])

        else:
            if distributed:
                iterator_stop.fill_(1)
                torch.distributed.all_reduce(iterator_stop, ReduceOp.SUM)
        torch.cuda.empty_cache()

        scores = torch.cat(scores).type(torch.float32)
        labels = torch.cat(labels).type(torch.int32).flatten()
        logging.info(f"Validation: {len(scores)} trials")
        if distributed:
            # get the number of trials assigned on each GPU
            length = to_device(
                torch.tensor([labels.size(0)], dtype=torch.int32), "cuda"
            )
            lengths_all = [
                to_device(torch.zeros(1, dtype=torch.int32), "cuda")
                for _ in range(torch.distributed.get_world_size())
            ]
            torch.distributed.all_gather(lengths_all, length)

            scores_all = [
                to_device(torch.zeros(i, dtype=torch.float32), "cuda")
                for i in lengths_all
            ]
            torch.distributed.all_gather(scores_all, scores)
            scores = torch.cat(scores_all)

            labels_all = [
                to_device(torch.zeros(i, dtype=torch.int32), "cuda")
                for i in lengths_all
            ]
            torch.distributed.all_gather(labels_all, labels)
            labels = torch.cat(labels_all)
            rank = torch.distributed.get_rank()
            torch.distributed.barrier()
        scores = scores.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        logging.info(f"Validation: {len(scores)} trials after the barrier")

        # calculate statistics in target and nontarget classes.
        n_trials = len(scores)
        scores_trg = []
        scores_nontrg = []
        for _s, _l in zip(scores, labels):
            if _l == 1:
                scores_trg.append(_s)
            elif _l == 0:
                scores_nontrg.append(_s)
            else:
                raise ValueError(f"{_l}, {type(_l)}")
        trg_mean = float(np.mean(scores_trg))
        trg_std = float(np.std(scores_trg))
        nontrg_mean = float(np.std(scores_nontrg))
        nontrg_std = float(np.std(scores_nontrg))

        # exception for collect_stats.
        if len(scores) == 1:
            reporter.register(stats=dict(eer=1.0, mindcf=1.0))
            return

        # import pdb; pdb.set_trace()

        # predictions, ground truth, and the false acceptance rates to calculate
        results = tuneThresholdfromScore(scores, labels, [1, 0.1])
        eer = results[1]
        fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)

        # p_target, c_miss, and c_falsealarm in NIST minDCF calculation
        p_trg, c_miss, c_fa = 0.05, 1, 1
        mindcf, _ = ComputeMinDcf(fnrs, fprs, thresholds, p_trg, c_miss, c_fa)

        logging.info(f"EER: {eer:.2f} %")
        reporter.register(
            stats=dict(
                eer=eer,
                mindcf=mindcf,
                n_trials=n_trials,
                trg_mean=trg_mean,
                trg_std=trg_std,
                nontrg_mean=nontrg_mean,
                nontrg_std=nontrg_std,
            )
        )

        logging.info(f"Validation finished")
        # added to reduce GRAM usage. May have minor speed boost when
        # this line is commented in case GRAM is not fully used.
        torch.cuda.empty_cache()

    @classmethod
    @torch.no_grad()
    def extract_embed(
        cls,
        model: torch.nn.Module,
        iterator: Iterable[Dict[str, torch.Tensor]],
        reporter: SubReporter,
        options: TrainerOptions,
        distributed_option: DistributedOption,
        output_dir: str,
        custom_bs: int,
        average: bool = False,
        sv_asr_joint_task: bool = False,

    ) -> None:
        assert check_argument_types()
        ngpu = options.ngpu
        no_forward_run = options.no_forward_run
        distributed = distributed_option.distributed

        model.eval()

        embed_condition = False
        if hasattr(model, "embed_condition"):
            embed_condition = model.embed_condition
        else:
            embed_condition = model.module.embed_condition
        logging.info(f"embed_condition: {embed_condition}")

        scores = []
        labels = []
        spk_embd_dic = {}

        # [For distributed] Because iteration counts are not always equals between
        # processes, send stop-flag to the other processes if iterator is finished
        iterator_stop = torch.tensor(0).to("cuda" if ngpu > 0 else "cpu")

        # fill dictionary with speech samples
        utt_id_list = []
        utt_id_whole_list = []
        speech_list = []
        langs_list = []
        task_token_list = []
        task_token = None
        if distributed:
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            rank = 0
            world_size = 1
        idx = 0
        for utt_id, batch in iterator:
            if "task_tokens" in batch:
                task_token = batch["task_tokens"][0]

            assert isinstance(batch, dict), type(batch)

            if embed_condition and "langs" in batch:
                for _utt_id, _speech, _speech2, _langs, _langs2 in zip(
                    utt_id, batch["speech"], batch["speech2"], batch["langs"], batch["langs2"]
                ):
                    _utt_id_1, _utt_id_2 = _utt_id.split("*")
                    if _utt_id_1 not in utt_id_whole_list:
                        utt_id_whole_list.append(_utt_id_1)
                        if idx % world_size == rank:
                            utt_id_list.append(_utt_id_1)
                            speech_list.append(_speech)
                            task_token_list.append(task_token)
                            langs_list.append(_langs)

                        if len(utt_id_list) == custom_bs:
                            num_eval = speech_list[0].shape[0] 
                            speech_list = torch.stack(speech_list, dim=0)
                            org_shape = (speech_list.size(0), speech_list.size(1))
                            speech_list = speech_list.flatten(0, 1)
                            speech_list = to_device(
                                speech_list, "cuda" if ngpu > 0 else "cpu"
                            )
                            if task_token is None:
                                task_tokens = None
                            else:
                                task_tokens = to_device(
                                    task_token.repeat(speech_list.size(0)),
                                    "cuda" if ngpu > 0 else "cpu",
                                ).unsqueeze(1)

                            langs_list = [_lang.expand(num_eval, 1) for _lang in langs_list]
                            langs_list = torch.stack(langs_list, dim=1)
                            langs_list = langs_list.flatten(0, 1)
                            langs_list = to_device(langs_list, "cuda" if ngpu > 0 else "cpu")
                            # logging.info(f"_langs: {_langs.shape}")
                            spk_embds = model(
                                speech=speech_list,
                                spk_labels=None,
                                extract_embd=True,
                                task_tokens=task_tokens,
                                langs=langs_list,
                            )
                            # removed to be use magnitude in qmf
                            # spk_embds = F.normalize(spk_embds, p=2, dim=1)
                            spk_embds = spk_embds.view(
                                org_shape[0], org_shape[1], -1
                            )

                            for uid, _spk_embd in zip(utt_id_list, spk_embds):
                                if average:
                                    spk_embd_dic[uid] = (
                                        _spk_embd.mean(0).detach().cpu().numpy()
                                    )
                                else:
                                    spk_embd_dic[uid] = _spk_embd.detach().cpu().numpy()
                            
                            utt_id_list = []
                            speech_list = []
                            langs_list = []

                 
                        idx += 1      

                    if _utt_id_2 not in utt_id_whole_list:
                        utt_id_whole_list.append(_utt_id_2)
                        if idx % world_size == rank:
                            utt_id_list.append(_utt_id_2)
                            speech_list.append(_speech2)
                            task_token_list.append(task_token)
                            langs_list.append(_langs2)
                            

                        if len(utt_id_list) == custom_bs:
                            num_eval = speech_list[0].shape[0] 
                            speech_list = torch.stack(speech_list, dim=0)
                            org_shape = (speech_list.size(0), speech_list.size(1))
                            speech_list = speech_list.flatten(0, 1)
                            speech_list = to_device(
                                speech_list, "cuda" if ngpu > 0 else "cpu"
                            )
                            if task_token is None:
                                task_tokens = None
                            else:
                                task_tokens = to_device(
                                    task_token.repeat(speech_list.size(0)),
                                    "cuda" if ngpu > 0 else "cpu",
                                ).unsqueeze(1)



                            langs_list = [_lang.expand(num_eval, 1) for _lang in langs_list]
                            langs_list = torch.stack(langs_list, dim=1)
                            langs_list = langs_list.flatten(0, 1)
                            langs_list = to_device(langs_list, "cuda" if ngpu > 0 else "cpu")
                            spk_embds = model(
                                speech=speech_list,
                                spk_labels=None,
                                extract_embd=True,
                                task_tokens=task_tokens,
                                langs=langs_list,
                            )
                            # removed to be use magnitude in qmf
                            # spk_embds = F.normalize(spk_embds, p=2, dim=1)
                            spk_embds = spk_embds.view(
                                org_shape[0], org_shape[1], -1
                            )

                            for uid, _spk_embd in zip(utt_id_list, spk_embds):
                                if average:
                                    spk_embd_dic[uid] = (
                                        _spk_embd.mean(0).detach().cpu().numpy()
                                    )
                                else:
                                    spk_embd_dic[uid] = _spk_embd.detach().cpu().numpy()
                            
                            utt_id_list = []
                            speech_list = []
                            langs_list = []

                        idx += 1
            else:
                for _utt_id, _speech, _speech2 in zip(
                    utt_id, batch["speech"], batch["speech2"]
                ):
                    _utt_id_1, _utt_id_2 = _utt_id.split("*")
                    if _utt_id_1 not in utt_id_whole_list:
                        utt_id_whole_list.append(_utt_id_1)
                        if idx % world_size == rank:
                            utt_id_list.append(_utt_id_1)
                            speech_list.append(_speech)

                        if len(utt_id_list) == custom_bs:
                            speech_list = torch.stack(speech_list, dim=0)
                            org_shape = (speech_list.size(0), speech_list.size(1))
                            speech_list = speech_list.flatten(0, 1)
                            speech_list = to_device(
                                speech_list, "cuda" if ngpu > 0 else "cpu"
                            )
                            if task_token is None:
                                task_tokens = None
                            else:
                                task_tokens = to_device(
                                    task_token.repeat(speech_list.size(0)),
                                    "cuda" if ngpu > 0 else "cpu",
                                ).unsqueeze(1)
                            # import pdb; pdb.set_trace()
                            if sv_asr_joint_task:
                                spk_embds = model.encode(
                                    speech=speech_list,
                                    speech_lengths=torch.tensor([sp.size(0) for sp in speech_list]).to("cuda" if ngpu > 0 else "cpu")
                                )[-1]
                                # if spk_embds is list
                                if isinstance(spk_embds, list):
                                    spk_embds = spk_embds[-1]
                            else:
                                spk_embds = model(
                                    speech=speech_list,
                                    spk_labels=None,
                                    extract_embd=True,
                                    task_tokens=task_tokens,
                                )
                            # removed to be use magnitude in qmf
                            # spk_embds = F.normalize(spk_embds, p=2, dim=1)
                            spk_embds = spk_embds.view(org_shape[0], org_shape[1], -1)

                            for uid, _spk_embd in zip(utt_id_list, spk_embds):
                                if average:
                                    spk_embd_dic[uid] = (
                                        _spk_embd.mean(0).detach().cpu().numpy()
                                    )
                                else:
                                    spk_embd_dic[uid] = _spk_embd.detach().cpu().numpy()

                            utt_id_list = []
                            speech_list = []

                        idx += 1
                    if _utt_id_2 not in utt_id_whole_list:
                        utt_id_whole_list.append(_utt_id_2)
                        if idx % world_size == rank:
                            utt_id_list.append(_utt_id_2)
                            speech_list.append(_speech2)

                        if len(utt_id_list) == custom_bs:
                            speech_list = torch.stack(speech_list, dim=0)
                            org_shape = (speech_list.size(0), speech_list.size(1))
                            speech_list = speech_list.flatten(0, 1)
                            speech_list = to_device(
                                speech_list, "cuda" if ngpu > 0 else "cpu"
                            )
                            if task_token is None:
                                task_tokens = None
                            else:
                                task_tokens = to_device(
                                    task_token.repeat(speech_list.size(0)),
                                    "cuda" if ngpu > 0 else "cpu",
                                ).unsqueeze(1)
                            if sv_asr_joint_task:
                                spk_embds = model.encode(
                                    speech=speech_list,
                                    speech_lengths=torch.tensor([sp.size(0) for sp in speech_list]).to("cuda" if ngpu > 0 else "cpu")
                                )[-1]

                                if isinstance(spk_embds, list):
                                    spk_embds = spk_embds[-1]
                                
                            else:
                                spk_embds = model(
                                    speech=speech_list,
                                    spk_labels=None,
                                    extract_embd=True,
                                    task_tokens=task_tokens,
                                )
                            # removed to be use magnitude in qmf
                            # spk_embds = F.normalize(spk_embds, p=2, dim=1)
                            spk_embds = spk_embds.view(org_shape[0], org_shape[1], -1)

                            for uid, _spk_embd in zip(utt_id_list, spk_embds):
                                if average:
                                    spk_embd_dic[uid] = (
                                        _spk_embd.mean(0).detach().cpu().numpy()
                                    )
                                else:
                                    spk_embd_dic[uid] = _spk_embd.detach().cpu().numpy()

                            utt_id_list = []
                            speech_list = []

                        idx += 1
                        
                

        if len(utt_id_list) != 0:
            num_eval = speech_list[0].shape[0]
            speech_list = torch.stack(speech_list, dim=0)
            org_shape = (speech_list.size(0), speech_list.size(1))
            speech_list = speech_list.flatten(0, 1)
            speech_list = to_device(speech_list, "cuda" if ngpu > 0 else "cpu")
            if task_token is None:
                task_tokens = None
            else:
                task_tokens = to_device(
                    task_token.repeat(speech_list.size(0)),
                    "cuda" if ngpu > 0 else "cpu",
                ).unsqueeze(1)

            if embed_condition and "langs" in batch:
                langs_list = [_lang.expand(num_eval, 1) for _lang in langs_list]
                langs_list = torch.stack(langs_list, dim=1)
                langs_list = langs_list.flatten(0, 1)
                langs_list = to_device(langs_list, "cuda" if ngpu > 0 else "cpu")
            else:
                langs_list = None
            
            if sv_asr_joint_task:
                speech_lengths = torch.tensor([speech_list.size(0)]).to("cuda" if ngpu > 0 else "cpu")
                spk_embds = model.encode(
                    speech=speech_list,
                    speech_lengths=torch.tensor([sp.size(0) for sp in speech_list]).to("cuda" if ngpu > 0 else "cpu")
                )[-1]

                if isinstance(spk_embds, list):
                    spk_embds = spk_embds[-1]
            else:
                spk_embds = model(
                    speech=speech_list,
                    spk_labels=None,
                    extract_embd=True,
                    task_tokens=task_tokens,
                    langs=langs_list,
                )
            spk_embds = F.normalize(spk_embds, p=2, dim=1)
            spk_embds = spk_embds.view(org_shape[0], org_shape[1], -1)

            for uid, _spk_embd in zip(utt_id_list, spk_embds):
                if average:
                    spk_embd_dic[uid] = _spk_embd.mean(0).detach().cpu().numpy()
                else:
                    spk_embd_dic[uid] = _spk_embd.detach().cpu().numpy()

        np.savez(output_dir + f"/embeddings{rank}", **spk_embd_dic)
