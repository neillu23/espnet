import argparse
from dataclasses import dataclass
import logging
from typing import Callable
from typing import Collection
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import torch
from typeguard import check_argument_types
from typeguard import check_return_type

from espnet2.enh.decoder.abs_decoder import AbsDecoder
from espnet2.enh.decoder.conv_decoder import ConvDecoder
from espnet2.enh.decoder.null_decoder import NullDecoder
from espnet2.enh.decoder.stft_decoder import STFTDecoder
from espnet2.enh.encoder.abs_encoder import AbsEncoder
from espnet2.enh.encoder.conv_encoder import ConvEncoder
from espnet2.enh.encoder.null_encoder import NullEncoder
from espnet2.enh.encoder.stft_encoder import STFTEncoder
from espnet2.enh.mixit_espnet_model import ESPnetEnhancementModel
from espnet2.enh.loss.criterions.abs_loss import AbsEnhLoss
from espnet2.enh.loss.criterions.tf_domain import FrequencyDomainL1
from espnet2.enh.loss.criterions.tf_domain import FrequencyDomainMSE
from espnet2.enh.loss.criterions.time_domain import CISDRLoss
from espnet2.enh.loss.criterions.time_domain import SISNRLoss
from espnet2.enh.loss.criterions.time_domain import SNRLoss
from espnet2.enh.loss.wrappers.abs_wrapper import AbsLossWrapper
from espnet2.enh.loss.wrappers.fixed_order import FixedOrderSolver
from espnet2.enh.loss.wrappers.mixit_solver import MixITSolver
from espnet2.enh.loss.wrappers.pit_solver import PITSolver
from espnet2.enh.separator.abs_separator import AbsSeparator
from espnet2.enh.separator.asteroid_models import AsteroidModel_Converter
from espnet2.enh.separator.conformer_separator import ConformerSeparator
from espnet2.enh.separator.dprnn_separator import DPRNNSeparator
from espnet2.enh.separator.neural_beamformer import NeuralBeamformer
from espnet2.enh.separator.rnn_separator import RNNSeparator
from espnet2.enh.separator.tcn_separator import TCNSeparator
from espnet2.enh.separator.transformer_separator import TransformerSeparator
from espnet2.iterators.abs_iter_factory import AbsIterFactory
from espnet2.iterators.chunk_multiple_seq_iter_factory import ChunkIterFactory
from espnet2.samplers.unsorted_batch_sampler import UnsortedBatchSampler
from espnet2.tasks.abs_task import AbsTask
from espnet2.tasks.enh import encoder_choices
from espnet2.tasks.enh import separator_choices
from espnet2.tasks.enh import decoder_choices
from espnet2.tasks.enh import loss_wrapper_choices
from espnet2.tasks.enh import criterion_choices
from espnet2.tasks.enh import MAX_REFERENCE_NUM
from espnet2.tasks.enh import EnhancementTask as Orig_EnhancementTask
from espnet2.torch_utils.initialize import initialize
from espnet2.train.class_choices import ClassChoices
from espnet2.train.collate_fn import CommonCollateFn
from espnet2.train.dataset import ESPnetDataset
from espnet2.train.distributed_utils import DistributedOption
from espnet2.train.trainer import Trainer
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.types import str2bool
from espnet2.utils.types import str_or_none





@dataclass
class IteratorOptions:
    preprocess_fn: callable
    collate_fn: callable
    data_path_and_name_and_type: list
    shape_files: list
    batch_size: int
    batch_bins: int
    batch_type: str
    max_cache_size: float
    max_cache_fd: int
    distributed: bool
    num_batches: Optional[int]
    num_iters_per_epoch: Optional[int]
    train: bool
    num_utterances: int


class EnhancementTask(Orig_EnhancementTask, AbsTask):

    @classmethod
    def add_task_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(description="Task related")

        # NOTE(kamo): add_arguments(..., required=True) can't be used
        # to provide --print_config mode. Instead of it, do as
        # required = parser.get_default("required")

        group.add_argument(
            "--init",
            type=lambda x: str_or_none(x.lower()),
            default=None,
            help="The initialization method",
            choices=[
                "chainer",
                "xavier_uniform",
                "xavier_normal",
                "kaiming_uniform",
                "kaiming_normal",
                None,
            ],
        )

        group.add_argument(
            "--model_conf",
            action=NestedDictAction,
            default=get_default_kwargs(ESPnetEnhancementModel),
            help="The keyword arguments for model class.",
        )

        group.add_argument(
            "--criterions",
            action=NestedDictAction,
            default=[
                {
                    "name": "si_snr",
                    "conf": {},
                    "wrapper": "fixed_order",
                    "wrapper_conf": {},
                },
            ],
            help="The criterions binded with the loss wrappers.",
        )

        group = parser.add_argument_group(description="Preprocess related")
        group.add_argument(
            "--use_preprocessor",
            type=str2bool,
            default=False,
            help="Apply preprocessing to data or not",
        )

        parser.add_argument(
            "--num_utterances",
            type=int,
            default=1,
            help="Number of utterances for chunk iterator to draw chunks from.",
        )
        parser.set_defaults(iterator_type="task")

        for class_choices in cls.class_choices_list:
            # Append --<name> and --<name>_conf.
            # e.g. --encoder and --encoder_conf
            class_choices.add_arguments(group)


    @classmethod
    def build_task_iter_factory(
        cls,
        args: argparse.Namespace,
        iter_options: IteratorOptions,
        mode: str
    ) -> AbsIterFactory:
        assert check_argument_types()

        dataset = ESPnetDataset(
            iter_options.data_path_and_name_and_type,
            float_dtype=args.train_dtype,
            preprocess=iter_options.preprocess_fn,
            max_cache_size=iter_options.max_cache_size,
            max_cache_fd=iter_options.max_cache_fd,
        )
        cls.check_task_requirements(
            dataset, args.allow_variable_data_keys, train=iter_options.train
        )

        if len(iter_options.shape_files) == 0:
            key_file = iter_options.data_path_and_name_and_type[0][0]
        else:
            key_file = iter_options.shape_files[0]

        batch_sampler = UnsortedBatchSampler(batch_size=iter_options.num_utterances, key_file=key_file)
        batches = list(batch_sampler)
        if iter_options.num_batches is not None:
            batches = batches[: iter_options.num_batches]
        logging.info(f"[{mode}] dataset:\n{dataset}")

        if iter_options.distributed:
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
            if len(batches) < world_size:
                raise RuntimeError("Number of samples is smaller than world_size")
            if iter_options.batch_size < world_size:
                raise RuntimeError("batch_size must be equal or more than world_size")

            if rank < iter_options.batch_size % world_size:
                batch_size = iter_options.batch_size // world_size + 1
            else:
                batch_size = iter_options.batch_size // world_size
            num_cache_chunks = args.num_cache_chunks // world_size
            # NOTE(kamo): Split whole corpus by sample numbers without considering
            #   each of the lengths, therefore the number of iteration counts are not
            #   always equal to each other and the iterations are limitted
            #   by the fewest iterations.
            #   i.e. the samples over the counts are discarded.
            batches = batches[rank::world_size]
        else:
            batch_size = iter_options.batch_size
            num_cache_chunks = args.num_cache_chunks

        return ChunkIterFactory(
            dataset=dataset,
            batches=batches,
            seed=args.seed,
            batch_size=batch_size,
            # For chunk iterator,
            # --num_iters_per_epoch doesn't indicate the number of iterations,
            # but indicates the number of samples.
            num_samples_per_epoch=iter_options.num_iters_per_epoch,
            shuffle=iter_options.train,
            num_workers=args.num_workers,
            collate_fn=iter_options.collate_fn,
            pin_memory=args.ngpu > 0,
            chunk_length=args.chunk_length,
            chunk_shift_ratio=args.chunk_shift_ratio,
            num_cache_chunks=num_cache_chunks,
        )

    @classmethod
    def build_iter_options(
        cls,
        args: argparse.Namespace,
        distributed_option: DistributedOption,
        mode: str,
    ):
        if mode == "train":
            preprocess_fn = cls.build_preprocess_fn(args, train=True)
            collate_fn = cls.build_collate_fn(args, train=True)
            data_path_and_name_and_type = args.train_data_path_and_name_and_type
            shape_files = args.train_shape_file
            batch_size = args.batch_size
            batch_bins = args.batch_bins
            batch_type = args.batch_type
            max_cache_size = args.max_cache_size
            max_cache_fd = args.max_cache_fd
            distributed = distributed_option.distributed
            num_batches = None
            num_iters_per_epoch = args.num_iters_per_epoch
            train = True
            num_utterances = args.num_utterances

        elif mode == "valid":
            preprocess_fn = cls.build_preprocess_fn(args, train=False)
            collate_fn = cls.build_collate_fn(args, train=False)
            data_path_and_name_and_type = args.valid_data_path_and_name_and_type
            shape_files = args.valid_shape_file

            if args.valid_batch_type is None:
                batch_type = args.batch_type
            else:
                batch_type = args.valid_batch_type
            if args.valid_batch_size is None:
                batch_size = args.batch_size
            else:
                batch_size = args.valid_batch_size
            if args.valid_batch_bins is None:
                batch_bins = args.batch_bins
            else:
                batch_bins = args.valid_batch_bins
            if args.valid_max_cache_size is None:
                # Cache 5% of maximum size for validation loader
                max_cache_size = 0.05 * args.max_cache_size
            else:
                max_cache_size = args.valid_max_cache_size
            max_cache_fd = args.max_cache_fd
            distributed = distributed_option.distributed
            num_batches = None
            num_iters_per_epoch = None
            train = False
            num_utterances = 1

        elif mode == "plot_att":
            preprocess_fn = cls.build_preprocess_fn(args, train=False)
            collate_fn = cls.build_collate_fn(args, train=False)
            data_path_and_name_and_type = args.valid_data_path_and_name_and_type
            shape_files = args.valid_shape_file
            batch_type = "unsorted"
            batch_size = 1
            batch_bins = 0
            num_batches = args.num_att_plot
            max_cache_fd = args.max_cache_fd
            # num_att_plot should be a few sample ~ 3, so cache all data.
            max_cache_size = np.inf if args.max_cache_size != 0.0 else 0.0
            # always False because plot_attention performs on RANK0
            distributed = False
            num_iters_per_epoch = None
            train = False
            num_utterances = 1
        else:
            raise NotImplementedError(f"mode={mode}")

        return IteratorOptions(
            preprocess_fn=preprocess_fn,
            collate_fn=collate_fn,
            data_path_and_name_and_type=data_path_and_name_and_type,
            shape_files=shape_files,
            batch_type=batch_type,
            batch_size=batch_size,
            batch_bins=batch_bins,
            num_batches=num_batches,
            max_cache_size=max_cache_size,
            max_cache_fd=max_cache_fd,
            distributed=distributed,
            num_iters_per_epoch=num_iters_per_epoch,
            train=train,
            num_utterances=num_utterances,
        )