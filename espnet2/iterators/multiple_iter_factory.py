import logging
from typing import Callable, Collection, Iterator

import numpy as np
from typeguard import typechecked

from espnet2.iterators.abs_iter_factory import AbsIterFactory


class MultipleIterFactory(AbsIterFactory):
    @typechecked
    def __init__(
        self,
        build_funcs: Collection[Callable[[], AbsIterFactory]],
        seed: int = 0,
        shuffle: bool = False,
    ):
        self.build_funcs = list(build_funcs)
        self.seed = seed
        self.shuffle = shuffle

    def build_iter(self, epoch: int, shuffle: bool = None) -> Iterator:
        if shuffle is None:
            shuffle = self.shuffle

        build_funcs = list(self.build_funcs)

        if shuffle:
            np.random.RandomState(epoch + self.seed).shuffle(build_funcs)

        split_num = len(build_funcs)
        assert(split_num % 8 == 0)

        iter_factory_nums = split_num // 8
        iter_factory_group = epoch % iter_factory_nums
        for i in range(iter_factory_nums):
            build_func = build_funcs[i + iter_factory_group * iter_factory_nums]
            logging.info(f"Building {i}th iter-factory...")
            iter_factory = build_func()
            assert isinstance(iter_factory, AbsIterFactory), type(iter_factory)
            yield from iter_factory.build_iter(epoch, shuffle)
