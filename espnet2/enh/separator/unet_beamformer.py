from collections import OrderedDict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_complex.tensor import ComplexTensor

from espnet2.enh.layers.complex_utils import is_complex
from espnet2.enh.layers.complex_utils import new_complex_like
from espnet2.enh.separator.abs_separator import AbsSeparator

EPS = torch.finfo(torch.double).eps


class Conv2DBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        layers: int = 1,
        bn_func=nn.BatchNorm2d,
        dropout_rate=0.0,
        act_func=nn.LeakyReLU,
        skip_res: bool = False,
        transpose: bool = False,
    ):
        super().__init__()
        # modules
        self.blocks = nn.ModuleList()
        self.skip_res = skip_res
        conv_layer = nn.ConvTranspose2d if transpose else nn.Conv2d

        for idx in range(layers):
            in_ = in_channels if idx == 0 else out_channels
            self.blocks.append(
                nn.Sequential(
                    *[
                        conv_layer(
                            in_,
                            out_channels,
                            kernel_size,
                            stride=stride,
                            padding=padding,
                        ),
                        bn_func(out_channels),
                        nn.Dropout(dropout_rate),
                        act_func(),
                    ]
                )
            )

    def forward(self, x: torch.tensor) -> torch.tensor:
        temp = x
        for idx, block in enumerate(self.blocks):
            x = block(x)

        if temp.size() != x.size() or self.skip_res:
            return x
        else:
            return x + temp


class UnetBeamformer(AbsSeparator):
    def __init__(
        self,
        input_dim: int,
        num_spk: int = 2,
        in_channels: int = 8,
        conv_channels: Optional[List[int]] = None,
        conv_kernel_size: Optional[List[List[int]]] = None,
        conv_stride: Optional[List[List[int]]] = None,
        dropout_rate: float = 0.5,
        nonlinear: str = "relu",
        use_noise_mask: bool = False,
    ):
        super().__init__()

        self._num_spk = num_spk
        self.use_noise_mask = use_noise_mask

        assert len(conv_channels) == len(conv_kernel_size) == len(conv_stride)

        channels = list(zip([in_channels] + conv_channels[:-1], conv_channels))
        self.encoder = torch.nn.ModuleList(
            [
                Conv2DBlock(
                    in_channel,
                    out_channel,
                    conv_kernel_size[block],
                    stride=conv_stride[block],
                    dropout_rate=dropout_rate,
                )
                for block, (in_channel, out_channel) in enumerate(channels)
            ]
        )
        self.decoder = torch.nn.ModuleList(
            reversed(
                [
                    Conv2DBlock(
                        out_channel,
                        in_channel,
                        conv_kernel_size[block],
                        stride=conv_stride[block],
                        transpose=True,
                        dropout_rate=dropout_rate,
                    )
                    for block, (in_channel, out_channel) in enumerate(channels[1:])
                ]
            )
        )
        self.post_conv = nn.ConvTranspose2d(
            channels[0][1], channels[0][0], conv_kernel_size[0], stride=conv_stride[0]
        )

        num_outputs = self.num_spk + 1 if use_noise_mask else self.num_spk
        self.linear = torch.nn.ModuleList(
            [torch.nn.Linear(input_dim * 2, input_dim * 2) for _ in range(num_outputs)]
        )

        if nonlinear not in ("sigmoid", "relu", "tanh", "none"):
            raise ValueError("Not supporting nonlinear={}".format(nonlinear))

        self.nonlinear = {
            "sigmoid": torch.nn.Sigmoid(),
            "relu": torch.nn.ReLU(),
            "tanh": torch.nn.Tanh(),
            "none": torch.nn.Identity(),
        }[nonlinear]

    def forward(self, xs: Union[torch.Tensor, ComplexTensor], ilens: torch.Tensor):
        """Unet-based beamforming.

        Args:
            xs: (Batch, Frames, Channels, Freq)
            ilens: (Batch,)
        Returns:
            enhanced: beamformed signal. (Batch, Frames, Freq)
            ilens: (Batch,)
            others predicted data, e.g. masks: OrderedDict[
                'mask_spk1': torch.Tensor(Batch, Frames, Freq),
                'mask_spk2': torch.Tensor(Batch, Frames, Freq),
                ...
                'mask_spkn': torch.Tensor(Batch, Frames, Freq),
            ]
        """
        assert is_complex(xs), type(xs)
        xs = xs[:, :, :4]
        # (B, T, C, F'=2F) -> (B, C, T, F'=2F)
        x = torch.cat([xs.real, xs.imag], dim=-1).permute(0, 2, 1, 3)
        TF = x.shape[2:]

        encoder_cache = []
        num_encoders = len(self.encoder)
        for i, conv_enc in enumerate(self.encoder):
            x = conv_enc(x)
            if i < num_encoders - 1:
                encoder_cache.append(x)

        for j, conv_dec in enumerate(self.decoder):
            x = conv_dec(x)
            res = encoder_cache.pop(-1)
            x = F.interpolate(
                x, size=res.shape[2:], mode="bilinear", align_corners=False
            )
            x = x + res

        x = self.post_conv(x)
        # (B, C, T, F'=2F)
        x = F.interpolate(x, size=TF, mode="bilinear", align_corners=False)

        # (B, C, T, F) -> (B, T, C, F)
        masks = [
            new_complex_like(xs, self.nonlinear(linear(x)).chunk(2, dim=-1)).permute(0, 2, 1, 3)
            for linear in self.linear
        ]

        # (B, T, F)
        x = [(m * xs).sum(dim=2) for m in masks[:self.num_spk]]

        others = OrderedDict(
            zip(["mask_spk{}".format(i + 1) for i in range(self.num_spk)], masks[:self.num_spk])
        )
        if self.use_noise_mask:
            others["mask_noise1"] = masks[self.num_spk]

        return x, ilens, others

    @property
    def num_spk(self):
        return self._num_spk
