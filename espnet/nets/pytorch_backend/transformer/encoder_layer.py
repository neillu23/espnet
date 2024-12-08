#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Encoder self-attention layer definition."""

import torch
from torch import nn
from s3prl.upstream.wav2vec2.cond_blocks import CC, DoubleCC, TCAC, DoubleTCAC
import logging
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm


class EncoderLayer(nn.Module):
    """Encoder layer module.

    Args:
        size (int): Input dimension.
        self_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` or `RelPositionMultiHeadedAttention` instance
            can be used as the argument.
        feed_forward (torch.nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward`, `MultiLayeredConv1d`, or `Conv1dLinear` instance
            can be used as the argument.
        dropout_rate (float): Dropout rate.
        normalize_before (bool): Whether to use layer_norm before the first block.
        concat_after (bool): Whether to concat attention layer's input and output.
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied. i.e. x -> x + att(x)
        stochastic_depth_rate (float): Proability to skip this layer.
            During training, the layer may skip residual computation and return input
            as-is with given probability.
    """

    def __init__(
        self,
        size,
        self_attn,
        feed_forward,
        dropout_rate,
        normalize_before=True,
        concat_after=False,
        stochastic_depth_rate=0.0,
        embed_condition: bool = False,
        embed_condition_size: int = 0,
        embed_condition_method: str = "CC",
    ):
        """Construct an EncoderLayer object."""
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = LayerNorm(size)
        self.norm2 = LayerNorm(size)
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear = nn.Linear(size + size, size)
        self.stochastic_depth_rate = stochastic_depth_rate
        self.embed_condition = embed_condition
        self.embed_condition_size = embed_condition_size
        self.embed_condition_method = embed_condition_method
        if self.embed_condition:
            # logging.info("size: {}".format(size))
            # import pdb; pdb.set_trace()
            if self.embed_condition_method == "CC":
                self.condition_layer = CC(size, embed_condition_size)
            elif self.embed_condition_method == "DoubleCC":
                self.condition_layer = DoubleCC(size, embed_condition_size)
            elif self.embed_condition_method == "TCAC":
                self.condition_layer = TCAC(size, embed_condition_size)
            elif self.embed_condition_method == "DoubleTCAC":
                self.condition_layer = DoubleTCAC(size, embed_condition_size)


    def forward(self, x, mask, condition_features=None, cache=None):
        """Compute encoded features.

        Args:
            x_input (torch.Tensor): Input tensor (#batch, time, size).
            mask (torch.Tensor): Mask tensor for the input (#batch, 1, time).
            cache (torch.Tensor): Cache tensor of the input (#batch, time - 1, size).

        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, 1, time).

        """
        skip_layer = False
        # with stochastic depth, residual connection `x + f(x)` becomes
        # `x <- x + 1 / (1 - p) * f(x)` at training time.
        stoch_layer_coeff = 1.0
        if self.training and self.stochastic_depth_rate > 0:
            skip_layer = torch.rand(1).item() < self.stochastic_depth_rate
            stoch_layer_coeff = 1.0 / (1 - self.stochastic_depth_rate)

        if skip_layer:
            if cache is not None:
                x = torch.cat([cache, x], dim=1)
            return x, mask

        residual = x
        if self.normalize_before:
            x = self.norm1(x)

        if cache is None:
            x_q = x
        else:
            assert cache.shape == (x.shape[0], x.shape[1] - 1, self.size)
            x_q = x[:, -1:, :]
            residual = residual[:, -1:, :]
            mask = None if mask is None else mask[:, -1:, :]

        if self.concat_after:
            x_concat = torch.cat((x, self.self_attn(x_q, x, x, mask)), dim=-1)
            x = residual + stoch_layer_coeff * self.concat_linear(x_concat)
        else:
            x = self.self_attn(x_q, x, x, mask)
            x = stoch_layer_coeff * self.dropout(x)
            # logging.info("x in EncoderLayer: {}".format(x))
            if self.embed_condition:
                if self.embed_condition_method in ["TCAC", "DoubleTCAC"]:
                    x = x.permute(1, 0, 2)
                    if self.embed_condition_method in ["DoubleTCAC"]:
                        condition_features[0] = condition_features[0].permute(1, 0, 2)
                        if condition_features[1] is not None:
                            condition_features[1] = condition_features[1].permute(1, 0, 2)
                    else:
                        condition_features = condition_features.permute(1, 0, 2)
                    # logging.info("x shape: {}".format(x.shape))
                    # logging.info("condition_features shape: {}".format(condition_features.shape))
                x = self.condition_layer(x, condition_features)
                if self.embed_condition_method in ["TCAC", "DoubleTCAC"]:
                    x = x.permute(1, 0, 2)
                    if self.embed_condition_method in ["DoubleTCAC"]:
                        condition_features[0] = condition_features[0].permute(1, 0, 2)
                        if condition_features[1] is not None:
                            condition_features[1] = condition_features[1].permute(1, 0, 2)
                    else:
                        condition_features = condition_features.permute(1, 0, 2)
            x = residual + x
        if not self.normalize_before:
            x = self.norm1(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        x = residual + stoch_layer_coeff * self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm2(x)

        if cache is not None:
            x = torch.cat([cache, x], dim=1)

        # logging.info("x out of EncoderLayer: {}".format(x))
        return x, mask, condition_features
