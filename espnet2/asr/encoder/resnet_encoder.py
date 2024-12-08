# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Transformer encoder definition."""

from typing import List, Optional, Tuple, Union

import logging
import numpy as np
import torch
import torch.nn as nn
# from typeguard import check_argument_types


from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask

from espnet.nets.pytorch_backend.resnet1d.dc1d_blocks import DC1dEncBlock
from espnet.nets.pytorch_backend.resnet1d.res2net1d_blocks import Res2Net1dBasicBlock, Res2Net1dBNBlock
from espnet.nets.pytorch_backend.resnet1d.resnet1d_blocks import (
    ResNet1dBasicBlock,
    ResNet1dBNBlock,
    ResNet1dEndpoint,
    SEResNet1dBasicBlock,
    SEResNet1dBNBlock,
)


from espnet.nets.pytorch_backend.resnet1d.layers.activation_factory import ActivationFactory as AF
from espnet.nets.pytorch_backend.resnet1d.layers.norm_layer_factory import NormLayer1dFactory as NLF

# from ..utils import seq_lengths_to_mask


# from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
# from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
# from espnet.nets.pytorch_backend.transformer.encoder_layer import EncoderLayer
# from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
# from espnet.nets.pytorch_backend.transformer.multi_layer_conv import (
#     Conv1dLinear,
#     MultiLayeredConv1d,
# )
# from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
#     PositionwiseFeedForward,
# )
# from espnet.nets.pytorch_backend.transformer.repeat import repeat
# from espnet.nets.pytorch_backend.transformer.subsampling import (
#     Conv1dSubsampling2,
#     Conv2dSubsampling,
#     Conv2dSubsampling1,
#     Conv2dSubsampling2,
#     Conv2dSubsampling6,
#     Conv2dSubsampling8,
#     TooShortUttError,
#     check_short_utt,
# )


def seq_lengths_to_mask(lengths, max_length=None, dtype=None, time_dim=1):
    """Creates a binary masks indicating the valid values in a sequence.

    Args:
      lengths: sequence lengths with shape=(batch,). If None, it returns None
      max_length: maximum length of the sequence.
      dtype: dtype for the mask.
      time_dim: dimension > 0 corresponding to time in the mask. This will
                return a view of the mask which will adapt to the shape
                of the tensor where we want to apply the mask.
                This has to be a positive integer.

    Returns:
      Binary mask with shape=(batch,...,max_length) or None
    """
    if lengths is None:
        return None

    assert time_dim > 0
    assert lengths.dim() == 1

    if max_length is None:
        max_length = lengths.max()
    idx = torch.arange(max_length, dtype=lengths.dtype, device=lengths.device)

    # compute mask shape=(batch, max_length)
    mask = idx.unsqueeze(0) < lengths.unsqueeze(1)

    # view to match the tensor where we want to apply the mask
    if time_dim > 1:
        shape = [1] * (time_dim + 1)
        shape[0] = lengths.size(0)
        shape[time_dim] = -1
        mask = mask.view(*shape)

    # change dtype if needed
    if dtype is not None:
        mask = mask.to(dtype)

    return mask



class ResNetEncoder(AbsEncoder):
    """Transformer encoder module.

    Args:
        input_size: input dim
        output_size: dimension of attention
        attention_heads: the number of heads of multi head attention
        linear_units: the number of units of position-wise feed forward
        num_blocks: the number of decoder blocks
        dropout_rate: dropout rate
        attention_dropout_rate: dropout rate in attention
        positional_dropout_rate: dropout rate after adding positional encoding
        input_layer: input layer type
        pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
        normalize_before: whether to use layer_norm before the first block
        concat_after: whether to concat attention layer's input and output
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied.
            i.e. x -> x + att(x)
        positionwise_layer_type: linear of conv1d
        positionwise_conv_kernel_size: kernel size of positionwise conv1d layer
        padding_idx: padding_idx for input_layer=embed
    """

    def __init__(
        self,
        input_size: int,
        in_conv_channels: int = 128,
        in_kernel_size: int = 3,
        in_stride: int = 1,
        resb_type: str = "basic",
        resb_repeats: List[int] = [1, 1, 1],
        resb_channels: Union[int, List[int]]  = 128,
        resb_kernel_sizes: Union[int, List[int]] = 3,
        resb_strides: Union[int, List[int]] = 2,
        resb_dilations: Union[int, List[int]] = 1,
        resb_groups: int = 1,
        head_channels: int = 0,
        hid_act: str = "relu",
        head_act: Optional[str] = None,
        dropout_rate: float = 0,
        drop_connect_rate: float = 0,
        se_r: int = 16,
        res2net_width_factor: int = 1,
        res2net_scale: int = 4,
        multilayer: bool = False,
        multilayer_concat: bool = False,
        endpoint_channels: int = None,
        endpoint_layers: Optional[Union[int, List[int]]] = None,
        endpoint_scale_layer: int = -1,
        use_norm: bool = True,
        norm_layer: Optional[str] = None,
        norm_before: bool = True,
        upsampling_mode: str = "nearest",
    ):
        # assert check_argument_types()
        super().__init__()

        in_feats = input_size

        self.resb_type = resb_type
        bargs = {}  # block's extra arguments
        if resb_type == "basic":
            self._block = ResNet1dBasicBlock
        elif resb_type == "bn":
            self._block = ResNet1dBNBlock
        elif resb_type == "sebasic":
            self._block = SEResNet1dBasicBlock
            bargs["se_r"] = se_r
        elif resb_type == "sebn":
            self._block = SEResNet1dBNBlock
            bargs["se_r"] = se_r
        elif resb_type in ["res2basic", "seres2basic", "res2bn", "seres2bn"]:
            bargs["width_factor"] = res2net_width_factor
            bargs["scale"] = res2net_scale
            if resb_type in ["seres2basic", "seres2bn"]:
                bargs["se_r"] = se_r
            if resb_type in ["res2basic", "seres2basic"]:
                self._block = Res2Net1dBasicBlock
            else:
                self._block = Res2Net1dBNBlock

        self.in_feats = in_feats
        self.in_conv_channels = in_conv_channels
        self.in_kernel_size = in_kernel_size
        self.in_stride = in_stride
        num_superblocks = len(resb_repeats)
        self.resb_repeats = resb_repeats
        self.resb_channels = self._standarize_resblocks_param(
            resb_channels, num_superblocks, "resb_channels"
        )
        self.resb_kernel_sizes = self._standarize_resblocks_param(
            resb_kernel_sizes, num_superblocks, "resb_kernel_sizes"
        )
        self.resb_strides = self._standarize_resblocks_param(
            resb_strides, num_superblocks, "resb_strides"
        )
        self.resb_dilations = self._standarize_resblocks_param(
            resb_dilations, num_superblocks, "resb_dilations"
        )
        self.resb_groups = resb_groups
        self.head_channels = head_channels
        self.hid_act = hid_act
        self.head_act = head_act
        self.dropout_rate = dropout_rate
        self.drop_connect_rate = drop_connect_rate
        self.use_norm = use_norm
        self.norm_before = norm_before
        self.se_r = se_r
        self.res2net_width_factor = res2net_width_factor
        self.res2net_scale = res2net_scale
        self.norm_layer = norm_layer
        norm_groups = None
        if norm_layer == "group-norm":
            norm_groups = min(np.min(resb_channels) // 2, 32)
            norm_groups = max(norm_groups, resb_groups)
        self._norm_layer = NLF.create(norm_layer, norm_groups)

        # stem block
        self.in_block = DC1dEncBlock(
            in_feats,
            in_conv_channels,
            in_kernel_size,
            stride=in_stride,
            activation=hid_act,
            dropout_rate=dropout_rate,
            use_norm=use_norm,
            norm_layer=self._norm_layer,
            norm_before=norm_before,
        )
        self._context = self.in_block.context
        self._downsample_factor = self.in_block.stride

        cur_in_channels = in_conv_channels
        total_blocks = np.sum(self.resb_repeats)

        # middle blocks
        self.blocks = nn.ModuleList([])
        k = 0
        self.resb_scales = []
        for i in range(num_superblocks):
            blocks_i = nn.ModuleList([])
            repeats_i = self.resb_repeats[i]
            channels_i = self.resb_channels[i]
            stride_i = self.resb_strides[i]
            kernel_size_i = self.resb_kernel_sizes[i]
            dilation_i = self.resb_dilations[i]
            # if there is downsampling the dilation of the first block
            # is set to 1
            dilation_i1 = dilation_i if stride_i == 1 else 1
            drop_i = drop_connect_rate * k / (total_blocks - 1)
            block_i1 = self._block(
                cur_in_channels,
                channels_i,
                kernel_size_i,
                stride=stride_i,
                dilation=dilation_i1,
                groups=self.resb_groups,
                activation=hid_act,
                dropout_rate=dropout_rate,
                drop_connect_rate=drop_i,
                use_norm=use_norm,
                norm_layer=self._norm_layer,
                norm_before=norm_before,
                **bargs,
            )

            blocks_i.append(block_i1)
            k += 1
            self._context += block_i1.context * self._downsample_factor
            self._downsample_factor *= block_i1.downsample_factor
            self.resb_scales.append(self._downsample_factor)

            for j in range(repeats_i - 1):
                drop_i = drop_connect_rate * k / (total_blocks - 1)
                block_ij = self._block(
                    channels_i,
                    channels_i,
                    kernel_size_i,
                    stride=1,
                    dilation=dilation_i,
                    groups=self.resb_groups,
                    activation=hid_act,
                    dropout_rate=dropout_rate,
                    drop_connect_rate=drop_i,
                    use_norm=use_norm,
                    norm_layer=self._norm_layer,
                    norm_before=norm_before,
                    **bargs,
                )
                blocks_i.append(block_ij)
                k += 1
                self._context += block_ij.context * self._downsample_factor
            self.blocks.append(blocks_i)

            cur_in_channels = channels_i

        if multilayer:
            if endpoint_layers is None:
                # if is None all layers are endpoints
                endpoint_layers = [i + 1 for i in range(num_superblocks)]

            if endpoint_channels is None:
                # if None, the number of endpoint channels matches the one of the endpoint level
                endpoint_channels = self.resb_channels[endpoint_scale_layer]

            # which layers are enpoints
            self.is_endpoint = [
                True if i + 1 in endpoint_layers else False
                for i in range(num_superblocks)
            ]
            # which endpoints have a projection layer ResNet1dEndpoint
            self.has_endpoint_block = [False] * num_superblocks
            # relates endpoint layers to their ResNet1dEndpoint object
            self.endpoint_block_idx = [0] * num_superblocks
            endpoint_scale = self.resb_scales[endpoint_scale_layer]
            endpoint_blocks = nn.ModuleList([])
            cur_endpoint = 0
            in_concat_channels = 0
            for i in range(num_superblocks):
                if self.is_endpoint[i]:
                    if multilayer_concat:
                        out_channels = self.resb_channels[i]
                        if self.resb_scales[i] != endpoint_scale:
                            self.has_endpoint_block[i] = True

                        # if self.resb_channels[i] != endpoint_channels:
                        #     out_channels = endpoint_channels
                        #     self.has_endpoint_block[i] = True

                        in_concat_channels += out_channels
                    else:
                        self.has_endpoint_block[i] = True
                        out_channels = endpoint_channels

                    if self.has_endpoint_block[i]:
                        endpoint_i = ResNet1dEndpoint(
                            self.resb_channels[i],
                            out_channels,
                            in_scale=self.resb_scales[i],
                            scale=endpoint_scale,
                            activation=hid_act,
                            upsampling_mode=upsampling_mode,
                            norm_layer=self._norm_layer,
                            norm_before=norm_before,
                        )
                        self.endpoint_block_idx[i] = cur_endpoint
                        endpoint_blocks.append(endpoint_i)
                        cur_endpoint += 1

            self.endpoint_blocks = endpoint_blocks
            if multilayer_concat:
                self.concat_endpoint_block = ResNet1dEndpoint(
                    in_concat_channels,
                    endpoint_channels,
                    in_scale=1,
                    scale=1,
                    activation=hid_act,
                    norm_layer=self._norm_layer,
                    norm_before=norm_before,
                )
        else:
            endpoint_channels = self.resb_channels[-1]

        self.multilayer = multilayer
        self.multilayer_concat = multilayer_concat
        self.endpoint_channels = endpoint_channels
        self.endpoint_layers = endpoint_layers
        self.endpoint_scale_layer = endpoint_scale_layer
        self.upsampling_mode = upsampling_mode

        # head feature block
        if self.head_channels > 0:
            self.head_block = DC1dEncBlock(
                cur_in_channels,
                head_channels,
                kernel_size=1,
                stride=1,
                activation=head_act,
                use_norm=False,
                norm_before=norm_before,
            )

        self._init_weights(hid_act)
        logging.info("head_channels: %d" % self.head_channels)
        logging.info("endpoint_channels: %d" % self.endpoint_channels)
        self._output_size = (
            self.head_channels if self.head_channels > 0 else self.endpoint_channels
        )
        

    def _init_weights(self, hid_act):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                if isinstance(hid_act, str):
                    act_name = hid_act
                if isinstance(hid_act, dict):
                    act_name = hid_act["name"]
                if act_name == "swish":
                    act_name = "relu"
                try:
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity=act_name
                    )
                except:
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def _standarize_resblocks_param(p, num_blocks, p_name):
        if isinstance(p, int):
            p = [p] * num_blocks
        elif isinstance(p, list):
            if len(p) == 1:
                p = p * num_blocks

            assert len(p) == num_blocks, "len(%s)(%d)!=%d" % (
                p_name,
                len(p),
                num_blocks,
            )
        else:
            raise TypeError("wrong type for param {}={}".format(p_name, p))

        return p

    def _compute_out_size(self, in_size):
        out_size = int((in_size - 1) // self.in_stride + 1)

        if self.multilayer:
            strides = self.resb_strides[self.endpoint_scale_layer]
        else:
            strides = self.resb_strides

        for stride in strides:
            out_size = int((out_size - 1) // stride + 1)

        return out_size

    def in_context(self):
        return (self._context, self._context)

    def in_shape(self):
        return (None, self.in_feats, None)

    # def out_shape(self, in_shape=None):
    #     out_channels = (
    #         self.head_channels if self.head_channels > 0 else self.endpoint_channels
    #     )
    #     if in_shape is None:
    #         return (None, out_channels, None)

    #     assert len(in_shape) == 3
    #     if in_shape[2] is None:
    #         T = None
    #     else:
    #         T = self._compute_out_size(in_shape[2])

    #     return (in_shape[0], out_channels, T)

    @staticmethod
    def _match_lens(endpoints):
        lens = [e.shape[-1] for e in endpoints]
        min_len = min(lens)
        for i in range(len(endpoints)):
            if lens[i] > min_len:
                t_start = (lens[i] - min_len) // 2
                t_end = t_start + min_len
                endpoints[i] = endpoints[i][:, :, t_start:t_end]

        return endpoints

    @staticmethod
    def _update_mask(x, x_lengths, x_mask=None):
        if x_lengths is None:
            return None

        if x_mask is not None and x.size(-1) == x_mask.size(-1):
            return x_mask

        return seq_lengths_to_mask(x_lengths, x.size(-1), dtype=torch.float32, time_dim=2)

        # return (~make_pad_mask(x_lengths)[:, None, :]).to(x.device)

    def output_size(self) -> int:
        return self._output_size



    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        condition_features: torch.Tensor = None,
        # prev_states: torch.Tensor = None,
        # ctc: CTC = None,
        # return_all_hs: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Embed positions in tensor.

        Args:
            xs_pad: input tensor (B, L, D)
            ilens: input length (B)
            prev_states: Not to be used now.
            ctc (CTC): ctc module for intermediate CTC loss
            return_all_hs (bool): whether to return all hidden states

        Returns:
            position embedded tensor and mask



        Args:
           x: input tensor of size=(batch, C, time)
           x_lengths:  it contains the lengths of the sequences.
        Returns:
           Tensor with output logits of size=(batch, out_units) if out_units>0,
           otherwise, it returns tensor of represeantions of size=(batch, Cout, out_time)

        """

        x = torch.permute(xs_pad, (0, 2, 1))
        x_lengths = ilens

        # logging.info(f"input x: {x.shape}")
        # logging.info(f"input x_lengths: {x_lengths}")
        x_mask = self._update_mask(x, x_lengths)
        x = self.in_block(x, x_mask=x_mask)
        endpoints = []

        for i, superblock in enumerate(self.blocks):
            for j, block in enumerate(superblock):
                # logging.info(f"block {i} {j} x: {x.shape}")
                
                # logging.info(f"block {i} {j} x_mask: {x_mask.shape}")
                # logging.info(x_mask)

                x_mask = self._update_mask(x, x_lengths, x_mask)
                x = block(x, x_mask=x_mask)

            if self.multilayer and self.is_endpoint[i]:
                endpoint_i = x
                if self.has_endpoint_block[i]:
                    idx = self.endpoint_block_idx[i]
                    endpoint_i = self.endpoint_blocks[idx](endpoint_i)

                endpoints.append(endpoint_i)

        if self.multilayer:
            endpoints = self._match_lens(endpoints)
            if self.multilayer_concat:
                try:
                    x = torch.cat(endpoints, dim=1)
                except:
                    for k in range(len(endpoints)):
                        print("epcat ", k, endpoints[k].shape, flush=True)

                x = self.concat_endpoint_block(x)
            else:
                x = torch.mean(torch.stack(endpoints), 0)

        if self.head_channels > 0:
            x_mask = self._update_mask(x, x_lengths, x_mask)
            x = self.head_block(x)

        olens = x_mask.squeeze(1).sum(1)
        return x, olens, None


    def forward_hid_feats(self, x, x_lengths=None, layers=None, return_output=False):

        assert layers is not None or return_output
        if layers is None:
            layers = []

        if return_output:
            last_layer = len(self.blocks) + 1
        else:
            last_layer = max(layers)

        h = []
        x = self.in_block(x)
        if 0 in layers:
            h.append(x)

        endpoints = []
        for i, superblock in enumerate(self.blocks):
            for j, block in enumerate(superblock):
                x = block(x)

            if i + 1 in layers:
                h.append(x)

            if return_output and self.multilayer and self.is_endpoint[i]:
                endpoint_i = x
                if self.has_endpoint_block[i]:
                    idx = self.endpoint_block_idx[i]
                    endpoint_i = self.endpoint_blocks[idx](endpoint_i)
                endpoints.append(endpoint_i)

            if last_layer == i + 1:
                break

        if not return_output:
            return h

        if self.multilayer:
            if self.multilayer_concat:
                x = torch.cat(endpoints, dim=1)
                x = self.concat_endpoint_block(x)
            else:
                x = torch.mean(torch.stack(endpoints), 0)

        if self.head_channels > 0:
            x = self.head_block(x)

        return h, x


        # masks = (~make_pad_mask(ilens)[:, None, :]).to(xs_pad.device)

        # if self.embed is None:
        #     xs_pad = xs_pad
        # elif (
        #     isinstance(self.embed, Conv2dSubsampling)
        #     or isinstance(self.embed, Conv1dSubsampling2)
        #     or isinstance(self.embed, Conv2dSubsampling1)
        #     or isinstance(self.embed, Conv2dSubsampling2)
        #     or isinstance(self.embed, Conv2dSubsampling6)
        #     or isinstance(self.embed, Conv2dSubsampling8)
        # ):
        #     short_status, limit_size = check_short_utt(self.embed, xs_pad.size(1))
        #     if short_status:
        #         raise TooShortUttError(
        #             f"has {xs_pad.size(1)} frames and is too short for subsampling "
        #             + f"(it needs more than {limit_size} frames), return empty results",
        #             xs_pad.size(1),
        #             limit_size,
        #         )
        #     xs_pad, masks = self.embed(xs_pad, masks)
        # else:
        #     xs_pad = self.embed(xs_pad)

        # intermediate_outs = []
        # if len(self.interctc_layer_idx) == 0:
        #     for layer_idx, encoder_layer in enumerate(self.encoders):
        #         xs_pad, masks, condition_features = encoder_layer(xs_pad, masks, condition_features)
        #         if return_all_hs:
        #             if isinstance(xs_pad, tuple):
        #                 intermediate_outs.append(xs_pad[0])
        #             else:
        #                 intermediate_outs.append(xs_pad)
        # else:
        #     for layer_idx, encoder_layer in enumerate(self.encoders):
        #         xs_pad, masks, condition_features = encoder_layer(xs_pad, masks, condition_features)

        #         if layer_idx + 1 in self.interctc_layer_idx:
        #             encoder_out = xs_pad

        #             # intermediate outputs are also normalized
        #             if self.normalize_before:
        #                 encoder_out = self.after_norm(encoder_out)

        #             intermediate_outs.append((layer_idx + 1, encoder_out))

        #             if self.interctc_use_conditioning:
        #                 ctc_out = ctc.softmax(encoder_out)
        #                 xs_pad = xs_pad + self.conditioning_layer(ctc_out)

        # if self.normalize_before:
        #     xs_pad = self.after_norm(xs_pad)

        # olens = masks.squeeze(1).sum(1)
        # if len(intermediate_outs) > 0:
        #     return (xs_pad, intermediate_outs), olens, None
        # return xs_pad, olens, None


