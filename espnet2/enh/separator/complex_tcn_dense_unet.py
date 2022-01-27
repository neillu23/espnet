#!/usr/bin/env python

#Copyright [2021] [Zhong-Qiu Wang] (The Ohio State University)

#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at

#http://www.apache.org/licenses/LICENSE-2.0

#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from espnet2.enh.layers.complex_utils import is_complex
from espnet2.enh.separator.abs_separator import AbsSeparator


def get_norm(use_batchnorm, num_features, ndims=4):
    if use_batchnorm == 2:
        norm = nn.GroupNorm(1, num_features, eps=1e-8, affine=True)
    elif use_batchnorm == 3:
        norm = nn.GroupNorm(num_features, num_features, eps=1e-5, affine=True)
    elif use_batchnorm == 0:
        norm = nn.Identity()
    else:
        raise ValueError(use_batchnorm)
    return norm


def get_actfn(use_act):
    if use_act == 1:
        actfn = nn.ELU()
    elif use_act == 0:
        actfn = nn.Identity()
    else:
        raise ValueError(use_act)
    return actfn


def get_padding_size(kernel_size, dilation):
    if kernel_size % 2 != 1: raise
    padding = (kernel_size // 2) * dilation
    return padding


class ConvBNReLUBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        use_batchnorm=2,
        use_deconv=0,
        use_act=1,
        use_convbnrelu=1,
        memory_efficient=0,
        ndims=4,
        use_depthwise=False,
    ):
        super(ConvBNReLUBlock, self).__init__()
        if use_depthwise:
            raise ValueError("use_depthwise")
        else:
            if ndims == 4:
                module_name = nn.ConvTranspose2d if use_deconv == 1 else nn.Conv2d
            elif ndims == 3:
                module_name = nn.ConvTranspose1d if use_deconv == 1 else nn.Conv1d
            else:
                raise ValueError("ndims")
            conv = module_name(
                in_channels, out_channels, kernel_size,
                stride=stride, padding=padding, dilation=dilation,
            )
            conv.bias.data[:] = 0.0
        self.add_module('conv', conv)
        if use_convbnrelu == 0:
            #bnreluconv
            norm = get_norm(use_batchnorm, in_channels, ndims=ndims)
        else:
            #convbnrelu or convrelubn
            norm = get_norm(use_batchnorm, out_channels, ndims=ndims)
        self.add_module('norm', norm)
        self.actfn = get_actfn(use_act)
        self.use_convbnrelu = use_convbnrelu
        self.memory_efficient = memory_efficient

    def __getitem__(self, key):
        return getattr(self, key)

    def forward(self, *h):
        def f_(*h_):
            h_ = torch.cat(h_, dim=1) if len(h_) > 1 else h_[0]
            if self.use_convbnrelu == 2:
                #convrelubn
                h_ = self['conv'](h_)
                return self['norm'](self.actfn(h_))
            elif self.use_convbnrelu == 1:
                #convbnrelu
                h_ = self['conv'](h_)
                return self.actfn(self['norm'](h_))
            else:
                #bnreluconv
                h_ = self['conv'](self.actfn(self['norm'](h_)))
                return h_
        return cp.checkpoint(f_, *h) if self.memory_efficient else f_(*h)


class Conv2dBNReLUBlock(ConvBNReLUBlock):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        use_batchnorm=2,
        use_deconv=0,
        use_act=1,
        use_convbnrelu=1,
        memory_efficient=0,
        use_depthwise=False,
    ):
        super(Conv2dBNReLUBlock, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            use_batchnorm=use_batchnorm,
            use_deconv=use_deconv,
            use_act=use_act,
            use_convbnrelu=use_convbnrelu,
            memory_efficient=memory_efficient,
            ndims=4,
            use_depthwise=use_depthwise
        )
        pass


class FreqMapping(nn.Module):
    def __init__(
        self,
        in_channels,
        growth_rate,
        n_freqs,
        use_batchnorm=2,
        use_act=1,
        use_convbnrelu=1,
        memory_efficient=0,
        use_depthwise=False,
    ):
        super(FreqMapping, self).__init__()

        self.convbnrelu1 = Conv2dBNReLUBlock(
            in_channels, growth_rate, kernel_size=(1,1),
            stride=(1,1), padding=(0,0), dilation=(1,1),
            use_batchnorm=use_batchnorm, use_deconv=0, use_act=use_act,
            use_convbnrelu=use_convbnrelu, memory_efficient=0,
            use_depthwise=use_depthwise,
        )

        self.convbnrelu2 = Conv2dBNReLUBlock(
            n_freqs, n_freqs, kernel_size=(1,1),
            stride=(1,1), padding=(0,0), dilation=(1,1),
            use_batchnorm=use_batchnorm, use_deconv=0, use_act=use_act,
            use_convbnrelu=use_convbnrelu, memory_efficient=0,
        )

        self.memory_efficient = memory_efficient

    def forward(self, x):
        def f_(x_):
            x_ = self.convbnrelu1(x_)
            x_ = x_.transpose(1,3)
            x_ = self.convbnrelu2(x_)
            x_ = x_.transpose(1,3)
            return x_
        return cp.checkpoint(f_, x) if self.memory_efficient else f_(x)


class DenseBlockNoOutCatFM(nn.Module):
    def __init__(
        self,
        in_channels,
        growth_rate,
        kernel_size,
        n_freqs,
        n_layers=5,
        use_dilation=0,
        use_batchnorm=2,
        use_act=1,
        use_convbnrelu=1,
        memory_efficient=0,
        last_growth_rate=-1,
        use_depthwise=False,
    ):
        super(DenseBlockNoOutCatFM, self).__init__()
        self._layers = []
        sum_channels = in_channels
        middle_layer = n_layers // 2
        assert n_layers > 1 and n_layers % 2 == 1
        if last_growth_rate == -1:
            last_growth_rate = growth_rate
        for ll in range(n_layers):
            if ll == middle_layer:
                convbnrelu = FreqMapping(
                    sum_channels, growth_rate, n_freqs,
                    use_batchnorm=use_batchnorm, use_act=use_act,
                    use_convbnrelu=use_convbnrelu, memory_efficient=0,
                    use_depthwise=use_depthwise,
                )
            else:
                dilation = 2 ** (ll - (1 if ll > middle_layer else 0)) if use_dilation else 1
                padding = (get_padding_size(kernel_size[0], dilation), kernel_size[1] // 2)
                convbnrelu = Conv2dBNReLUBlock(
                    sum_channels, last_growth_rate if ll == n_layers-1 else growth_rate, kernel_size,
                    stride=(1,1), padding=padding, dilation=(dilation,1),
                    use_batchnorm=use_batchnorm, use_deconv=0, use_act=use_act,
                    use_convbnrelu=use_convbnrelu, memory_efficient=0,
                    use_depthwise=use_depthwise,
                )
            self.add_module("convbnrelu%d"%ll, convbnrelu)
            self._layers.append(convbnrelu)
            sum_channels += growth_rate
        self.n_layers = n_layers
        self.memory_efficient = memory_efficient

    def __getitem__(self, key):
        return getattr(self, key)

    def forward(self,*x):
        x = list(x)
        for ll, nn_module in enumerate(self._layers):
            def g_(nn_module_=nn_module):
                def f_(*x_):
                    x_ = torch.cat(x_, dim=1)
                    return nn_module_(x_)
                return f_
            h = cp.checkpoint(g_(), *x) if self.memory_efficient else g_()(*x)
            x.append(h)
        return h


class TCNDepthWise(nn.Module):
    def __init__(
        self,
        input_channels,
        out_channels,
        kernel_size=3,
        use_batchnorm=1,
        use_act=1,
        dilation=1,
    ):
        super(TCNDepthWise, self).__init__()

        padding = get_padding_size(kernel_size, dilation)

        self.norm0 = get_norm(use_batchnorm,input_channels,ndims=3)
        self.conv00 = nn.Conv1d(input_channels, input_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, groups=input_channels)
        self.conv01 = nn.Conv1d(input_channels, out_channels, kernel_size=1)

        self.norm1 = get_norm(use_batchnorm,out_channels,ndims=3)
        self.conv10 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, groups=out_channels)
        self.conv11 = nn.Conv1d(out_channels, out_channels, kernel_size=1)

        self.actfn = get_actfn(use_act)

        self.init_weights()

    def init_weights(self):
        self.conv00.weight.data.normal_(0, 0.01)
        self.conv01.weight.data.normal_(0, 0.01)
        self.conv10.weight.data.normal_(0, 0.01)
        self.conv11.weight.data.normal_(0, 0.01)
        self.conv00.bias.data[:] = 0.0
        self.conv01.bias.data[:] = 0.0
        self.conv10.bias.data[:] = 0.0
        self.conv11.bias.data[:] = 0.0

    def forward(self, x, hidden_dropout_rate=0.0, dilation_dropout_rate=0.0):

        h = self.actfn(self.norm0(x))

        if hidden_dropout_rate:
            h = F.dropout(h, p=hidden_dropout_rate, training=True, inplace=False)

        h = self.conv00(h)
        h = self.conv01(h)

        h = self.actfn(self.norm1(h))

        if hidden_dropout_rate:
            h = F.dropout(h, p=hidden_dropout_rate, training=True, inplace=False)

        h = self.conv10(h)
        h = self.conv11(h)

        return h+x


class CSeqUNetDenseSeg(AbsSeparator):
    def __init__(self,
        input_dim,
        num_spk,
        nlayer,
        n_units,
        target_dim,
        use_seqmodel,
        masking_or_mapping,
        pitactivation,
        rmax,
        approx_method,
        loss_function,
        sigparams,
        use_inConv,
        n_imics,
        n_omics,
        use_batchnorm,
        use_convbnrelu,
        use_act,
        memory_efficient,
        use_sad=0,
        sad_weight=1.0,
        n_outputs=2,
    ):
        super(CSeqUNetDenseSeg, self).__init__()
        #
        # The default parameters are
        # num_spk = 2
        # input_dim = 257 * 7
        # nlayer = 2
        # n_units = 384
        # target_dim = 257
        # use_seqmodel = 1
        # masking_or_mapping = 1
        # pitactivation = 'linear'
        # approx_method = 'MSA-RIx2'
        # loss_function = 'l1loss'
        # use_inConv = 2147
        # n_imics = 7
        # n_omics = 1
        # use_batchnorm = 3
        # use_convbnrelu = 2
        # use_act = 1
        # memory_efficient = 1
        #
        approx_method = approx_method.split('-')
        self.approx_method = approx_method

        if loss_function not in ['l1loss','l2loss']: raise
        self.loss_function = loss_function

        assert n_imics >= n_omics

        if target_dim not in [257*n_omics]: raise

        self.sigparams=sigparams

        if input_dim % (target_dim//n_omics) != 0: raise
        in_channels = input_dim // (target_dim//n_omics)
        assert in_channels >= 1

        self.use_inConv = use_inConv
        t_ksize = 3
        in_channels *= 2
        if use_inConv in [2147]:
            #
            #              257,   1                                   2/4
            #(257-3)/1+1 = 255,  24 ---5*24--> 24  +  24  ---5*24---> 48
            #(255-3)/2+1 = 127,  32 ---5*32--> 32  +  32  ---5*32---> 64
            #(127-3)/2+1 =  63,  32 ---5*32--> 32  +  32  ---5*32---> 64
            #(63-3)/2+1  =  31,  32 ---5*32--> 32  +  32  ---5*32---> 64
            #(31-3)/2+1  =  15,  32 ---5*32--> 32  +  32  ---5*32---> 64
            #(15-3)/2+1  =   7,  64                +  64
            #(7-3)/2+1   =   3, 128                +  128
            #(3-3)/1+1   =   1, 384                +  384
            #
            encoder_dim = 384
            ks, padding = (t_ksize,3), (t_ksize//2,0)
            self.conv0 = Conv2dBNReLUBlock(in_channels,24,ks,stride=(1,1),padding=padding,use_batchnorm=0,use_act=0,use_convbnrelu=1)
            kwargs = {'use_batchnorm':use_batchnorm, 'use_act':use_act, 'use_convbnrelu':use_convbnrelu}
            self.eden0 = DenseBlockNoOutCatFM(24,24,ks,255,n_layers=5,memory_efficient=memory_efficient,**kwargs)
            self.conv1 = Conv2dBNReLUBlock(24,32,ks,stride=(1,2),padding=padding,**kwargs)
            self.eden1 = DenseBlockNoOutCatFM(32,32,ks,127,n_layers=5,memory_efficient=memory_efficient,**kwargs)
            self.conv2 = Conv2dBNReLUBlock(32,32,ks,stride=(1,2),padding=padding,**kwargs)
            self.eden2 = DenseBlockNoOutCatFM(32,32,ks,63,n_layers=5,memory_efficient=memory_efficient,**kwargs)
            self.conv3 = Conv2dBNReLUBlock(32,32,ks,stride=(1,2),padding=padding,**kwargs)
            self.eden3 = DenseBlockNoOutCatFM(32,32,ks,31,n_layers=5,memory_efficient=memory_efficient,**kwargs)
            self.conv4 = Conv2dBNReLUBlock(32,32,ks,stride=(1,2),padding=padding,**kwargs)
            self.eden4 = DenseBlockNoOutCatFM(32,32,ks,15,n_layers=5,memory_efficient=memory_efficient,**kwargs)
            self.conv5 = Conv2dBNReLUBlock(32,64,ks,stride=(1,2),padding=padding,**kwargs)
            self.conv6 = Conv2dBNReLUBlock(64,128,ks,stride=(1,2),padding=padding,**kwargs)
            self.conv7 = Conv2dBNReLUBlock(128,encoder_dim,ks,stride=(1,1),padding=padding,**kwargs)
        elif use_inConv in [215]:
            #
            #              257,   1
            #(257-3)/1+1 = 255,  32 ---5*32--> 32
            #(255-3)/2+1 = 127,  32 ---5*32--> 32
            #(127-3)/2+1 =  63,  32 ---5*32--> 32
            #(63-3)/2+1  =  31,  32 ---5*32--> 32
            #(31-3)/2+1  =  15,  32 ---5*32--> 32
            #(15-3)/2+1  =   7,  64
            #(7-3)/2+1   =   3, 128
            #(3-3)/1+1   =   1, 384
            #                     3
            #
            assert use_sad
            encoder_dim = 384
            ks, padding = (t_ksize,3), (t_ksize//2,0)
            self.conv0 = Conv2dBNReLUBlock(in_channels,32,ks,stride=(1,1),padding=padding,use_batchnorm=0,use_act=0,use_convbnrelu=1)
            kwargs = {'use_batchnorm':use_batchnorm, 'use_act':use_act, 'use_convbnrelu':use_convbnrelu}
            self.eden0 = DenseBlockNoOutCatFM(32,32,ks,255,n_layers=5,memory_efficient=memory_efficient,**kwargs)
            self.conv1 = Conv2dBNReLUBlock(32,32,ks,stride=(1,2),padding=padding,**kwargs)
            self.eden1 = DenseBlockNoOutCatFM(32,32,ks,127,n_layers=5,memory_efficient=memory_efficient,**kwargs)
            self.conv2 = Conv2dBNReLUBlock(32,32,ks,stride=(1,2),padding=padding,**kwargs)
            self.eden2 = DenseBlockNoOutCatFM(32,32,ks,63,n_layers=5,memory_efficient=memory_efficient,**kwargs)
            self.conv3 = Conv2dBNReLUBlock(32,32,ks,stride=(1,2),padding=padding,**kwargs)
            self.eden3 = DenseBlockNoOutCatFM(32,32,ks,31,n_layers=5,memory_efficient=memory_efficient,**kwargs)
            self.conv4 = Conv2dBNReLUBlock(32,32,ks,stride=(1,2),padding=padding,**kwargs)
            self.eden4 = DenseBlockNoOutCatFM(32,32,ks,15,n_layers=5,memory_efficient=memory_efficient,**kwargs)
            self.conv5 = Conv2dBNReLUBlock(32,64,ks,stride=(1,2),padding=padding,**kwargs)
            self.conv6 = Conv2dBNReLUBlock(64,128,ks,stride=(1,2),padding=padding,**kwargs)
            self.conv7 = Conv2dBNReLUBlock(128,encoder_dim,ks,stride=(1,1),padding=padding,**kwargs)
        else:
            raise
        input_dim = encoder_dim
        assert n_units == encoder_dim

        if use_seqmodel == 1:
            tcn_classname = TCNDepthWise
            for ii in range(nlayer):
                self.add_module('tcn-conv%d-0'%ii, tcn_classname(input_dim,input_dim,t_ksize,use_batchnorm=use_batchnorm,use_act=use_act,dilation=1))
                self.add_module('tcn-conv%d-1'%ii, tcn_classname(input_dim,input_dim,t_ksize,use_batchnorm=use_batchnorm,use_act=use_act,dilation=2))
                self.add_module('tcn-conv%d-2'%ii, tcn_classname(input_dim,input_dim,t_ksize,use_batchnorm=use_batchnorm,use_act=use_act,dilation=4))
                self.add_module('tcn-conv%d-3'%ii, tcn_classname(input_dim,input_dim,t_ksize,use_batchnorm=use_batchnorm,use_act=use_act,dilation=8))
                self.add_module('tcn-conv%d-4'%ii, tcn_classname(input_dim,input_dim,t_ksize,use_batchnorm=use_batchnorm,use_act=use_act,dilation=16))
                self.add_module('tcn-conv%d-5'%ii, tcn_classname(input_dim,input_dim,t_ksize,use_batchnorm=use_batchnorm,use_act=use_act,dilation=32))
        else:
            raise

        if use_sad > 0:
            if use_sad == 1:
                self.n_sad_classes = 3
            elif use_sad == 2:
                self.n_sad_classes = 2
            else:
                raise
            self.add_module('to_sad', nn.Conv2d(input_dim,self.n_sad_classes,kernel_size=1))
        self.use_sad = use_sad
        self.sad_weight = sad_weight

        initial_bias = 0.0
        input_dim += encoder_dim

        if use_inConv in [2147]:
            #
            #              257,   1                                   2/4
            #(257-3)/1+1 = 255,  24 ---5*24--> 24  +  24  ---5*24---> 48
            #(255-3)/2+1 = 127,  32 ---5*32--> 32  +  32  ---5*32---> 64
            #(127-3)/2+1 =  63,  32 ---5*32--> 32  +  32  ---5*32---> 64
            #(63-3)/2+1  =  31,  32 ---5*32--> 32  +  32  ---5*32---> 64
            #(31-3)/2+1  =  15,  32 ---5*32--> 32  +  32  ---5*32---> 64
            #(15-3)/2+1  =   7,  64                +  64
            #(7-3)/2+1   =   3, 128                +  128
            #(3-3)/1+1   =   1, 384                +  384
            #
            #non-causal
            ks, padding = (t_ksize,3), (t_ksize//2,0)
            kwargs = {'use_batchnorm':use_batchnorm, 'use_act':use_act, 'use_convbnrelu':use_convbnrelu}
            self.deconv0 = Conv2dBNReLUBlock(input_dim,128,ks,stride=(1,1),padding=padding,use_deconv=1,**kwargs)
            self.deconv1 = Conv2dBNReLUBlock(128+128,64,ks,stride=(1,2),padding=padding,use_deconv=1,**kwargs)
            self.deconv2 = Conv2dBNReLUBlock(64+64,32,ks,stride=(1,2),padding=padding,use_deconv=1,**kwargs)
            self.dden2 = DenseBlockNoOutCatFM(32+32,32,ks,15,n_layers=5,memory_efficient=memory_efficient,last_growth_rate=64,**kwargs)
            self.deconv3 = Conv2dBNReLUBlock(64,32,ks,stride=(1,2),padding=padding,use_deconv=1,**kwargs)
            self.dden3 = DenseBlockNoOutCatFM(32+32,32,ks,31,n_layers=5,memory_efficient=memory_efficient,last_growth_rate=64,**kwargs)
            self.deconv4 = Conv2dBNReLUBlock(64,32,ks,stride=(1,2),padding=padding,use_deconv=1,**kwargs)
            self.dden4 = DenseBlockNoOutCatFM(32+32,32,ks,63,n_layers=5,memory_efficient=memory_efficient,last_growth_rate=64,**kwargs)
            self.deconv5 = Conv2dBNReLUBlock(64,32,ks,stride=(1,2),padding=padding,use_deconv=1,**kwargs)
            self.dden5 = DenseBlockNoOutCatFM(32+32,32,ks,127,n_layers=5,memory_efficient=memory_efficient,last_growth_rate=64,**kwargs)
            self.deconv6 = Conv2dBNReLUBlock(64,24,ks,stride=(1,2),padding=padding,use_deconv=1,**kwargs)
            self.dden6 = DenseBlockNoOutCatFM(24+24,24,ks,255,n_layers=5,memory_efficient=memory_efficient,last_growth_rate=48,**kwargs)
            self.deconv7 = Conv2dBNReLUBlock(48,num_spk*n_outputs*n_omics,ks,stride=(1,1),padding=padding,use_deconv=1,use_batchnorm=0,use_act=0,use_convbnrelu=1)
            self.deconv7.conv.bias.data[:] = initial_bias
        elif use_inConv in [215]:
            pass
        else:
            raise

        self.pitactivation = pitactivation
        self.rmax = rmax

        self.n_clusters = num_spk
        self._num_spk = num_spk
        self.target_dim = target_dim
        self.n_imics = n_imics
        self.n_omics = n_omics
        self.nlayer = nlayer
        self.n_units = n_units
        self.use_seqmodel = use_seqmodel
        self.masking_or_mapping = masking_or_mapping
        self.n_outputs = n_outputs

    def __getitem__(self, key):
        return getattr(self, key)

    def forward_encoder_seq(
        self,
        input: torch.Tensor,
        ilens: torch.Tensor,
        input_dropout_rate=0.0,
        hidden_dropout_rate=0.0,
    ):
        """Forward.

        Args:
            input (torch.Tensor): Encoded feature [B, T, C, N], C for channel
            ilens (torch.Tensor): input lengths [Batch]
        Returns:
            feature (torch.Tensor): mixed feature after encoder [Batch, flens, channel]
        """
        batchsize, max_len, n_channel, feat_dim = input.shape[:4]

        if input_dropout_rate:
            input = F.dropout(
                input, p=input_dropout_rate, training=True, inplace=False,
            )

        batch = input.transpose(1, 2)  # (B, C, T, N)

        all_conv_batch = []
        for cc in range(100):
            conv_link_name = 'conv%d'%cc
            if hasattr(self, conv_link_name):
                batch = self[conv_link_name](batch)
                eden_link_name = 'eden%d'%cc
                if hasattr(self, eden_link_name):
                    batch = self[eden_link_name](batch)
                all_conv_batch.append(batch)
            else:
                break

        if self.use_seqmodel in [1]:
            batch = batch.view([batchsize, self.n_units, max_len])
            for ii in range(self.nlayer):
                for cc in range(20):
                    conv_link_name = 'tcn-conv%d-%d'%(ii,cc)
                    if hasattr(self, conv_link_name):
                        batch = self[conv_link_name](batch,
                                hidden_dropout_rate=hidden_dropout_rate)
                    else:
                        break
            batch = batch.unsqueeze(dim=-1)
        else:
            raise

        if self.use_sad > 0:
            sad_activations = self['to_sad'](batch)
            sad_activations = sad_activations.transpose(1,2).squeeze(dim=-1)
            sad_activations = [sad_activations[bb,:utt_len] for bb,utt_len in enumerate(ilenvec)]
            self.sad_activations = torch.cat(sad_activations, dim=0)

        return batch, all_conv_batch

    def forward(
        self,
        input: torch.Tensor,
        ilens: torch.Tensor,
        input_dropout_rate=0.0,
        hidden_dropout_rate=0.0,
    ):
        """Forward.

        Args:
            input (torch.ComplexTensor): Encoded feature [Batch, T, N] or [Batch, T, C, N], C is for channel
            ilens (torch.Tensor): input lengths [Batch]
        Returns:
            feature (torch.Tensor): mixed feature after encoder [Batch, flens, channel]
        """

        if len(input.shape) == 3:
            input = input.unsqueeze(-2)

        if is_complex(input):
            input = torch.cat([input.real, input.imag], dim=2)  # (B, T, 2C, N)

        batch, all_conv_batch = self.forward_encoder_seq(
            input,
            ilens,
            input_dropout_rate=input_dropout_rate,
            hidden_dropout_rate=hidden_dropout_rate,
        )

        for cc in range(100):
            deconv_link_name = 'deconv%d'%cc
            if hasattr(self, deconv_link_name):
                if cc-1 >= 0 and hasattr(self, 'dden%d'%(cc-1)):
                    batch = self[deconv_link_name](batch)
                else:
                    batch = self[deconv_link_name](batch,all_conv_batch[-1-cc])
                dden_link_name = 'dden%d'%cc
                if hasattr(self, dden_link_name):
                    batch = self[dden_link_name](batch,all_conv_batch[-1-cc-1])
            else:
                break

        batch = batch.transpose(1,2)
        real, imag = batch.chunk(2, dim=2)
        batch = torch.complex(real.squeeze(2), imag.squeeze(2))

        return [batch], ilens, None

    @property
    def num_spk(self):
        return self._num_spk

