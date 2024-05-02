import copy
import logging
from typing import Optional, Tuple, Union

import humanfriendly
import torch
from typeguard import check_argument_types

from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet.nets.pytorch_backend.frontends.frontend import Frontend


class S3prlSHALLiFrontend(AbsFrontend):
    """Speech Pretrained Representation frontend structure for ASR."""

    def __init__(
        self,
        fs: Union[int, str] = 16000,
        frontend_conf: Optional[dict] = get_default_kwargs(Frontend),
        download_dir: str = None,
        asr_layer_selections: Optional[list] = None,
        lid_layer_selections: Optional[list] = None,
        spk_layer_selections: Optional[list] = None,
        # feature_lid: Optional[str] = None, # can be "lid" or "hier_lid"
        # feature_spk: Optional[str] = None, # can be "spk" or "hier_spk"
    ):
        try:
            import s3prl
            from s3prl.nn import Featurizer, S3PRLUpstream
        except Exception as e:
            print("Error: S3PRL is not properly installed.")
            print("Please install S3PRL: cd ${MAIN_ROOT}/tools && make s3prl.done")
            raise e

        assert check_argument_types()
        super().__init__()

        if isinstance(fs, str):
            fs = humanfriendly.parse_size(fs)
        if fs != 16000:
            logging.warning(
                "All the upstream models in S3PRL now only support 16 kHz audio."
            )

        if download_dir is not None:
            s3prl.util.download.set_dir(download_dir)

        assert frontend_conf.get("upstream", None) in S3PRLUpstream.available_names()
        upstream = S3PRLUpstream(
            frontend_conf.get("upstream"),
            path_or_url=frontend_conf.get("path_or_url", None),
            normalize=frontend_conf.get("normalize", False),
            embed_condition=frontend_conf.get("embed_condition", False),
            self_condition=frontend_conf.get("self_condition", False),
            extra_conf=frontend_conf.get("extra_conf", None),
        )
        if getattr(upstream.upstream, "model", None):
            if getattr(upstream.upstream.model, "feature_grad_mult", None) is not None:
                upstream.upstream.model.feature_grad_mult = 1.0
        upstream.eval()
        
        self.upstream = upstream

        self.asr_layer_selections = asr_layer_selections
        self.lid_layer_selections = lid_layer_selections
        self.spk_layer_selections = spk_layer_selections

        if self.asr_layer_selections is not None:
            self.asr_featurizers = torch.nn.ModuleList([Featurizer(upstream, layer_selections=[i for i in range(layer + 1)]) for layer in asr_layer_selections])
            self.hop_length = self.asr_featurizers[0].downsample_rate
            self.out_size = self.asr_featurizers[0].output_size
        
        if self.lid_layer_selections is not None:
            self.lid_featurizers = torch.nn.ModuleList([Featurizer(upstream, layer_selections=[i for i in range(layer + 1)]) for layer in lid_layer_selections])
            self.hop_length = self.lid_featurizers[0].downsample_rate
            self.out_size = self.lid_featurizers[0].output_size
        
        if self.spk_layer_selections is not None:
            self.spk_featurizers = torch.nn.ModuleList([Featurizer(upstream, layer_selections=[i for i in range(layer + 1)]) for layer in spk_layer_selections])
            self.hop_length = self.spk_featurizers[0].downsample_rate
            self.out_size = self.spk_featurizers[0].output_size
        
        self.pretrained_params = copy.deepcopy(self.upstream.state_dict())
        self.frontend_type = "s3prl"
        self.self_condition = frontend_conf.get("self_condition", False)
        
        self.tile_factor = frontend_conf.get("tile_factor", 1)

    def _tile_representations(self, feature):
        """Tile up the representations by `tile_factor`.

        Input - sequence of representations
                shape: (batch_size, seq_len, feature_dim)

        Output - sequence of tiled representations
                 shape: (batch_size, seq_len * factor, feature_dim)
        """
        assert (
            len(feature.shape) == 3
        ), "Input argument `feature` has invalid shape: {}".format(feature.shape)
        tiled_feature = feature.repeat(1, 1, self.tile_factor)
        tiled_feature = tiled_feature.reshape(
            feature.size(0), feature.size(1) * self.tile_factor, feature.size(2)
        )
        return tiled_feature

    def output_size(self) -> int:
        return self.out_size

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor, 
        condition_features: torch.Tensor=None, 
        langs: torch.Tensor=None, 
        langs_lens: torch.Tensor=None,
        layer: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # import pdb; pdb.set_trace()
        if layer == -1:
            layer = self.upstream.num_layers - 1

        if self.self_condition:
            feats, feats_lens, self_condition_loss = self.upstream(
                input, 
                input_lengths, 
                condition_features, 
                langs=langs, 
                langs_lens=langs_lens, 
                start_layer=0, 
                end_layer=layer) 
            return feats, feats_lens, self_condition_loss
        else:
            feats, feats_lens = self.upstream(input, 
                                            input_lengths, 
                                            condition_features, 
                                            split_forward=True, 
                                            start_layer=0, 
                                            end_layer=layer)        
            return feats, feats_lens

    def _match_length(self, xs, target_max_len: int):
        xs_max_len = xs.size(1)

        if xs_max_len > target_max_len:
            assert xs_max_len // target_max_len == 1, f"{xs_max_len}, {target_max_len}"
            xs = xs[:, :target_max_len, :]

        elif xs_max_len < target_max_len:
            assert target_max_len // xs_max_len == 1, f"{target_max_len}, {xs_max_len}"
            xs = torch.cat(
                (xs, xs[:, -1:, :].repeat(1, target_max_len - xs_max_len, 1)), dim=1
            )

        return xs


    def encode_layers(
        self, input: torch.Tensor, input_lengths: torch.Tensor,
        condition_features: torch.Tensor=None, 
        langs: torch.Tensor=None, 
        langs_lens: torch.Tensor=None,
        last_layer_result: Optional[torch.Tensor]=None,
        layers: tuple = (0, 24),
        feats_layers: list = [],
        feats_lengths_layers: list = [],
        padding_mask: Optional[torch.Tensor]=None,
    ):
        if padding_mask is None:
            wav_padding_mask = ~torch.lt(
                torch.arange(max(input_lengths)).unsqueeze(0).to(input.device),
                input_lengths.unsqueeze(1),
            )

            extra = wav_padding_mask.size(1) % last_layer_result.size(0)
            if extra > 0:
                wav_padding_mask = wav_padding_mask[:, :-extra]
            padding_mask = wav_padding_mask.view(wav_padding_mask.size(0), last_layer_result.size(0), -1)
            padding_mask = padding_mask.all(-1)

        x = last_layer_result

        if padding_mask is not None:
            # logging.info("padding_mask shape: {}".format(padding_mask.shape))
            # logging.info("padding_mask true num: {}".format(padding_mask.sum(dim=1)))
            pad_length = min(padding_mask.sum(dim=1)).item()
        else:
            pad_length = 0
        # import pdb; pdb.set_trace()
        if pad_length > 0:
            layer_results = [self._match_length(last_layer_result.permute(1, 0, 2)[:,:-pad_length], feats_layers[-1].size(1))]
        else:
            layer_results = [self._match_length(last_layer_result.permute(1, 0, 2), feats_layers[-1].size(1))]
        
        # import pdb; pdb.set_trace()
        # logging.info(f"Encoding layers from {layers[0]} to {layers[1]}")
        for layer_index in range(layers[0], layers[1]):
            # logging.info("Encoding layer {},  x input size: {}, x input: {}".format(layer_index, x.size(), x))
            # logging.info((self.upstream.upstream.model.encoder.layers[layer_index].self_attn.k_proj.weight))
            # logging.info((self.upstream.upstream.model.encoder.layers[layer_index+1].self_attn.k_proj.weight))

            # logging.info("padding_mask: {}".format(padding_mask))
            
            x, (z, lr, padding_mask) = self.upstream.upstream.model.encoder.layers[layer_index](
                x, condition_features, self_attn_padding_mask=padding_mask, need_weights=False
            )

            # logging.info("Encoding layer {},  x output size: {}, x output: {}".format(layer_index, x.size(), x))
            # import pdb; pdb.set_trace()
            # if layer_index == self.upstream.num_layers - 2 and self.upstream.upstream.model.encoder.layer_norm_first:
            #     x = self.upstream.upstream.model.encoder.layer_norm(x)
            if layer_index == layers[1] - 1 and self.upstream.upstream.model.encoder.layer_norm_first:
                intermediate_output = x.clone()
                x = self.upstream.upstream.model.encoder.layer_norm(x)

            if self.upstream.normalize:
                logging.info("Layer normalization needed")

            # layer_results.append(self._match_length(x.permute(1, 0, 2)[:,:-pad_length],feats_layers[-1].size(1)))
                
            # TODO:
            # max_wav_len = int(max(wavs_len))
            # all_hs = []
            # all_lens = []
            # for h, stride in zip(hidden_states, self.downsample_rates):
            #     expected_max_h_len = len(range(0, max_wav_len, stride))
            #     h = self._match_length(h, expected_max_h_len)
            #     assert h.size(1) == expected_max_h_len

            #     h_len = torch.div(original_wavs_len - 1, stride, rounding_mode="floor") + 1
            #     h = h[:, : max(h_len), :]
            #     if self.upstream..normalize:
            #         h = F.layer_norm(h, h.shape[-1:])

            #     all_hs.append(h)
            #     all_lens.append(h_len)


            if pad_length > 0:
                layer_results.append(self._match_length(x.permute(1, 0, 2)[:,:-pad_length],feats_layers[-1].size(1)))
            else:
                layer_results.append(self._match_length(x.permute(1, 0, 2),feats_layers[-1].size(1)))
        # logging.info("layer results: {}".format(layer_results))
        
        # TODO: use the same length as previous feats_layers
        # layer_results = [undo_pad(*u) for u in layer_results]

        # import pdb; pdb.set_trace()
        feats_layers = feats_layers[:-1]
        feats_layers.extend(layer_results)
        # use same lengths as previous feats_lengths_layers, copy the last one
        # feats_lengths_layer
        # feats_lengths_new = [input_lengths for _ in range(len(layer_results))]
        # # feats_lengths_layers = feats_lengths_layers[:-1]
        feats_lengths_layers.extend([feats_lengths_layers[-1] for _ in range(len(layer_results))])
        
        return intermediate_output, feats_layers, feats_lengths_layers


    def encode_layers_ori(
        self, input: torch.Tensor, input_lengths: torch.Tensor,
        condition_features: torch.Tensor=None, 
        langs: torch.Tensor=None, 
        langs_lens: torch.Tensor=None,
        last_layer_result: Optional[torch.Tensor]=None,
        layers: tuple = (0, 24),
        feats_layers: list = [],
        feats_lengths_layers: list = [],
        padding_mask: Optional[torch.Tensor]=None,
    ):
        # TODO: directly call the wav2vec2 instead of the upstream
        feats_layers_new, feats_lengths_new = self.upstream(
                                                input, 
                                                input_lengths, 
                                                condition_features=condition_features, 
                                                split_forward=True, 
                                                last_layer_result=last_layer_result, 
                                                start_layer=layers[0], 
                                                end_layer=layers[1])

        feats_layers = feats_layers[:-1]
        feats_layers.extend(feats_layers_new)
        feats_lengths_layers.extend(feats_lengths_new)
        # feats, feats_lengths = self.frontend.featurizers[featurizer_index](feats_layers, feats_lengths_layers)

        return feats_layers, feats_lengths_layers



    def reload_pretrained_parameters(self):
        self.upstream.load_state_dict(self.pretrained_params)
        logging.info("Pretrained S3PRL frontend model parameters reloaded!")
