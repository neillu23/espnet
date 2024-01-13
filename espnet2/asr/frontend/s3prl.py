import copy
import logging
from typing import Optional, Tuple, Union

import humanfriendly
import torch
from typeguard import check_argument_types

from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet.nets.pytorch_backend.frontends.frontend import Frontend


class S3prlFrontend(AbsFrontend):
    """Speech Pretrained Representation frontend structure for ASR."""

    def __init__(
        self,
        fs: Union[int, str] = 16000,
        frontend_conf: Optional[dict] = get_default_kwargs(Frontend),
        download_dir: str = None,
        multilayer_feature: bool = False,
        layer: int = -1,
        layer_selections: Optional[list] = None,
        featurizer_num: int = 1,
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
        if layer_selections is not None:
            assert(layer == -1), "layer should be -1 when layer_selections is not None"
            assert(multilayer_feature), "multilayer feature should be True when layer_selections is not None"
        elif layer != -1:
            layer_selections = [layer]
            assert (
                not multilayer_feature
            ), "multilayer feature will be deactivated, when specific layer used"

        

        self.multilayer_feature = multilayer_feature
        self.layer = layer
        self.upstream = upstream

        self.featurizer_num = featurizer_num

        if featurizer_num <= 2:
            self.featurizer = Featurizer(upstream, layer_selections=layer_selections)
            self.hop_length = self.featurizer.downsample_rate

        if featurizer_num == 2:
            self.featurizer2 = Featurizer(upstream, layer_selections=None)

        elif featurizer_num > 2:
            self.featurizers = torch.nn.ModuleList([Featurizer(upstream, layer_selections=layer_selections[i]) for i in range(featurizer_num - 1)])

            self.featurizer_asr = Featurizer(upstream, layer_selections=None)
            self.hop_length = self.featurizer_asr.downsample_rate

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
        if self.featurizer_num > 2:
            return self.featurizer_asr.output_size
        return self.featurizer.output_size

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor, condition_features: torch.Tensor=None, langs: torch.Tensor=None, langs_lens: torch.Tensor=None, 
        split_forward: bool = False, end_layer: int = 24
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # import pdb; pdb.set_trace()
        if self.self_condition:
            feats, feats_lens, self_condition_loss = self.upstream(input, input_lengths, condition_features, langs=langs, langs_lens=langs_lens, split_forward=split_forward, start_layer=0, end_layer=end_layer)
        else:
            feats, feats_lens = self.upstream(input, input_lengths, condition_features, langs=langs, langs_lens=langs_lens, split_forward=split_forward, start_layer=0, end_layer=end_layer)
        if self.layer != -1:
            layer = self.layer
            feats, feats_lens = feats[layer], feats_lens[layer]
            return feats, feats_lens

        if self.featurizer_num <= 2:
            featurizer = self.featurizer
        else:
            featurizer = self.featurizers[0]
            
        if self.multilayer_feature:
            feats_fused, feats_lens_fused = featurizer(feats, feats_lens)
        else:
            feats_fused, feats_lens_fused = featurizer(feats[-1:], feats_lens[-1:])

        if self.tile_factor != 1:
            feats_fused = self._tile_representations(feats_fused)
        
        
        
        if self.featurizer_num > 1:
            return feats_fused, feats_lens_fused, feats, feats_lens

        elif self.self_condition:
            return feats_fused, feats_lens_fused, self_condition_loss
        else:
            return feats_fused, feats_lens_fused

    def reload_pretrained_parameters(self):
        self.upstream.load_state_dict(self.pretrained_params)
        logging.info("Pretrained S3PRL frontend model parameters reloaded!")
