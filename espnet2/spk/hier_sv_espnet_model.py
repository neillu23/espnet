# Copyright 2023 Jee-weon Jung
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from typing import Dict, List, Optional, Tuple, Union
import logging
import torch
from typeguard import check_argument_types

from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.spk.loss.aamsoftmax import AAMSoftmax
from espnet2.spk.loss.abs_loss import AbsLoss
from espnet2.spk.pooling.abs_pooling import AbsPooling
from espnet2.spk.projector.abs_projector import AbsProjector
from espnet2.spk.espnet_model import ESPnetSpeakerModel
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel

from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.postencoder.abs_postencoder import AbsPostEncoder


from packaging.version import parse as V
if V(torch.__version__) >= V("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield

def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True, export=False):
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)

class ESPnetHierSVModel(ESPnetSpeakerModel):
    """
    Speaker embedding extraction model.
    Core model for diverse speaker-related tasks (e.g., verification, open-set
    identification, diarization)

    The model architecture comprises mainly 'encoder', 'pooling', and
    'projector'.
    In common speaker recognition field, the combination of three would be
    usually named as 'speaker_encoder' (or speaker embedding extractor).
    We splitted it into three for flexibility in future extensions:
      - 'encoder'   : extract frame-level speaker embeddings.
      - 'pooling'   : aggregate into single utterance-level embedding.
      - 'projector' : (optional) additional processing (e.g., one fully-
                      connected layer) to derive speaker embedding.

    Possibly, in the future, 'pooling' and/or 'projector' can be integrated as
    a 'decoder', depending on the extension for joint usage of different tasks
    (e.g., ASR, SE, target speaker extraction).
    """

    def __init__(
        self,
        frontend: Optional[AbsFrontend],
        specaug: Optional[AbsSpecAug],
        normalize: Optional[AbsNormalize],
        preencoder: Optional[AbsPreEncoder],
        encoder: Optional[AbsEncoder],
        pooling: Optional[AbsPooling],
        projector: Optional[AbsProjector],
        loss: Optional[AbsLoss],
        preencoder_lid: Union[AbsPreEncoder,torch.nn.modules.container.ModuleList],
        encoder_lid: AbsEncoder,
        postencoder_lid: Union[AbsPostEncoder,AbsPooling],
        projector_lid: Optional[AbsProjector],
        lid_tokens: Union[Tuple[str, ...], List[str]] = None,
        langs_num: int = 0,
        embed_condition: bool = False,
        embed_condition_size: int = 0,
        lid_condition_feature: str = "soft",
        lid_condition_activate: Optional[str] = None,
        preencoder_lid_nums: int = 1,
        sep_layers: List[int] = [],
        droprate: float = 0.3,
        separate_forward: bool = True,
    ):
        assert check_argument_types()

        super().__init__(
            frontend=frontend,
            specaug=specaug,
            normalize=normalize,
            preencoder=preencoder,
            encoder=encoder,
            pooling=pooling,
            projector=projector,
            loss=loss,
            lid_tokens=lid_tokens,
            langs_num=langs_num,
            embed_condition=embed_condition,
            embed_condition_size=embed_condition_size,
            lid_condition_feature=lid_condition_feature,

        )
        self.preencoder_lid = preencoder_lid
        self.encoder_lid = encoder_lid
        self.postencoder_lid = postencoder_lid
        self.projector_lid = projector_lid
        # self.loss_lid = loss_lid
        self.lid_condition_activate = lid_condition_activate
        if len(sep_layers) == 0:
            self.sep_layers = [self.frontend.upstream.num_layers - 1]
        else:
            self.sep_layers = sep_layers
        self.preencoder_lid_nums = preencoder_lid_nums
        
        self.separate_forward = separate_forward
        # self.droprate = droprate

        assert(separate_forward == True), "separate_forward must be True"

        if self.embed_condition and self.lid_condition_feature == "soft":
            # 256 is the size of the lid/speaker embedding
            sep_num = len(self.sep_layers)
            if self.sep_layers[-1] == self.frontend.upstream.num_layers - 1:
                sep_num = sep_num - 1
            self.lang_embeddings = torch.nn.ModuleList([torch.nn.Linear(256, embed_condition_size) for i in range(sep_num)])

            if self.lid_condition_activate == "bndrop":
                self.lns = torch.nn.ModuleList([LayerNorm(embed_condition_size, export=False) for i in range(sep_num)])
                self.activation_fns = torch.nn.ModuleList([torch.nn.PReLU() for i in range(sep_num)])
                self.dropouts = torch.nn.ModuleList([torch.nn.Dropout(p=droprate) for i in range(sep_num)])


    def forward(
        self,
        speech: torch.Tensor,
        spk_labels: torch.Tensor,
        task_tokens: torch.Tensor = None,
        extract_embd: bool = False,
        langs: torch.Tensor = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """
        Feed-forward through encoder layers and aggregate into utterance-level
        feature.

        Args:
            speech: (Batch, samples)
            speech_lengths: (Batch,)
            extract_embd: a flag which doesn't go through the classification
                head when set True
            spk_labels: (Batch, )
            one-hot speaker labels used in the train phase
            task_tokens: (Batch, )
            task tokens used in case of token-based trainings
        """
        if spk_labels is not None:
            assert speech.shape[0] == spk_labels.shape[0], (
                speech.shape,
                spk_labels.shape,
            )
        if task_tokens is not None:
            assert speech.shape[0] == task_tokens.shape[0], (
                speech.shape,
                task_tokens.shape,
            )
        batch_size = speech.shape[0]



        def my_hook(module, input, output):
            # This will be executed upon the forward pass of the hooked layer.
            # 'module' is the layer the hook is attached to,
            # 'input' is the input to the layer,
            # 'output' is the output of the layer.
            # Here you can do things with the output, for instance:
            x, (attn, layer_result) = output  # Storing it in the instance for later use
            self.intermediate_outputs = x

        condition_features = None
        feats_layers = []
        feats_lengths_layers = []
    
        lid_embd_list = []
        for index in range(len(self.sep_layers)):
            # condition_features = None for testing purpose
            # condition_features = None 
            if index == 0:
                start_layer = 0
            else:
                start_layer = self.sep_layers[index-1]

            end_layer = self.sep_layers[index]

            hook_handle = self.frontend.upstream.upstream.model.encoder.layers[end_layer-1].register_forward_hook(my_hook)

            with autocast(False):
                # 1. Extract LID feats
                # lid_speech = speech
                # lid_speech_lengths = speech_lengths
                feats_lid, feats_lid_lengths, feats_layers, feats_lengths_layers = self.extract_feats(speech, None, condition_features, start_layer=start_layer, end_layer=end_layer, featurizer_index=index, feats_layers=feats_layers, feats_lengths_layers=feats_lengths_layers)
                   
                # 2. Data augmentation
                if self.specaug is not None and self.training:
                    feats_lid, feats_lid_lengths = self.specaug(feats_lid, feats_lid_lengths)

                # 3. Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
                if self.normalize is not None:
                    feats_lid, feats_lid_lengths = self.normalize(feats_lid, feats_lid_lengths)

                
                hook_handle.remove()


                # 4. Forward encoder for LID
                # Pre-encoder, e.g. used for raw input data
                if self.preencoder_lid_nums > 1:
                    feats_lid, feats_lid_lengths = self.preencoder_lid[index](feats_lid, feats_lid_lengths)
                else:
                    if self.preencoder_lid is not None:
                        feats_lid, feats_lid_lengths = self.preencoder_lid(feats_lid, feats_lid_lengths)

                encoder_lid_out, encoder_lid_out_lens, _ = self.encoder_lid(feats_lid, feats_lid_lengths)

                # Post-encoder, e.g. NLU
                if self.postencoder_lid is not None:
                    encoder_lid_out = self.postencoder_lid(
                        encoder_lid_out
                    )
                # import pdb; pdb.set_trace()
                lid_embd = self.project_lid_embd(encoder_lid_out)
                lid_embd_list.append(lid_embd)

            # does not need to generate condition features from the last layer
            if self.sep_layers[index] == self.frontend.upstream.num_layers - 1:
                break
                
            with autocast(False):
                # 5. generate lid condition features
                if self.embed_condition:
                    if self.lid_condition_feature == "hard":
                        # Compute cosine similarity
                        cosine = F.linear(F.normalize(lid_embd), F.normalize(self.loss.weight))
                        
                        # Get the predicted speaker index
                        cosine_similarity = torch.max(cosine, dim=1)
                        langs_token = torch.argmax(cosine, dim=1)

                    # condition_features = self.lang_embedding(langs)
                        condition_features = self.lang_embedding(langs_token).unsqueeze(1)
                    elif self.lid_condition_feature == "GT":
                        condition_features = self.lang_embedding(langs)
                    elif self.lid_condition_feature == "soft":
                        if self.lid_condition_activate == "LeakyReLU":
                            # import pdb; pdb.set_trace()
                            condition_features = self.lang_embedding(lid_embd).unsqueeze(1)
                            condition_features = torch.nn.LeakyReLU()(condition_features)
                        elif self.lid_condition_activate == "bndrop":
                            # import pdb; pdb.set_trace()
                            condition_features = self.lang_embeddings[index](lid_embd)
                            condition_features = self.lns[index](condition_features)
                            condition_features = condition_features.unsqueeze(1)
                            condition_features = self.activation_fns[index](condition_features.to(torch.float32))
                            condition_features = self.dropouts[index](condition_features)





        with autocast(False):
            if self.sep_layers[-1] < self.frontend.upstream.num_layers - 1:
                feats_layers_new, feats_lengths_new = self.frontend.upstream(speech, speech_lengths, condition_features, split_forward=True, last_layer_result=self.intermediate_outputs, start_layer=self.sep_layers[-1], end_layer=24)
                feats_layers = feats_layers[:-1]
                feats_layers.extend(feats_layers_new)
                feats_lengths_layers.extend(feats_lengths_new)


            # ori_feats_layers, ori_feats_lengths_layers = self.frontend.upstream(speech, speech_lengths, condition_features)
            # for i in range(25):
            #     logging.info("feats_layers {} is the same as ori_feats_layers: {}".format(i, torch.all(torch.eq(ori_feats_layers[i],feats_layers[i]))))
            # for i in range(25):
            #     # logging.info("feats_layers {} is the same as ori_feats_layers: {}".format(i, torch.all(torch.eq(ori_feats_layers[i],feats_layers[i]))))
            #     logging.info("layer_index: {}".format(i))
            #     logging.info("feats_layers shape: {}".format(feats_layers[i].shape))
            #     logging.info("feats_layers features: {}".format(feats_layers[i]))
            # exit()


            feats, feats_lengths = self.frontend.featurizer_asr(feats_layers, feats_lengths_layers)


        # Pre-encoder, e.g. used for raw input data
        if self.preencoder is not None:
            feats, feats_lengths = self.preencoder(feats, feats_lengths)

        frame_level_feats = self.encode_frame(feats)

        # 2. aggregation into utterance-level
        utt_level_feat = self.pooling(frame_level_feats, task_tokens)

        # 3. (optionally) go through further projection(s)
        spk_embd = self.project_spk_embd(utt_level_feat)

        if extract_embd:
            return spk_embd

        # 4. calculate loss
        loss = self.loss(spk_embd, spk_labels.squeeze())

        stats = dict(loss=loss.detach())

        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

        
    def extract_feats(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor, condition_features: torch.Tensor = None, start_layer: int = 0, end_layer: int = 24, featurizer_index: int = 0, feats_layers: Optional[List] = None, feats_lengths_layers: Optional[List] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        batch_size = speech.shape[0]
        speech_lengths = (
            speech_lengths
            if speech_lengths is not None
            else torch.ones(batch_size).int() * speech.shape[1]
        ).to(speech.device)
        assert speech_lengths.dim() == 1, speech_lengths.shape


        # for data-parallel
        speech = speech[:, : speech_lengths.max()]

        if self.frontend is not None:
            # Frontend
            #  e.g. STFT and Feature extract
            #       data_loader may send time-domain signal in this case
            # speech (Batch, NSamples) -> feats: (Batch, NFrames, Dim)
            if start_layer == 0:
                feats, feats_lengths, feats_layers, feats_lengths_layers = self.frontend(speech, speech_lengths, condition_features=condition_features, split_forward=self.separate_forward, end_layer=end_layer)
                return feats, feats_lengths, feats_layers, feats_lengths_layers 
            else:
                feats_layers_new, feats_lengths_new = self.frontend.upstream(speech, speech_lengths, condition_features=condition_features, split_forward=True, last_layer_result=self.intermediate_outputs, start_layer=start_layer, end_layer=end_layer)
                
                feats_layers = feats_layers[:-1]
                feats_layers.extend(feats_layers_new)
                feats_lengths_layers.extend(feats_lengths_new)
                feats, feats_lengths = self.frontend.featurizers[featurizer_index](feats_layers, feats_lengths_layers)

                return feats, feats_lengths, feats_layers, feats_lengths_layers
        
    def encode_frame(self, feats: torch.Tensor) -> torch.Tensor:
        frame_level_feats = self.encoder(feats)

        return frame_level_feats

    def aggregate(self, frame_level_feats: torch.Tensor) -> torch.Tensor:
        utt_level_feat = self.aggregator(frame_level_feats)

        return utt_level_feat

    def project_spk_embd(self, utt_level_feat: torch.Tensor) -> torch.Tensor:
        if self.projector is not None:
            spk_embd = self.projector(utt_level_feat)
        else:
            spk_embd = utt_level_feat

        return spk_embd

    def project_lid_embd(self, utt_level_feat: torch.Tensor) -> torch.Tensor:
        if self.projector_lid is not None:
            lid_embd = self.projector_lid(utt_level_feat)
        else:
            lid_embd = utt_level_feat

        return lid_embd

    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        spk_labels: torch.Tensor = None,
        langs: torch.Tensor = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:

        condition_features = None
        if self.embed_condition:
            condition_features = self.lang_embedding(langs)
            
        feats, feats_lengths, feats_layers, feats_lengths_layers = self.extract_feats(speech, speech_lengths, condition_features)
        return {"feats": feats}
