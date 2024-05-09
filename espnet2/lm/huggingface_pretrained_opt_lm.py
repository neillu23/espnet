import copy
import logging
from typing import Any, List, Tuple

import torch
import torch.nn as nn
from typeguard import typechecked

from espnet2.lm.abs_model import AbsLM
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
from peft import prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

class HuggingfaceOPTModel(AbsLM):
    @typechecked
    def __init__(
        self,
        vocab_size: int,
        opt_name: str,
        max_position_embeddings: int = 2048,
        qlora: bool = False,
        lora: bool = False,
        local_files_only: bool = False,
        load_in_8bit: bool = False,
        bnb_4bit_quant_type="nf4",
        lora_config=None,
    ):
        super().__init__()
        try:
            from transformers import OPTModel, AutoModelForCausalLM
        except Exception as e:
            print("Error: transformers is not properly installed.")
            print("Please install transformers")
            raise e

        # opt_model_name_pattern = re.compile(r"facebook/opt-\d+m")
        # assert opt_model_name_pattern.match(opt_name) is not None


        if qlora and load_in_8bit:

            nf4_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
            pretrained_opt_model = OPTModel.from_pretrained(opt_name, torch_dtype=torch.float16, quantization_config=nf4_config, local_files_only=local_files_only)
            pretrained_opt_model.gradient_checkpointing_enable()
            pretrained_opt_model = prepare_model_for_kbit_training(pretrained_opt_model)
        elif qlora:


            nf4_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            pretrained_opt_model = OPTModel.from_pretrained(opt_name, torch_dtype=torch.float16, quantization_config=nf4_config, local_files_only=local_files_only)
            pretrained_opt_model.gradient_checkpointing_enable()
            pretrained_opt_model = prepare_model_for_kbit_training(pretrained_opt_model)


        else:
            pretrained_opt_model = OPTModel.from_pretrained(opt_name, local_files_only=local_files_only, load_in_8bit=load_in_8bit)
        

        #pretrained_opt_model = AutoModelForCausalLM.from_pretrained(opt_name)
        pretrained_opt_model_dict = pretrained_opt_model.state_dict()
        pretrained_opt_model_dict.pop("decoder.embed_tokens.weight")
        self.pretrained_params = copy.deepcopy(pretrained_opt_model_dict)


        


        
        config = pretrained_opt_model.config
        config.max_position_embeddings = max_position_embeddings
        config.vocab_size = vocab_size
        config.bos_token_id = vocab_size - 1
        config.eos_token_id = vocab_size - 1
        config.pad_token_id = 0
        self.adjust_decoder_positions(max_position_embeddings + 2)

        self.decoder = OPTModel(config)

        self.lm_head = nn.Linear(
            config.word_embed_proj_dim, config.vocab_size, bias=False
        )

        if qlora or lora:
            default_lora_config = {
                "r": 8,
                "lora_alpha": 32,
                "target_modules": ["q_proj", "v_proj", "k_proj"],
                "lora_dropout": 0.05,
                "bias": "none",
            }
            if lora_config is None:
                lora_config = {}
            lora_config = LoraConfig(
                **{**default_lora_config, **lora_config}
            )

            self.decoder = get_peft_model(self.decoder, lora_config)



    def _target_mask(self, ys_in_pad):
        ys_mask = ys_in_pad != 0
        m = subsequent_mask(ys_mask.size(-1), device=ys_mask.device).unsqueeze(0)
        return ys_mask.unsqueeze(-2) & m

    def forward(self, input: torch.Tensor, hidden: None) -> Tuple[torch.Tensor, None]:
        """Compute LM loss value from buffer sequences.

        Args:
            input (torch.Tensor): Input ids. (batch, len)
            hidden (torch.Tensor): Target ids. (batch, len)

        """
        pad_mask = input != 0
        y = self.decoder(
            input,
            attention_mask=pad_mask,
            return_dict=True,
        )
        y = y.last_hidden_state

        logits = self.lm_head(y)

        return logits, None

    def score(
        self, y: torch.Tensor, state: Any, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Any]:
        """Score new token.

        Args:
            y (torch.Tensor): 1D torch.int64 prefix tokens.
            state: Scorer state for prefix tokens
            x (torch.Tensor): encoder feature that generates ys.

        Returns:
            tuple[torch.Tensor, Any]: Tuple of
                torch.float32 scores for next token (vocab_size)
                and next state for ys

        """
        if state is None:
            _use_cache = True
        else:
            _use_cache = False

        y = y.unsqueeze(0)

        output = self.decoder(
            y,
            past_key_values=state,
            use_cache=_use_cache,
            output_hidden_states=True,
            return_dict=True,
        )

        h = output.last_hidden_state[:, -1]
        h = self.lm_head(h)
        cache = output.past_key_values
        logp = h.log_softmax(dim=-1).squeeze(0)
        return logp, cache

    def batch_score(
        self, ys: torch.Tensor, states: List[Any], xs: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Any]]:
        """Score new token batch.

        Args:
            ys (torch.Tensor): torch.int64 prefix tokens (n_batch, ylen).
            states (List[Any]): Scorer states for prefix tokens.
            xs (torch.Tensor):
                The encoder feature that generates ys (n_batch, xlen, n_feat).

        Returns:
            tuple[torch.Tensor, List[Any]]: Tuple of
                batchfied scores for next token with shape of `(n_batch, vocab_size)`
                and next state list for ys.

        """
        # merge states
        n_batch = len(ys)
        n_layers = len(self.decoder.decoder.layers)
        if states[0] is None:
            _use_cache = True
        else:
            _use_cache = False

        # batch decoding
        output = self.decoder(
            ys,
            use_cache=_use_cache,
            output_hidden_states=True,
            return_dict=True,
        )
        h = output.last_hidden_state
        h = self.lm_head(h[:, -1])

        logp = h.log_softmax(dim=-1)

        state_list = [[[] for i in range(n_layers)] for b in range(n_batch)]
        return logp, state_list


    def adjust_decoder_positions(self, max_position_embeddings):
        old_param = self.pretrained_params["decoder.embed_positions.weight"]
        if old_param.size(0) != max_position_embeddings:
            new_param = torch.empty((max_position_embeddings, old_param.size(1)), dtype=old_param.dtype, device=old_param.device)
            torch.nn.init.normal_(new_param, mean=0, std=0.02)
            fill_size = min(old_param.size(0), max_position_embeddings)
            new_param[:fill_size, :] = old_param[:fill_size, :]
            self.pretrained_params["decoder.embed_positions.weight"] = new_param
            logging.info(f"'decoder.embed_positions.weight' adjusted to new max position embeddings size: {max_position_embeddings}")

    def reload_pretrained_parameters(self):
        self.decoder.load_state_dict(self.pretrained_params, strict=False)
        logging.info("Pretrained OPT model parameters reloaded!")