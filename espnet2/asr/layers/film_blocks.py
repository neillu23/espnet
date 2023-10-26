import torch
import torch.nn as nn

class FiLM(nn.Module):
    def __init__(self, input_size, condition_size, film_type="linear"):
        # condition_size: the size of the language id vector
        # input_size: the size of the RNN input to the FiLM layer
        super(FiLM, self).__init__()
        if film_type == "tanh":
            self.linear_scale = nn.Sequential(
                nn.Linear(condition_size, input_size),
                nn.Tanh()
            )
            self.linear_shift = nn.Sequential(
                nn.Linear(condition_size, input_size),
                nn.Tanh()
            )
        elif film_type == "linear":
            self.linear_scale = nn.Linear(condition_size, input_size)
            self.linear_shift = nn.Linear(condition_size, input_size)

    def forward(self, x, lang_condition):
        # import pdb; pdb.set_trace()
        if x.ndim == 3:
            gamma = self.linear_scale(lang_condition).expand_as(x)
            beta = self.linear_shift(lang_condition).expand_as(x)
            x = x * gamma + beta
        elif x.ndim == 4:
            gamma = self.linear_scale(lang_condition).unsqueeze(1).expand_as(x)
            beta = self.linear_shift(lang_condition).unsqueeze(1).expand_as(x)
            x = x * gamma + beta
        return x


