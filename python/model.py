import torch
from torch import nn


class HmmModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # state * ctx * emit
        self.emit_probs = nn.Parameter(torch.rand(size=(4, 16, 12)))
        self.ctx_state_probs = nn.Parameter(torch.rand(size=(16, 4)))

    def forward(self, read, template):
        dp_matrix = torch.tensor()
        pass
