import torch
import torch.nn as nn


class SigmoidCrisp(nn.Module):
    """Crisp Sigmoid activation with learnable gamma parameter"""

    def __init__(self, smooth=1.0e-2, **kwargs):
        super().__init__()

        self.smooth = smooth
        self.gamma = nn.Parameter(torch.ones(1))

    def forward(self, input):
        out = self.smooth + torch.sigmoid(self.gamma)
        out = torch.reciprocal(out)
        out = input * out
        out = torch.sigmoid(out)
        return out
