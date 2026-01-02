import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPModule(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(MLPModel, self).__init__()

        layers = []
        in_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


