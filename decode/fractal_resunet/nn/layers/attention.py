import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv2Dnormed import Conv2DNormed
from .ftnmt import FTanimoto


class RelFTAttention2D(nn.Module):
    """Relative FracTAL Attention 2D"""

    def __init__(
        self, nkeys, kernel_size=3, padding=1, nheads=1, norm="BatchNorm", norm_groups=None, ftdepth=5, **kwargs
    ):
        super().__init__()

        self.query = Conv2DNormed(
            channels=nkeys, kernel_size=kernel_size, padding=padding, norm_type=norm, norm_groups=norm_groups
        )
        self.key = Conv2DNormed(
            channels=nkeys, kernel_size=kernel_size, padding=padding, norm_type=norm, norm_groups=norm_groups
        )
        self.value = Conv2DNormed(
            channels=nkeys, kernel_size=kernel_size, padding=padding, norm_type=norm, norm_groups=norm_groups
        )

        self.metric_channel = FTanimoto(depth=ftdepth, axis=[2, 3])
        self.metric_space = FTanimoto(depth=ftdepth, axis=1)

        if norm == "BatchNorm":
            self.norm = nn.BatchNorm2d(nkeys)
        elif norm == "InstanceNorm":
            self.norm = nn.InstanceNorm2d(nkeys)
        elif norm == "LayerNorm":
            self.norm = nn.LayerNorm([nkeys, 1, 1])
        elif norm == "GroupNorm" and norm_groups is not None:
            self.norm = nn.GroupNorm(num_groups=norm_groups, num_channels=nkeys)
        else:
            raise NotImplementedError(f"Normalization {norm} not implemented")

    def forward(self, input1, input2, input3):

        q = torch.sigmoid(self.query(input1))
        k = torch.sigmoid(self.key(input2))
        v = torch.sigmoid(self.value(input3))

        att_spat = self.metric_space(q, k)
        v_spat = att_spat * v

        att_chan = self.metric_channel(q, k)
        v_chan = att_chan * v

        v_cspat = 0.5 * (v_chan + v_spat)
        v_cspat = self.norm(v_cspat)

        return v_cspat


class FTAttention2D(nn.Module):
    """FracTAL Attention 2D"""

    def __init__(
        self, nkeys, kernel_size=3, padding=1, nheads=1, norm="BatchNorm", norm_groups=None, ftdepth=5, **kwargs
    ):
        super().__init__()

        self.att = RelFTAttention2D(
            nkeys=nkeys,
            kernel_size=kernel_size,
            padding=padding,
            nheads=nheads,
            norm=norm,
            norm_groups=norm_groups,
            ftdepth=ftdepth,
            **kwargs,
        )

    def forward(self, input):
        return self.att(input, input, input)
