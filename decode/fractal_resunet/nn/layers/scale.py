import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv2Dnormed import Conv2DNormed


class DownSample(nn.Module):
    """Downsampling layer"""

    def __init__(self, nfilters, factor=2, norm_type="BatchNorm", norm_groups=None, **kwargs):
        super().__init__()

        self.factor = factor
        self.nfilters = nfilters * self.factor

        self.kernel_size = (3, 3)
        self.strides = (factor, factor)
        self.pad = (1, 1)

        self.convdn = Conv2DNormed(
            self.nfilters,
            kernel_size=self.kernel_size,
            stride=self.strides,
            padding=self.pad,
            norm_type=norm_type,
            norm_groups=norm_groups,
            in_channels=nfilters,
            out_channels=self.nfilters,
        )

    def forward(self, xl):
        x = self.convdn(xl)
        return x


class UpSample(nn.Module):
    """Upsampling layer"""

    def __init__(self, nfilters, factor=2, norm_type="BatchNorm", norm_groups=None, **kwargs):
        super().__init__()

        self.factor = factor
        self.nfilters = nfilters // self.factor

        self.convup_normed = Conv2DNormed(
            self.nfilters,
            kernel_size=(1, 1),
            norm_type=norm_type,
            norm_groups=norm_groups,
            in_channels=nfilters,
            out_channels=self.nfilters,
        )

    def forward(self, xl):
        x = F.interpolate(xl, scale_factor=self.factor, mode="nearest")
        x = self.convup_normed(x)
        return x
