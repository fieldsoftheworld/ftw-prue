import torch
import torch.nn as nn
from .FracTAL_ResUNet_features import FracTAL_ResUNet_features
from ..heads.head_cmtsk import Head_CMTSK_BC


class FracTAL_ResUNet_cmtsk(nn.Module):
    """
    FracTAL_ResUNet + conditioned multitasking.
    PyTorch implementation of the original MXNet model.
    """

    def __init__(
        self,
        nfilters_init,
        depth,
        NClasses,
        widths=[1],
        psp_depth=4,
        verbose=True,
        norm_type="BatchNorm",
        norm_groups=None,
        nheads_start=8,
        upFuse=False,
        ftdepth=5,
        in_channels=None,
        **kwargs,
    ):
        super().__init__()

        self.features = FracTAL_ResUNet_features(
            nfilters_init=nfilters_init,
            depth=depth,
            widths=widths,
            psp_depth=psp_depth,
            verbose=verbose,
            norm_type=norm_type,
            norm_groups=norm_groups,
            nheads_start=nheads_start,
            upFuse=upFuse,
            ftdepth=ftdepth,
            in_channels=in_channels,
            **kwargs,
        )

        self.head = Head_CMTSK_BC(nfilters_init, NClasses, norm_type=norm_type, norm_groups=norm_groups, **kwargs)

    def forward(self, input):
        out1, out2 = self.features(input)
        return self.head(out1, out2)
