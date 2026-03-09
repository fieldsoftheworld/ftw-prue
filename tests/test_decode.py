"""Tests for DECODE (FracTAL ResUNet) model."""

import pytest
import torch

from decode.fractal_resunet.models.semanticsegmentation.FracTAL_ResUNet import FracTAL_ResUNet_cmtsk


class TestFracTALResUNetCMTSK:
    @pytest.mark.parametrize("in_channels", [4, 8])
    @pytest.mark.parametrize("n_classes", [2, 3])
    def test_forward(self, in_channels, n_classes):
        model = FracTAL_ResUNet_cmtsk(
            nfilters_init=64,
            depth=3,
            NClasses=n_classes,
            in_channels=in_channels,
        )
        x = torch.randn(2, in_channels, 256, 256)
        seg, bound, dist = model(x)
        assert seg.shape == (2, n_classes, 256, 256)
        assert bound.shape == (2, n_classes, 256, 256)
        assert dist.shape == (2, 1, 256, 256)

    def test_output_ranges(self):
        model = FracTAL_ResUNet_cmtsk(
            nfilters_init=64,
            depth=3,
            NClasses=2,
            in_channels=8,
        )
        x = torch.randn(1, 8, 256, 256)
        seg, bound, dist = model(x)
        # seg uses softmax, bound uses crisp sigmoid, dist uses sigmoid
        assert seg.min() >= 0 and seg.max() <= 1
        assert dist.min() >= 0 and dist.max() <= 1
