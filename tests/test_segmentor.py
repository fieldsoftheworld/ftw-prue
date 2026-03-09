"""Tests for segmentation head models."""

import pytest
import torch

from ftw_tools.models.segmentor import (
    ConvASPPDecoderHead,
    ConvLightDecoderHead,
    MLPCombiner,
    SegmentationHead,
)

B = 2
NUM_CLASSES = 3

MODEL_CONFIGS = {
    "croma": {"input": 120, "patch": 8, "dim": 768, "embed_hw": 15},
    "galileo": {"input": 256, "patch": 4, "dim": 768, "embed_hw": 64},
    "decur": {"input": 224, "patch": 16, "dim": 384, "embed_hw": 14},
    "dofa": {"input": 224, "patch": 16, "dim": 1024, "embed_hw": 14},
    "prithvi": {"input": 224, "patch": 16, "dim": 1024, "embed_hw": 14},
    "satlas": {"input": 256, "patch": 16, "dim": 768, "embed_hw": 16},
    "softcon": {"input": 224, "patch": 14, "dim": 384, "embed_hw": 16},
    "clay": {"input": 256, "patch": 8, "dim": 1024, "embed_hw": 32},
    "dinov3": {"input": 256, "patch": 16, "dim": 1024, "embed_hw": 16},
    "terrafm": {"input": 256, "patch": 16, "dim": 768, "embed_hw": 16},
    "terramind": {"input": 256, "patch": 16, "dim": 768, "embed_hw": 16},
}


class TestMLPCombiner:
    def test_forward(self):
        D, L = 768, 16
        combiner = MLPCombiner(D=D)
        x = torch.randn(B, 2 * L, D)
        out = combiner(x)
        assert out.shape == (B, L, D)


class TestConvLightDecoderHead:
    def test_forward(self):
        D, L = 768, 256
        decoder = ConvLightDecoderHead(dim=D, out_size=256, num_classes=NUM_CLASSES)
        x = torch.randn(B, L, D)
        out = decoder(x)
        assert out.shape == (B, NUM_CLASSES, 256, 256)


class TestConvASPPDecoderHead:
    def test_forward(self):
        D, L, patch = 768, 256, 16
        decoder = ConvASPPDecoderHead(dim=D, patch_size=patch, num_classes=NUM_CLASSES)
        x = torch.randn(B, L, D)
        out = decoder(x)
        assert out.shape == (B, NUM_CLASSES, 256, 256)


class TestSegmentationHead:
    @pytest.mark.parametrize("decoder_type", ["conv_w_aspp", "conv_light"])
    @pytest.mark.parametrize(
        "model_name",
        ["clay", "terrafm", "satlas", "croma", "galileo", "decur", "dofa"],
    )
    def test_forward(self, model_name, decoder_type):
        cfg = MODEL_CONFIGS[model_name]
        tokens_per_view = cfg["embed_hw"] ** 2
        feats = torch.randn(B, 2, tokens_per_view, cfg["dim"])

        model = SegmentationHead(
            fusion_type="mlp",
            decoder_type=decoder_type,
            dim=cfg["dim"],
            patch_size=cfg["patch"],
            num_classes=NUM_CLASSES,
            original_input_size=cfg["input"],
        )
        out = model({"feat": feats})
        assert out.shape == (B, NUM_CLASSES, 256, 256)

    def test_forward_3d_input(self):
        """Test with pre-concatenated (B, 2L, D) input."""
        cfg = MODEL_CONFIGS["clay"]
        L = cfg["embed_hw"] ** 2
        feats = torch.randn(B, 2 * L, cfg["dim"])

        model = SegmentationHead(
            fusion_type="mlp",
            decoder_type="conv_w_aspp",
            dim=cfg["dim"],
            patch_size=cfg["patch"],
            num_classes=NUM_CLASSES,
            original_input_size=cfg["input"],
        )
        out = model(feats)
        assert out.shape == (B, NUM_CLASSES, 256, 256)
