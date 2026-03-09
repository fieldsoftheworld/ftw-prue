"""Tests for loss functions."""

import pytest
import torch

from ftw_tools.torchgeo.losses import PixelWeightedCE, logCoshDice, logCoshDiceCE


@pytest.fixture
def batch():
    B, C, H, W = 2, 3, 64, 64
    logits = torch.randn(B, C, H, W)
    masks = torch.randint(0, C, (B, H, W))
    return logits, masks


class TestPixelWeightedCE:
    def test_forward(self, batch):
        logits, masks = batch
        loss_fn = PixelWeightedCE(kernel_size=5, sigma=3.0, target_class=2)
        loss = loss_fn(logits, masks)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_with_ignore_index(self, batch):
        logits, masks = batch
        masks[0, :5, :5] = 3
        loss_fn = PixelWeightedCE(kernel_size=5, sigma=3.0, target_class=2, ignore_index=3)
        loss = loss_fn(logits, masks)
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_with_class_weights(self, batch):
        logits, masks = batch
        weights = torch.tensor([1.0, 2.0, 3.0])
        loss_fn = PixelWeightedCE(kernel_size=5, sigma=3.0, class_weights=weights)
        loss = loss_fn(logits, masks)
        assert loss.shape == ()


class TestLogCoshDice:
    def test_forward(self, batch):
        logits, masks = batch
        loss_fn = logCoshDice(mode="multiclass")
        loss = loss_fn(logits, masks)
        assert loss.shape == ()
        assert loss.item() >= 0


class TestLogCoshDiceCE:
    def test_forward(self, batch):
        logits, masks = batch
        loss_fn = logCoshDiceCE(mode="multiclass")
        loss = loss_fn(logits, masks)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_with_ignore_index(self, batch):
        logits, masks = batch
        masks[0, :10, :10] = 3
        loss_fn = logCoshDiceCE(mode="multiclass", ignore_index=3)
        loss = loss_fn(logits, masks)
        assert torch.isfinite(loss)
