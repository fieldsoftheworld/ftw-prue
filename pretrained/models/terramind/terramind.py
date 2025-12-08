"""
Terramind v1 encoder for 4-channel Sentinel-2 (RGB + NIR) inputs.
"""

import torch
import torch.nn as nn
from terratorch.registry import BACKBONE_REGISTRY


class SegmentEncoder(nn.Module):
    """
    Encoder wrapper for semantic segmentation (or other dense prediction) tasks
    using a Terramind backbone.

    This module builds a `terramind_v1_base` backbone from `terratorch` with the
    `S2L2A` modality and uses the BLUE, GREEN, RED, and NIR_BROAD bands as
    input channels.

    Attributes:
        model (nn.Module): Terramind backbone created via `BACKBONE_REGISTRY`,
            expecting inputs of shape (B, 4, H, W) corresponding to
            [BLUE, GREEN, RED, NIR_BROAD].
    """

    def __init__(self, model_type="terramind_v1_base", modalities=["S2L2A"], bands={"S2L2A":["BLUE", "GREEN", "RED", "NIR_BROAD"]}, pretrained=True):  # noqa: PLR0913, ARG002
        """
        Initialize the SegmentEncoder.

        Args:
            ckpt_path (str, optional): Reserved argument for compatibility with
                other encoders that may load checkpoints explicitly. Terramind
                weights are loaded via `pretrained=True`, so this argument is
                currently unused.
        """
        super().__init__()

        self.model = BACKBONE_REGISTRY.build(
            name=model_type,
            pretrained=pretrained,
            modalities=modalities,
            bands=bands,
        )

    def forward(self, img_tensor: torch.Tensor):
        """
        Forward pass of the Terramind encoder.

        Args:
            img_tensor (torch.Tensor):
                Input image tensor of shape (B, C, H, W). For Terramind v1 with
                the configuration above, C is expected to be 4 corresponding to
                [BLUE, GREEN, RED, NIR_BROAD].

        Returns:
            torch.Tensor:
                A tensor of patch / feature embeddings produced by the
                Terramind backbone, typically of shape (B, num_patches,
                embed_dim), depending on the underlying model implementation.
        """
        B, C, H, W = img_tensor.shape  # noqa: F841 (kept for clarity)

        if C == 4:
            # Forward the 4-channel Sentinel-2 tensor through Terramind to obtain
            # patch-level embeddings or feature representations.
            patches = self.model(img_tensor)  # (B, num_patches, embed_dim)
            last_layer_patches = patches[-1]  # Take the last layer's output
            return last_layer_patches

        raise ValueError(
            f"Expected 4 channels [BLUE, GREEN, RED, NIR_BROAD], got C={C}."
        )