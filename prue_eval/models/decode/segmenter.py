"""
Segmenter adapter for DECODE (FracTAL-ResUNet).

This wraps DECODE inference logic and converts outputs to SemanticOutput,
so the rest of the pipeline stays backend-agnostic.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch
import yaml

from ...intermediate_formats import SemanticOutput
from ...converters import convert_decode_output
from ..registry import Segmenter, register_model

# Import DECODE model from the repo's decode/ package
from decode.fractal_resunet.models.semanticsegmentation.FracTAL_ResUNet import (
    FracTAL_ResUNet_cmtsk as decode_model,
)


class DecodeSegmenter:
    """
    DECODE Segmenter adapter that wraps DECODE inference and returns SemanticOutput.

    Args:
        model_weights: Path to DECODE checkpoint
        config: Optional config dict or path to YAML config
        device: Device to run inference on
        model_name: Identifier for logging/debugging
    """

    def __init__(
        self,
        model_weights: str,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
        model_name: str = "decode",
    ):
        self.model_weights = model_weights
        self.model_name = model_name

        # Determine device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load config
        self.config = self._load_config(config)

        # Load model
        self._load_model()

    def _load_config(self, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Load config from dict or YAML file."""
        if config is None:
            # Use default config
            return {
                "data": {
                    "in_channels": 8,
                    "n_classes": 2,
                    "temporal_option": "stacked",
                    "crop_size": [256, 256],
                },
                "model": {
                    "nfilters_init": 32,
                    "depth": 6,
                    "ftdepth": 5,
                    "psp_depth": 4,
                    "norm_type": "GroupNorm",
                    "norm_groups": 4,
                    "nheads_start": 4,
                },
            }
        elif isinstance(config, dict):
            return config
        elif isinstance(config, str):
            # Assume it's a path to YAML
            with open(config, "r") as f:
                return yaml.safe_load(f)
        else:
            raise TypeError(f"Config must be dict, str (path), or None, got {type(config)}")

    def _load_model(self):
        """Load DECODE model from checkpoint."""
        cfg = self.config

        self.model = decode_model(
            nfilters_init=cfg["model"]["nfilters_init"],
            NClasses=cfg["data"]["n_classes"],
            depth=cfg["model"]["depth"],
            ftdepth=cfg["model"]["ftdepth"],
            psp_depth=cfg["model"]["psp_depth"],
            norm_type=cfg["model"]["norm_type"],
            norm_groups=cfg["model"]["norm_groups"],
            nheads_start=cfg["model"]["nheads_start"],
            in_channels=cfg["data"]["in_channels"],
        ).to(self.device)

        # Load weights
        checkpoint = torch.load(self.model_weights, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.eval()

    def predict(self, batch: Any) -> Iterable[SemanticOutput]:
        """
        Run DECODE inference on a batch of images and return SemanticOutput.

        Args:
            batch: Can be:
                - Single image tensor (C, H, W) or (B, C, H, W)
                - List of image tensors
                - Dict with 'image' key

        Yields:
            SemanticOutput for each image in the batch
        """
        # Normalize batch to tensor
        images = self._normalize_batch(batch)

        with torch.inference_mode():
            # Run inference
            preds = self.model(images)

            # DECODE returns tuple: (seg_logits, boundary_logits, distance)
            seg_logits, boundary_logits, distance = preds

            # Process each sample in batch
            batch_size = images.shape[0]
            for i in range(batch_size):
                # Create tuple for converter
                sample_output = (
                    seg_logits[i : i + 1],  # Keep batch dim for converter
                    boundary_logits[i : i + 1],
                    distance[i : i + 1],
                )

                # Convert to SemanticOutput using existing converter
                semantic_output = convert_decode_output(sample_output, image_id=i)

                yield semantic_output

    def _normalize_batch(self, batch: Any) -> torch.Tensor:
        """Normalize various batch formats to a tensor."""
        if isinstance(batch, dict):
            if "image" in batch:
                batch = batch["image"]
            else:
                raise ValueError("Batch dict must contain 'image' key")

        if isinstance(batch, list):
            batch = torch.stack([self._to_tensor(img) for img in batch])
        elif isinstance(batch, (torch.Tensor, np.ndarray)):
            batch = self._to_tensor(batch)
            if batch.ndim == 3:
                batch = batch.unsqueeze(0)  # Add batch dimension
        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}")

        return batch.to(self.device)

    def _to_tensor(self, x: Any) -> torch.Tensor:
        """Convert numpy array or tensor to torch tensor."""
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x)
        elif isinstance(x, torch.Tensor):
            return x
        else:
            raise TypeError(f"Cannot convert {type(x)} to tensor")


@register_model("decode", family="decode")
def create_decode_segmenter(**kwargs) -> Segmenter:
    """
    Registry factory for DECODE models.

    Required kwargs:
        model_weights: Path to DECODE checkpoint

    Optional kwargs:
        config: Dict or path to YAML config file (default: uses defaults)
        device: Device string (default: auto-detect)
    """
    return DecodeSegmenter(**kwargs)
