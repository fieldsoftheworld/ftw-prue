from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch
from detectron2.config import CfgNode
from detectron2.engine import DefaultPredictor


class SatellitePredictor(DefaultPredictor):
    """Custom predictor for multi-band satellite imagery (4 or 8 channels).

    This mirrors the behavior used in the standalone Mask2Former branch:
    it accepts an HWC multi-band array and forwards it through the model
    without imposing RGB-only assumptions.
    """

    def __init__(self, cfg: CfgNode):
        super().__init__(cfg)

    def __call__(self, original_image: np.ndarray) -> Dict[str, Any]:
        """
        Args:
            original_image: Multi-band satellite image in (H, W, C) format.

        Returns:
            Model predictions dict (may contain 'panoptic_seg', 'sem_seg', etc.).
        """
        with torch.no_grad():
            height, width = original_image.shape[:2]
            # Convert HWC -> CHW float tensor
            image = torch.as_tensor(original_image.astype("float32").transpose(2, 0, 1))
            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs])[0]
            return predictions
