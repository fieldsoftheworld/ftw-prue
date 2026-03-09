import numpy as np
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import CfgNode
from typing import Optional, Tuple, Dict, Any


class SatellitePredictor(DefaultPredictor):
    """Custom predictor for multi-band satellite imagery."""

    def __init__(self, cfg: CfgNode):
        super().__init__(cfg)

    def __call__(self, original_image: np.ndarray) -> Dict[str, Any]:
        """
        Args:
            original_image (np.ndarray): Multi-band satellite image in (H,W,C) format.
                Can be 8-channel RGBNRGBN (two temporal windows) or 4-channel RGBN.

        Returns:
            predictions (dict): Model predictions including 'instances',
                              'panoptic_seg', and 'sem_seg' if available
        """
        with torch.no_grad():
            height, width = original_image.shape[:2]
            image = torch.as_tensor(original_image.astype("float32").transpose(2, 0, 1))  # transpose to CHW

            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs])[0]

            return predictions
