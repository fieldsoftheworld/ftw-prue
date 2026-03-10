from __future__ import annotations

from typing import Any, Iterable, Optional

import torch
from detectron2.config import get_cfg, CfgNode
from detectron2.projects.deeplab import add_deeplab_config
from mask2former.config import add_maskformer2_config

from ...intermediate_formats import PanopticOutput
from ...converters import convert_d2_panoptic_output
from ..registry import Segmenter, register_model

from .prediction import SatellitePredictor


class Mask2FormerSegmenter:
    """Mask2Former (Detectron2) adapter that returns PanopticOutput.

    Args:
        model_weights: Path to Mask2Former checkpoint (.pth).
        config_file: Path to Mask2Former config YAML (Swin-Small default-style).
        device: Device string, e.g. "cuda" or "cpu" (auto-detected if None).
        model_name: Optional identifier for logging/debugging.
    """

    def __init__(
        self,
        model_weights: str,
        config_file: str,
        device: Optional[str] = None,
        model_name: str = "mask2former",
        **_: Any,
    ):
        self.model_weights = model_weights
        self.config_file = config_file
        self.model_name = model_name

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.cfg = self._build_cfg()
        self.predictor = SatellitePredictor(self.cfg)

    def _build_cfg(self) -> CfgNode:
        cfg = get_cfg()
        cfg.set_new_allowed(True)
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)

        cfg.merge_from_file(self.config_file)
        cfg.MODEL.WEIGHTS = self.model_weights

        # Register FTW panoptic metadata so the model can resolve thing_dataset_id_to_contiguous_id
        from detectron2.data import MetadataCatalog
        from .metadata import get_metadata

        meta = get_metadata()
        for name in list(cfg.DATASETS.TRAIN) + list(cfg.DATASETS.TEST):
            MetadataCatalog.get(name).set(**meta)

        # Ensure correct input format for 8-channel FTW tensors
        if not hasattr(cfg.INPUT, "FORMAT") or cfg.INPUT.FORMAT is None:
            cfg.INPUT.FORMAT = "RGBNRGBN"

        # Inference-only setup
        cfg.MODEL.DEVICE = str(self.device)
        cfg.freeze()
        return cfg

    def predict(self, batch: Any) -> Iterable[PanopticOutput]:
        """Run Mask2Former panoptic inference on a batch of FTW images.

        The batch is expected to be a torch tensor of shape (B, C, H, W)
        or a dict with an "image" key provided by the FTW dataloader.
        """
        # Normalize batch to tensor (B, C, H, W)
        if isinstance(batch, dict):
            if "image" not in batch:
                raise ValueError("Batch dict must contain 'image' key for Mask2Former")
            images = batch["image"]
        elif isinstance(batch, torch.Tensor):
            images = batch
        else:
            raise TypeError(f"Unsupported batch type for Mask2Former: {type(batch)}")

        if images.ndim == 3:
            images = images.unsqueeze(0)
        if images.ndim != 4:
            raise ValueError(f"Expected images of shape (B, C, H, W), got {images.shape}")

        images = images.to(self.device)

        batch_size = images.shape[0]
        for idx in range(batch_size):
            img_chw = images[idx]
            # Convert CHW -> HWC on CPU for SatellitePredictor
            img_hwc = img_chw.detach().cpu().permute(1, 2, 0).numpy()

            preds = self.predictor(img_hwc)
            panoptic = convert_d2_panoptic_output(preds, image_id=idx)
            yield panoptic


@register_model("mask2former", family="d2")
def create_mask2former_segmenter(**kwargs: Any) -> Segmenter:
    """Registry factory for Detectron2/Mask2Former models."""
    if "model_weights" not in kwargs:
        raise ValueError("model_weights is required for Mask2FormerSegmenter")
    if "config_file" not in kwargs:
        raise ValueError("config_file is required for Mask2FormerSegmenter")
    return Mask2FormerSegmenter(**kwargs)
