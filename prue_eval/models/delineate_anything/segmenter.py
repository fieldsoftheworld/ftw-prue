"""
Segmenter adapter for DelineateAnything (YOLO-based field delineation model).

This wraps DelineateAnything inference logic and converts outputs to InstanceOutput,
so the rest of the pipeline stays backend-agnostic.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, Optional, Literal

import torch
import torchvision.transforms.v2 as T

from ...intermediate_formats import InstanceOutput
from ...converters import convert_delineate_anything_output
from ..registry import Segmenter, register_model

try:
    from ftw.models.delineate_anything import DelineateAnything
except ImportError:
    raise ImportError(
        "DelineateAnything requires ftw-tools>=1.2.0. "
        "Install it with: pip install 'ftw-tools>=1.2.0'"
    )


class AdaptiveDelineateAnything(DelineateAnything):
    """
    DelineateAnything with adaptive percentile-based normalization.
    
    This matches the original Delineate Anything implementation which computes
    per-image normalization bounds using 1st and 99th percentiles, rather than
    using fixed division by 3000.
    
    Benefits:
    - Better handles varying atmospheric conditions and seasons
    - Utilizes full dynamic range for better contrast
    - More robust across different regions and acquisition dates
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Replace the transforms with adaptive normalization
        # We'll apply normalization in __call__ instead
        self.transforms = torch.nn.Sequential(
            T.Lambda(lambda x: x.unsqueeze(dim=0) if x.ndim == 3 else x),
            T.Lambda(lambda x: x[:, :3, ...]),  # Extract RGB only
            # Normalization will be done adaptively in __call__
            T.Resize(self.image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ConvertImageDtype(torch.float32),
        ).to(self.device)
    
    def compute_normalization_bounds(self, image: torch.Tensor, percentiles=(1, 99)):
        """
        Compute per-channel normalization bounds using percentiles.
        
        This matches the original Delineate Anything normalization:
        - Computes 1st and 99th percentile per channel
        - Uses only positive (non-zero) values
        - Averages across batch dimension
        
        Args:
            image: Tensor of shape (B, C, H, W) containing RGB channels
            percentiles: Tuple of (lower, upper) percentiles (default: 1, 99)
            
        Returns:
            min_vals: Tensor of shape (C,) with lower bounds per channel
            max_vals: Tensor of shape (C,) with upper bounds per channel
        """
        B, C, H, W = image.shape
        min_vals = []
        max_vals = []
        
        for c in range(C):
            channel_data = image[:, c, :, :].reshape(B, -1)  # (B, H*W)
            
            # For each image in batch, compute percentiles on positive values
            batch_mins = []
            batch_maxs = []
            
            for b in range(B):
                data = channel_data[b]
                # Use positive values only (matching original implementation)
                positive_data = data[data > 0]
                
                if len(positive_data) > 0:
                    p_low = torch.quantile(positive_data, percentiles[0] / 100.0)
                    p_high = torch.quantile(positive_data, percentiles[1] / 100.0)
                    batch_mins.append(p_low)
                    batch_maxs.append(p_high)
                else:
                    # Fallback if no positive values
                    batch_mins.append(torch.tensor(0.0, device=data.device))
                    batch_maxs.append(torch.tensor(3000.0, device=data.device))
            
            # Average across batch (matching original implementation)
            min_vals.append(torch.stack(batch_mins).mean())
            max_vals.append(torch.stack(batch_maxs).mean())
        
        return torch.stack(min_vals), torch.stack(max_vals)
    
    def __call__(self, image: torch.Tensor):
        """
        Forward pass with adaptive normalization.
        
        Args:
            image: Input tensor of shape (B, C, H, W) where C >= 3
        
        Returns:
            List of Results from YOLO model
        """
        # Extract RGB channels first (before normalization)
        if image.ndim == 3:
            image = image.unsqueeze(0)
        
        image = image[:, :3, ...].to(self.device)
        
        # Compute adaptive normalization bounds
        min_vals, max_vals = self.compute_normalization_bounds(image)
        
        # Apply per-channel normalization
        normalized_image = torch.zeros_like(image)
        for c in range(3):
            normalized_image[:, c] = (image[:, c] - min_vals[c]) / (max_vals[c] - min_vals[c] + 1e-8)
        
        # Clip to [0, 1]
        normalized_image = normalized_image.clip(0.0, 1.0)
        
        # Resize to target inference size
        normalized_image = T.Resize(
            self.image_size, 
            interpolation=T.InterpolationMode.BILINEAR
        )(normalized_image)
        
        # Convert to float32
        normalized_image = normalized_image.float()
        
        # Run YOLO inference
        results = self.model.predict(
            normalized_image,
            conf=self.conf_threshold,
            max_det=self.max_detections,
            iou=self.iou_threshold,
            device=self.device,
            half=False,
            verbose=False,
        )
        
        # Rescale masks and boxes to original patch size
        for result in results:
            if result.masks is not None:
                result.masks.orig_shape = self.patch_size
            if result.boxes is not None:
                result.boxes.orig_shape = self.patch_size
        
        return results


class DelineateAnythingSegmenter:
    """
    DelineateAnything Segmenter adapter that wraps DA inference and returns InstanceOutput.

    DelineateAnything is a YOLO-based model that delineates agricultural fields from
    satellite imagery. It takes RGB input and produces instance masks.

    Args:
        model_weights: Path to checkpoint or model variant name
                      ("DelineateAnything", "DelineateAnything-S")
        model_variant: Explicit model variant (if None, inferred from model_weights)
        patch_size: Size of input patches (default: 256)
        resize_factor: Factor to resize patches before inference (default: 2)
        max_detections: Maximum detections per image (default: 100)
        iou_threshold: IoU threshold for NMS (default: 0.3)
        conf_threshold: Confidence threshold for detections (default: 0.05)
        device: Device to run inference on
        config: Optional config dict (not used, kept for compatibility)
        model_name: Identifier for logging/debugging
    """

    def __init__(
        self,
        model_weights: str,
        model_variant: Optional[Literal["DelineateAnything", "DelineateAnything-S"]] = None,
        patch_size: int = 256,
        resize_factor: int = 2,
        max_detections: int = 100,
        iou_threshold: float = 0.3,
        conf_threshold: float = 0.05,
        device: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        model_name: str = "delineate_anything",
    ):
        self.model_weights = model_weights
        self.patch_size = patch_size
        self.resize_factor = resize_factor
        self.max_detections = max_detections
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold
        self.model_name = model_name
        self.config = config or {}

        # Determine device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Determine model variant
        if model_variant is None:
            # Auto-detect from model_weights string
            if model_weights in ["DelineateAnything", "DelineateAnything-S"]:
                model_variant = model_weights
            elif "small" in model_weights.lower() or "-s" in model_weights.lower():
                model_variant = "DelineateAnything-S"
            else:
                model_variant = "DelineateAnything"  # Default to full model

        self.model_variant = model_variant

        # Load model
        self._load_model()

    def _load_model(self):
        """Load DelineateAnything model with adaptive normalization."""
        print(f"Loading DelineateAnything model: {self.model_variant}")
        print("Using adaptive percentile-based normalization (matching original DA implementation)")
        
        # If model_weights is a local path (not a registry name), patch the checkpoints
        if self.model_weights not in ["DelineateAnything", "DelineateAnything-S"]:
            if os.path.exists(self.model_weights):
                # User provided local path - override the registry
                original_checkpoints = DelineateAnything.checkpoints.copy()
                DelineateAnything.checkpoints[self.model_variant] = self.model_weights
                print(f"Using local weights: {self.model_weights}")
                
                # Initialize model with adaptive normalization
                self.model = AdaptiveDelineateAnything(
                    model=self.model_variant,
                    patch_size=self.patch_size,
                    resize_factor=self.resize_factor,
                    max_detections=self.max_detections,
                    iou_threshold=self.iou_threshold,
                    conf_threshold=self.conf_threshold,
                    device=str(self.device),
                )
                
                # Restore original checkpoints
                DelineateAnything.checkpoints = original_checkpoints
            else:
                print(f"Warning: Model weights not found at {self.model_weights}, "
                      f"will try downloading from registry as {self.model_variant}")
                # Fall through to use registry name
                self.model = AdaptiveDelineateAnything(
                    model=self.model_variant,
                    patch_size=self.patch_size,
                    resize_factor=self.resize_factor,
                    max_detections=self.max_detections,
                    iou_threshold=self.iou_threshold,
                    conf_threshold=self.conf_threshold,
                    device=str(self.device),
                )
        else:
            # Use registry name directly (will download if needed)
            self.model = AdaptiveDelineateAnything(
                model=self.model_variant,
                patch_size=self.patch_size,
                resize_factor=self.resize_factor,
                max_detections=self.max_detections,
                iou_threshold=self.iou_threshold,
                conf_threshold=self.conf_threshold,
                device=str(self.device),
            )

        print(f"DelineateAnything model loaded successfully on {self.device}")

    def predict(
        self,
        images: torch.Tensor,
    ) -> Iterable[InstanceOutput]:
        """
        Run inference on a batch of images.

        Args:
            images: Tensor of shape (B, C, H, W) where C >= 3
                   Note: DA uses only RGB channels (first 3), others ignored.
                   For Sentinel-2 data, use temporal_options='windowA' or 'windowB'
                   to get 4-channel RGBN input (RGB extracted automatically).

        Yields:
            InstanceOutput for each image in the batch
        """
        if not isinstance(images, torch.Tensor):
            images = torch.from_numpy(images)

        # Ensure 4D: (B, C, H, W)
        if images.ndim == 3:
            images = images.unsqueeze(0)

        batch_size = images.shape[0]

        # Process each image individually (DA handles batching internally via YOLO)
        for i in range(batch_size):
            image = images[i:i+1]  # Keep batch dimension: (1, C, H, W)
            
            # Run inference - returns list of Results
            results = self.model(image)
            
            # Extract first result (single image)
            result = results[0] if isinstance(results, list) else results
            
            # Convert to InstanceOutput using converter
            instance_output = convert_delineate_anything_output(
                result,
                image_id=i
            )
            
            yield instance_output


# Register both "delineate_anything" and "da" as aliases
@register_model("delineate_anything", family="delineate_anything")
def create_delineate_anything_segmenter(
    model_weights: str,
    model_variant: Optional[str] = None,
    patch_size: int = 256,
    resize_factor: int = 2,
    max_detections: int = 100,
    iou_threshold: float = 0.3,
    conf_threshold: float = 0.05,
    device: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Segmenter:
    """
    Factory function to create DelineateAnythingSegmenter.

    Args:
        model_weights: Path to checkpoint or model variant name
        model_variant: Model variant ("DelineateAnything" or "DelineateAnything-S")
        patch_size: Size of input patches (default: 256)
        resize_factor: Factor to resize patches (default: 2)
        max_detections: Maximum detections per image (default: 100)
        iou_threshold: IoU threshold for NMS (default: 0.3)
        conf_threshold: Confidence threshold (default: 0.05, YOLO default)
        device: Device to run on
        config: Optional config dict
        **kwargs: Additional arguments (ignored)

    Returns:
        DelineateAnythingSegmenter instance
    """
    return DelineateAnythingSegmenter(
        model_weights=model_weights,
        model_variant=model_variant,
        patch_size=patch_size,
        resize_factor=resize_factor,
        max_detections=max_detections,
        iou_threshold=iou_threshold,
        conf_threshold=conf_threshold,
        device=device,
        config=config,
    )


@register_model("da", family="delineate_anything")
def create_da_segmenter(
    model_weights: str,
    model_variant: Optional[str] = None,
    patch_size: int = 256,
    resize_factor: int = 2,
    max_detections: int = 100,
    iou_threshold: float = 0.3,
    conf_threshold: float = 0.05,
    device: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Segmenter:
    """
    Factory function to create DelineateAnythingSegmenter (alias: "da").

    This is an alias for "delineate_anything" for convenience.

    Args:
        Same as create_delineate_anything_segmenter

    Returns:
        DelineateAnythingSegmenter instance
    """
    return create_delineate_anything_segmenter(
        model_weights=model_weights,
        model_variant=model_variant,
        patch_size=patch_size,
        resize_factor=resize_factor,
        max_detections=max_detections,
        iou_threshold=iou_threshold,
        conf_threshold=conf_threshold,
        device=device,
        config=config,
        **kwargs
    )
