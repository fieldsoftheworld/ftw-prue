"""
Segmenter adapter for SAM (Segment Anything Model).

This wraps SAM inference logic from sam_controller.py and converts outputs
to InstanceOutput, so the rest of the pipeline stays backend-agnostic.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch

# Add src/ to path for imports
src_path = Path(__file__).parent.parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from intermediate_formats import InstanceOutput
from models.registry import Segmenter, register_model
from converters import convert_sam_output

# Import SAM-specific components (relative imports since we're in sam/ directory)
from .build_sam import sam_model_registry
from .new_automatic_mask_generator import SamAutomaticMaskGenerator
from .sam_mask_decoder_change import new_predict_masks
from .sam_predictor_set_image_change import new_set_image
from .image_mlp import PixelMLP
from segment_anything.modeling.image_encoder import PatchEmbed
from segment_anything.modeling.mask_decoder import MaskDecoder


class SamSegmenter:
    """
    SAM Segmenter adapter that wraps SAM inference and returns InstanceOutput.

    Args:
        model_weights: Path to SAM checkpoint
        model_type: SAM variant ('vit_b', 'vit_l', 'vit_h')
        in_chans: Input channels (3 for RGB, 8 for stacked Sentinel-2)
        device: Device to run inference on
        config: Optional config dict with SAM parameters (pred_iou_thresh, etc.)
        model_name: Identifier for logging/debugging
    """

    def __init__(
        self,
        model_weights: str,
        model_type: str = "vit_h",
        in_chans: int = 3,
        device: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        model_name: str = "sam",
    ):
        self.model_weights = model_weights
        self.model_type = model_type
        self.in_chans = in_chans
        self.model_name = model_name
        self.config = config or {}

        # Determine device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Apply monkey-patching for SAM modifications
        self._apply_monkey_patches()

        # Load SAM model
        self._load_model()

        # Setup mask generator
        self._setup_mask_generator()

    def _apply_monkey_patches(self):
        """Apply SAM monkey-patches for custom behavior."""
        import segment_anything.modeling.image_encoder
        from .new_image_encoder import newImageEncoderViT
        segment_anything.modeling.image_encoder.ImageEncoderViT = newImageEncoderViT

        import segment_anything.predictor
        from .new_sam_predictor import newSamPredictor
        segment_anything.predictor.SamPredictor = newSamPredictor

    def _load_model(self):
        """Load SAM model from checkpoint."""
        # Build model without checkpoint first, so we can modify patch_embed if needed
        # before loading weights
        self.sam = sam_model_registry[self.model_type](
            in_chans=self.in_chans, checkpoint=None
        )

        # Replace patch_embed if needed for 8-channel input
        # (This is a no-op if in_chans was already 8, but kept for clarity)
        if self.in_chans == 8:
            patch_size = getattr(
                self.sam.image_encoder, "patch_size", 16
            )  # Default SAM patch size
            embed_dim = getattr(
                self.sam.image_encoder, "embed_dim", None
            ) or self.sam.image_encoder.patch_embed.proj.out_channels

            self.sam.image_encoder.patch_embed = PatchEmbed(
                kernel_size=(patch_size, patch_size),
                stride=(patch_size, patch_size),
                in_chans=8,
                embed_dim=embed_dim,
            )

        # Now load the checkpoint
        if self.model_weights:
            print(f"[SamSegmenter] Loading checkpoint from {self.model_weights} (map_location=cpu)...", flush=True)
            try:
                state_dict = torch.load(self.model_weights, map_location="cpu")
            except TypeError:
                # Older torch signatures
                with open(self.model_weights, "rb") as f:
                    state_dict = torch.load(f, map_location="cpu")
            
            # Check if checkpoint has different channel count than model
            checkpoint_key = "image_encoder.patch_embed.proj.weight"
            if checkpoint_key in state_dict:
                checkpoint_in_chans = state_dict[checkpoint_key].shape[1]
                model_in_chans = self.sam.image_encoder.patch_embed.proj.weight.shape[1]
                if checkpoint_in_chans != model_in_chans:
                    # Update patch_embed to match checkpoint
                    patch_size = getattr(self.sam.image_encoder, "patch_size", 16)
                    embed_dim = self.sam.image_encoder.patch_embed.proj.out_channels
                    self.sam.image_encoder.patch_embed = PatchEmbed(
                        kernel_size=(patch_size, patch_size),
                        stride=(patch_size, patch_size),
                        in_chans=checkpoint_in_chans,
                        embed_dim=embed_dim,
                    )
                    # Update in_chans to match checkpoint
                    self.in_chans = checkpoint_in_chans
                    print(f"[SamSegmenter] Adjusted model to match checkpoint: {checkpoint_in_chans} input channels", flush=True)
            
            print("[SamSegmenter] Checkpoint loaded, applying state_dict...", flush=True)
            self.sam.load_state_dict(state_dict, strict=True)
            print("[SamSegmenter] Weights loaded.", flush=True)

        # Register pixel mean/std for 8-channel input
        if self.in_chans == 8:
            pixel_mean = 2 * [123.675, 116.28, 103.53, 123.675]
            pixel_std = 2 * [58.395, 57.12, 57.375, 58.395]
            self.sam.register_buffer(
                "pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False
            )
            self.sam.register_buffer(
                "pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False
            )

        self.sam.to(device=self.device)
        self.sam.eval()

    def _setup_mask_generator(self):
        """Setup SAM AutomaticMaskGenerator with config parameters."""
        generator_kwargs = {
            "pred_iou_thresh": self.config.get("pred_iou_thresh", 0.88),
            "stability_score_thresh": self.config.get("stability_score_thresh", 0.95),
            "points_per_side": self.config.get("points_per_side", 32),
            "points_per_batch": self.config.get("points_per_batch", 64),
            "box_nms_thresh": self.config.get("box_nms_thresh", 0.7),
            "crop_n_layers": self.config.get("crop_n_layers", 0),
            "crop_nms_thresh": self.config.get("crop_nms_thresh", 0.7),
            "crop_overlap_ratio": self.config.get("crop_overlap_ratio", 512 / 1500),
            "min_mask_region_area": self.config.get("min_mask_region_area", 0),
        }

        self.generator = SamAutomaticMaskGenerator(self.sam, **generator_kwargs)

        # Apply monkey-patching to mask generator
        self.generator.predictor.model.mask_decoder.predict_masks = (
            new_predict_masks.__get__(
                self.generator.predictor.model.mask_decoder, MaskDecoder
            )
        )
        # Ensure predictor uses newSamPredictor's set_torch_image (allows >= 1 channels)
        # This is needed because the standard predictor only allows 3 channels
        from .new_sam_predictor import newSamPredictor
        import types
        # Replace set_torch_image with the version that allows >= 1 channels (not just 3)
        self.generator.predictor.set_torch_image = types.MethodType(
            newSamPredictor.set_torch_image, self.generator.predictor
        )
        
        # Apply collaborator's set_image replacement
        self.generator.predictor.set_image = new_set_image.__get__(
            self.generator.predictor, type(self.generator.predictor)
        )
        
        # Fix the set_image method to handle numpy arrays correctly
        # (Workaround for collaborator's code that calls apply_image_torch on numpy)
        original_set_image = self.generator.predictor.set_image
        
        def fixed_set_image(self, image, image_format="RGB"):
            """Fixed version that properly converts numpy to torch before apply_image_torch."""
            import torch
            import numpy as np
            
            # Store original size first
            original_size = image.shape[:2] if isinstance(image, np.ndarray) else (image.shape[-2], image.shape[-1])
            
            # Convert numpy HWC to torch BCHW format
            if isinstance(image, np.ndarray):
                # Convert to torch tensor: HWC -> CHW -> BCHW
                # Keep values in [0, 255] range as float32
                image_torch = torch.from_numpy(image).permute(2, 0, 1).float()  # HWC -> CHW, float32
                image_torch = image_torch.unsqueeze(0)  # CHW -> BCHW
                # Ensure values are in [0, 255] range
                if image_torch.max() <= 1.0:
                    image_torch = image_torch * 255.0
            else:
                # Already a torch tensor, ensure it's in BCHW format
                if image.ndim == 3:
                    image_torch = image.unsqueeze(0)  # CHW -> BCHW
                else:
                    image_torch = image
                if image_torch.dtype != torch.float32:
                    image_torch = image_torch.float()
                # Ensure values are in [0, 255] range
                if image_torch.max() <= 1.0:
                    image_torch = image_torch * 255.0
            
            # Move to device
            image_torch = image_torch.to(self.device)
            
            # Apply resize transform (expects BCHW float32 in [0, 255])
            # This resizes to have long side = 1024
            input_image_torch = self.transform.apply_image_torch(image_torch)
            
            # Verify the output shape is correct
            # Should be BCHW with long side = 1024
            assert len(input_image_torch.shape) == 4, f"Expected 4D tensor, got {input_image_torch.shape}"
            assert input_image_torch.shape[0] == 1, f"Expected batch size 1, got {input_image_torch.shape[0]}"
            h, w = input_image_torch.shape[2:]
            max_side = max(h, w)
            expected_size = self.model.image_encoder.img_size
            if max_side != expected_size:
                raise ValueError(
                    f"Resized image has long side {max_side}, expected {expected_size}. "
                    f"Shape: {input_image_torch.shape}"
                )
            
            # Use set_torch_image (should use newSamPredictor's version which allows >= 1 channels)
            self.set_torch_image(input_image_torch, original_size)
        
        # Replace with fixed version
        import types
        self.generator.predictor.set_image = types.MethodType(fixed_set_image, self.generator.predictor)

    def predict(self, batch: Any) -> Iterable[InstanceOutput]:
        """
        Run SAM inference on a batch of images and return InstanceOutput.

        Args:
            batch: Can be:
                - Single image tensor (C, H, W) or (1, C, H, W)
                - List of image tensors
                - Dict with 'image' key

        Yields:
            InstanceOutput for each image in the batch
        """
        # Normalize batch to list of images
        images = self._normalize_batch(batch)

        for idx, image in enumerate(images):
            # Ensure image is tensor
            if isinstance(image, np.ndarray):
                image = torch.from_numpy(image)
            if not isinstance(image, torch.Tensor):
                raise TypeError(f"Expected tensor or numpy array, got {type(image)}")

            # Handle channel selection for 8-channel input
            if image.ndim == 3 and image.shape[0] > self.in_chans:
                image = image[: self.in_chans]
            elif image.ndim == 4 and image.shape[1] > self.in_chans:
                image = image[:, : self.in_chans]

            # Remove batch dimension if present (we'll process one at a time)
            if image.ndim == 4:
                image = image.squeeze(0)  # (1, C, H, W) -> (C, H, W)
            
            # Ensure we have (C, H, W) format
            if image.ndim != 3:
                raise ValueError(f"Expected 3D tensor (C, H, W) or 4D tensor (1, C, H, W), got {image.shape}")

            # Convert from CHW to HWC and to numpy
            # SAM expects HWC uint8 format with values in [0, 255]
            image_np = image.permute(1, 2, 0).cpu().numpy()  # (C, H, W) -> (H, W, C)
            
            # Convert to uint8 if needed
            if image_np.dtype != np.uint8:
                # Handle different value ranges
                if image_np.max() <= 1.0:
                    # Values are in [0, 1] range, scale to [0, 255]
                    image_np = (image_np * 255).astype(np.uint8)
                else:
                    # Values might already be in [0, 255] or other range
                    # Clip to [0, 255] and convert
                    image_np = np.clip(image_np, 0, 255).astype(np.uint8)

            # Generate masks (expects HWC uint8 numpy array)
            outputs = self.generator.generate(image_np)

            # Convert to InstanceOutput using existing converter
            instance_output = convert_sam_output(outputs, image_id=idx)

            yield instance_output

    def _normalize_batch(self, batch: Any) -> list:
        """Normalize various batch formats to a list of images."""
        if isinstance(batch, dict):
            if "image" in batch:
                batch = batch["image"]
            else:
                raise ValueError("Batch dict must contain 'image' key")

        if isinstance(batch, (list, tuple)):
            return list(batch)
        elif isinstance(batch, (torch.Tensor, np.ndarray)):
            # Single image or batch
            if batch.ndim == 3:
                return [batch]
            elif batch.ndim == 4:
                return [batch[i] for i in range(batch.shape[0])]
            else:
                raise ValueError(f"Unexpected image shape: {batch.shape}")
        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}")


@register_model("sam", family="sam")
def create_sam_segmenter(**kwargs) -> Segmenter:
    """
    Registry factory for SAM models.

    Required kwargs:
        model_weights: Path to SAM checkpoint

    Optional kwargs:
        model_type: 'vit_b', 'vit_l', 'vit_h' (default: 'vit_h')
        in_chans: Input channels (default: 3)
        device: Device string (default: auto-detect)
        config: Dict with SAM generator parameters
    """
    return SamSegmenter(**kwargs)

