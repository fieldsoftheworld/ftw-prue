#!/usr/bin/env python3
"""
Script to run inference on different models and convert outputs to unified Detections format.
"""

import argparse
import os
import pickle
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "models" / "ftw"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "models"))

from detections import Detections
from intermediate_formats import SemanticOutput, InstanceOutput, PanopticOutput
from converters import (
    convert_baseline_output, 
    convert_decode_output, 
    convert_sam_output, 
    convert_delineate_anything_output,
)
from ftw_tools.settings import ALL_COUNTRIES

# Import registry-based segmenters (triggers registration via __init__.py)
try:
    from models.registry import create_segmenter, available_models
    # Trigger adapter registration by importing models module
    import models  # noqa: F401
    REGISTRY_AVAILABLE = True
except ImportError:
    REGISTRY_AVAILABLE = False
    print("Warning: Model registry not available, falling back to MODEL_RUNNERS")


def _sort_dataset_filenames(ds):
    """
    Sort dataset filenames by (country, aoi_id) to ensure consistent ordering.
    This matches the evaluation script's ordering and ensures predictions align with GT.
    
    Handles both FTW dataset (filenames attribute) and DECODE dataset (file_list attribute).
    
    Args:
        ds: Dataset instance with filenames or file_list attribute
    """
    import re
    
    def get_sort_key(filename_dict):
        """Extract (country, aoi_id) from filename dict for sorting."""
        # Try different key names for different dataset types
        path = (filename_dict.get("window_b") or 
                filename_dict.get("window_a") or 
                filename_dict.get("mask") or
                filename_dict.get("image_b") or
                filename_dict.get("image_a") or
                "")
        # Path format: .../country/s2_images/window_b/aoi_id.tif
        # or: .../country/label_masks/.../aoi_id.tif
        match = re.search(r'/([^/]+)/(?:s2_images/[^/]+|label_masks/[^/]+)/([^/]+)\.tif$', path)
        if match:
            country = match.group(1)
            aoi_id_str = match.group(2)
            # Try to convert to int for numeric sorting, fall back to string
            try:
                aoi_id = int(aoi_id_str)
            except ValueError:
                aoi_id = aoi_id_str
            return (country, aoi_id)
        return ("", "")
    
    # Handle FTW dataset (filenames attribute)
    if hasattr(ds, 'filenames') and ds.filenames:
        ds.filenames = sorted(ds.filenames, key=get_sort_key)
    # Handle DECODE dataset (file_list attribute)
    elif hasattr(ds, 'file_list') and ds.file_list:
        ds.file_list = sorted(ds.file_list, key=get_sort_key)

def run_ftw_inference(
    data_dir: str,
    model_weights: str,
    output_dir: str,
    **kwargs
) -> List[Detections]:
    """
    Run inference using FTW baseline model.
    
    ⚠️  DEPRECATED: This function is deprecated in favor of the registry-based
    segmenter framework. Use `run_registry_inference()` with `--use_registry` flag instead.
    
    This function is kept for backward compatibility and will be removed in a future version.
    
    Args:
        data_dir: Directory containing the dataset
        model_weights: Path to model checkpoint
        output_dir: Directory to save outputs
        
    Returns:
        List of Detections objects
    """
    print("Running FTW baseline inference...")
    
    # Import FTW components
    from ftw_tools.training.datasets import FTW
    from ftw_tools.training.datamodules import preprocess
    from ftw_tools.training.trainers import CustomSemanticSegmentationTask
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = CustomSemanticSegmentationTask.load_from_checkpoint(model_weights, map_location="cpu")
    model = trainer.model.eval().to(device)
    
    # Sort countries to ensure consistent ordering
    countries = kwargs.get("countries", ["belgium"])
    if isinstance(countries, str):
        countries = [countries]
    countries_sorted = sorted(countries)
    
    # Create dataset with standard "stacked" temporal option (8-channel RGBNRGBN)
    ds = FTW(
        root=data_dir,
        countries=countries_sorted,
        split=kwargs.get("split", "test"),
        transforms=preprocess,
        load_boundaries=False,
        temporal_options="stacked",  # Force stacked option for consistency
    )
    
    # Sort filenames to ensure consistent ordering with evaluation
    _sort_dataset_filenames(ds)
    
    dl = DataLoader(ds, batch_size=kwargs.get("batch_size", 16), shuffle=False, num_workers=4)
    
    detections_list = []
    
    # Store logits if requested
    logits_list = [] if kwargs.get("save_logits", False) else None
    
    # Get total number of samples for progress bar
    total_samples = len(ds)
    batch_size = kwargs.get("batch_size", 16)
    total_batches = len(dl)
    
    print(f"Processing {total_samples} samples in {total_batches} batches (batch_size={batch_size})")
    
    with torch.inference_mode():
        # Progress bar shows batches, but we'll track samples in description
        pbar = tqdm(dl, desc=f"Processing FTW model (0/{total_samples} samples)", total=total_batches, unit="batch")
        for batch in pbar:
            images = batch["image"].to(device)
            
            # Get raw model outputs (before argmax)
            raw_outputs = model(images)
            
            # Compute softmax probabilities for confidence, and argmax for instance extraction
            if isinstance(raw_outputs, torch.Tensor):
                # raw logits -> probs
                probs = torch.softmax(raw_outputs, dim=1)
                argmax_outputs = torch.argmax(raw_outputs, dim=1).cpu().numpy().astype(np.uint8)
                probs_np = probs.detach().cpu().numpy()  # (B, C, H, W)
            else:
                # If outputs are numpy probabilities already
                # Ensure shape (B, C, H, W)
                if raw_outputs.ndim == 3:
                    probs_np = np.expand_dims(raw_outputs, 0)
                else:
                    probs_np = raw_outputs
                argmax_outputs = np.argmax(probs_np, axis=1).astype(np.uint8)
            
            # Process each sample in the batch
            for i in range(argmax_outputs.shape[0]):
                # Create binary field mask from argmax output (matches paper approach)
                # FTW 3-class system: 0=background, 1=ag_field, 2=boundary
                field_mask = (argmax_outputs[i] == 1).astype(np.uint8)  # Only class 1 (ag_field)
                
                # Create Detections directly from binary mask (matches paper's approach)
                # Use the same polygonization method as our fixed Detections.from_semantic_logits
                import rasterio.features
                import shapely.geometry
                
                masks = []
                xyxys = []
                confidences = []
                class_ids = []
                
                # Extract shapes using rasterio (matches paper's approach exactly)
                for geom, val in rasterio.features.shapes(field_mask):
                    if val == 1:  # Only process field pixels
                        shapely_geom = shapely.geometry.shape(geom)
                        
                        # Skip small areas
                        if shapely_geom.area < kwargs.get("min_area", 0):
                            continue
                        
                        # Create mask for this shape
                        mask = rasterio.features.rasterize(
                            [shapely_geom], 
                            out_shape=field_mask.shape,
                            fill=0,
                            default_value=1,
                            dtype=np.uint8
                        )
                        masks.append(mask)
                        
                        # Get bounding box from shapely geometry
                        bounds = shapely_geom.bounds
                        xyxys.append([bounds[0], bounds[1], bounds[2], bounds[3]])
                        
                        # Compute per-instance confidence as mean probability of field class
                        # Use softmax probabilities computed above
                        try:
                            field_probs = probs_np[i, 1]  # class 1 = ag_field
                            inst_conf = float(field_probs[mask == 1].mean()) if np.any(mask == 1) else 0.0
                            # Clamp to [0, 1]
                            if inst_conf < 0.0:
                                inst_conf = 0.0
                            if inst_conf > 1.0:
                                inst_conf = 1.0
                            confidences.append(inst_conf)
                        except Exception:
                            # Fallback if probabilities unavailable
                            confidences.append(1.0)
                
                detections = Detections(
                    xyxy=np.array(xyxys) if xyxys else np.empty((0, 4)),
                    mask=np.array(masks) if masks else None,
                    confidence=np.array(confidences) if confidences else None,
                    class_id=np.array(class_ids) if class_ids else None
                )

                detections_list.append(detections)
                
                # Update progress bar to show sample count
                pbar.set_description(f"Processing FTW model ({len(detections_list)}/{total_samples} samples)")
                
                # Store raw logits if requested (store the argmax output as "logits")
                if logits_list is not None:
                    # Create a dummy SemanticOutput for compatibility
                    from intermediate_formats import SemanticOutput
                    # Convert argmax back to one-hot for compatibility
                    one_hot = np.zeros((3, field_mask.shape[0], field_mask.shape[1]))
                    one_hot[0] = (argmax_outputs[i] == 0).astype(np.float32)  # background
                    one_hot[1] = (argmax_outputs[i] == 1).astype(np.float32)  # field
                    one_hot[2] = (argmax_outputs[i] == 2).astype(np.float32)  # boundary
                    semantic_output = SemanticOutput(logits=one_hot, image_id=i)
                    logits_list.append(semantic_output)
    
    if logits_list is not None:
        return detections_list, logits_list
    return detections_list


def run_decode_inference(
    data_dir: str,
    model_weights: str,
    output_dir: str,
    **kwargs
) -> List[Detections]:
    """
    Run inference using DECODE model.
    
    ⚠️  DEPRECATED: This function is deprecated in favor of the registry-based
    segmenter framework. Use `run_registry_inference()` with `--use_registry` flag instead.
    
    This function is kept for backward compatibility and will be removed in a future version.
    
    Args:
        data_dir: Directory containing the dataset
        model_weights: Path to model weights
        output_dir: Directory to save outputs
        
    Returns:
        List of Detections objects
    """
    print("Running DECODE inference...")
    
    # Import DECODE components using the same approach as our working test script
    import sys
    import yaml
    decode_path = Path(__file__).parent.parent / "src" / "models" / "decode"
    sys.path.insert(0, str(decode_path))
    
    from fractal_resunet.models.semanticsegmentation.FracTAL_ResUNet import FracTAL_ResUNet_cmtsk as decode_model
    from data_module import FTWMultiCountryDataset
    
    # Load config if provided
    config_file = kwargs.get("config_file")
    if config_file and os.path.exists(config_file):
        with open(config_file, "r") as f:
            cfg = yaml.safe_load(f)
    else:
        # Use default config values
        cfg = {
            "data": {
                "in_channels": 8,
                "n_classes": 2,
                "temporal_option": "stacked",
                "crop_size": [256, 256]
            },
            "model": {
                "nfilters_init": 32,
                "depth": 6,
                "ftdepth": 5,
                "psp_depth": 4,
                "norm_type": "GroupNorm",
                "norm_groups": 4,
                "nheads_start": 4
            }
        }
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = decode_model(
        nfilters_init=cfg["model"]["nfilters_init"],
        NClasses=cfg["data"]["n_classes"],
        depth=cfg["model"]["depth"],
        ftdepth=cfg["model"]["ftdepth"],
        psp_depth=cfg["model"]["psp_depth"],
        norm_type=cfg["model"]["norm_type"],
        norm_groups=cfg["model"]["norm_groups"],
        nheads_start=cfg["model"]["nheads_start"],
        in_channels=cfg["data"]["in_channels"],
    ).to(device)
    
    # Load weights
    checkpoint = torch.load(model_weights, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    
    # Create dataset using the same approach as our working test script
    countries = kwargs.get("countries", ["belgium"])
    if isinstance(countries, str):
        countries = [countries]
    
    # Sort countries to ensure consistent ordering (matches GT script)
    countries_sorted = sorted(countries)
    
    dataset = FTWMultiCountryDataset(
        root_dir=data_dir,
        countries=countries_sorted,
        split=kwargs.get("split", "test"),
        load_boundaries=False,
        temporal_option=cfg["data"]["temporal_option"],
        crop_size=tuple(cfg["data"]["crop_size"]),
        num_samples=kwargs.get("num_samples", -1),
    )
    
    # Sort file_list to ensure consistent ordering with evaluation
    _sort_dataset_filenames(dataset)
    
    dataloader = DataLoader(
        dataset,
        batch_size=kwargs.get("batch_size", 1),
        shuffle=False,
        num_workers=kwargs.get("num_workers", 0),
    )
    
    detections_list = []
    
    with torch.inference_mode():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing DECODE model")):
            # Extract batch components (same as our working test script)
            if len(batch) == 6:  # (win_a, win_b, images, mask, boundary, distance)
                win_a, win_b, images, mask, boundary, distance = batch
            else:
                images, mask = batch[:2]
                win_a, win_b = None, None
            
            # Move to device
            images = images.to(device)
            
            # Run inference
            preds = model(images)
            
            # DECODE returns a tuple: (seg_logits, boundary_logits, distance)
            seg_logits, boundary_logits, distance = preds
            
            # Process each sample in the batch
            for i in range(images.shape[0]):
                # Create the 3-tuple output for convert_decode_output
                sample_output = (
                    seg_logits[i:i+1],  # Keep batch dimension for consistency
                    boundary_logits[i:i+1],
                    distance[i:i+1]
                )
                
                # Convert to SemanticOutput using our working converter
                semantic_output = convert_decode_output(
                    sample_output,
                    image_id=batch_idx * dataloader.batch_size + i
                )
                
                # Convert to Detections using our working converter
                detections = semantic_output.to_detections(
                    field_class_id=1,  # Field class is index 1
                    min_area=kwargs.get("min_area", 0)
                )
                detections_list.append(detections)
    
    return detections_list


def run_sam_inference(
    data_dir: str,
    model_weights: str,
    output_dir: str,
    **kwargs
) -> Dict[str, List[Detections]]:
    """
    Run inference using SAM model via AutomaticMaskGenerator and convert to Detections.
    
    ⚠️  DEPRECATED: This function is deprecated in favor of the registry-based
    segmenter framework. Use `run_registry_inference()` with `--use_registry` flag instead.
    
    Note: This legacy function returns a per-country dict, while the registry path returns
    a flat list. The saving logic handles both formats.
    
    This function is kept for backward compatibility and will be removed in a future version.
    
    Uses custom SAM setup (monkey-patching, custom registry, etc.)
    to match sam_controller.py behavior.

    Returns a mapping of country -> list[Detections] to enable per-country saving.
    """
    print("Running SAM inference...")
    
    from pathlib import Path as _Path
    import sys as _sys
    sam_path = _Path(__file__).parent.parent / "src" / "models" / "sam"
    if str(sam_path) not in _sys.path:
        _sys.path.insert(0, str(sam_path))
    
    # Apply monkey-patching first (same as sam_controller.py)
    import segment_anything.modeling.image_encoder
    from models.sam.new_image_encoder import newImageEncoderViT
    segment_anything.modeling.image_encoder.ImageEncoderViT = newImageEncoderViT

    import segment_anything.predictor
    from models.sam.new_sam_predictor import newSamPredictor
    segment_anything.predictor.SamPredictor = newSamPredictor

    # Use our custom registry and automatic mask generator (same as sam_controller.py)
    from models.sam.build_sam import sam_model_registry
    from models.sam.new_automatic_mask_generator import SamAutomaticMaskGenerator
    from segment_anything.modeling.image_encoder import PatchEmbed
    from segment_anything.modeling.mask_decoder import MaskDecoder
    from models.sam.sam_mask_decoder_change import new_predict_masks
    from models.sam.sam_predictor_set_image_change import new_set_image

    # Load config if provided to get model parameters
    config_file = kwargs.get("config_file")
    if config_file and os.path.exists(config_file):
        import yaml
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        in_chans = config.get('sam_model', {}).get('in_chans', 3)
        change_chan_num = config.get('sam_model', {}).get('change_chan_num', False)
        use_img_proj = config.get('img_proj', {}).get('use_img_proj', False)
    else:
        # Defaults (can be overridden by kwargs)
        in_chans = kwargs.get('in_chans', 3)
        change_chan_num = kwargs.get('change_chan_num', False)
        use_img_proj = kwargs.get('use_img_proj', False)

    # Load model (always use vit_h for fine-tuned checkpoints; same as sam_controller.py)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get model_type from kwargs or config (default to vit_h for fine-tuned checkpoints)
    # Note: argparse will set this to the default ('vit_h') if not provided
    model_type = kwargs.get("sam_model_type", "vit_h")  # Default to vit_h (not vit_b)
    
    # If model_type was explicitly passed as empty string or None, reset to default
    if not model_type or model_type == "":
        model_type = "vit_h"
    
    # Allow config file to override (but usually not set in config)
    if config_file and os.path.exists(config_file):
        config_model_type = config.get('sam_model', {}).get('model_type')
        if config_model_type:
            model_type = config_model_type
    
    print(f"Loading SAM {model_type} checkpoint from {model_weights} (in_chans={in_chans}, change_chan_num={change_chan_num})...", flush=True)

    # Determine build channel count by inspecting checkpoint when possible
    detected_ckpt_in_chans = None
    try:
        ckpt = torch.load(model_weights, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        # Find the patch_embed proj weight key robustly
        proj_w_key = None
        for k in state_dict.keys():
            if k.endswith("patch_embed.proj.weight"):
                proj_w_key = k
                break
        if proj_w_key is not None:
            detected_ckpt_in_chans = int(state_dict[proj_w_key].shape[1])
            print(f"  Detected checkpoint input channels: {detected_ckpt_in_chans} via {proj_w_key}", flush=True)
    except Exception as _e:
        print(f"  Warning: could not inspect checkpoint for in_chans ({_e}); using heuristics", flush=True)

    if detected_ckpt_in_chans in (3, 8):
        build_in_chans = detected_ckpt_in_chans
        print(f"  Building with detected checkpoint in_chans={build_in_chans}", flush=True)
    else:
        # Heuristic fallback
        if change_chan_num:
            build_in_chans = 3
            print(f"  Building with in_chans={build_in_chans} (change_chan_num=True)", flush=True)
        elif in_chans == 8:
            build_in_chans = 3
            print(f"  Heuristic: in_chans=8 specified; building with in_chans={build_in_chans} first", flush=True)
        else:
            build_in_chans = in_chans
            print(f"  Building with in_chans={build_in_chans} (from config)", flush=True)
    
    sam = sam_model_registry[model_type](in_chans=build_in_chans, checkpoint=model_weights)
    
    # Apply patch_embed replacement only if runtime input channels differ from build/checkpoint channels
    if in_chans != build_in_chans and in_chans == 8:
        # Get patch_size - try from attribute (monkey-patched newImageEncoderViT) or from conv layer
        if hasattr(sam.image_encoder, 'patch_size'):
            patch_size = sam.image_encoder.patch_size
        else:
            # Fallback: get from patch_embed.proj.kernel_size (should be 16 for SAM)
            # Conv2d kernel_size is a tuple, get first element
            patch_size = sam.image_encoder.patch_embed.proj.kernel_size[0]
        
        # Get embed_dim - try from attribute or from patch_embed
        if hasattr(sam.image_encoder, 'embed_dim'):
            embed_dim = sam.image_encoder.embed_dim
        else:
            # Fallback: get from patch_embed.proj.out_channels
            embed_dim = sam.image_encoder.patch_embed.proj.out_channels
        
        sam.image_encoder.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=8,  # Hardcoded to 8 as in sam_controller.py line 97
            embed_dim=embed_dim,
        )
        print(f"Applied patch_embed replacement for 8-channel input (patch_size={patch_size}, embed_dim={embed_dim})")

    # Register pixel mean/std for 8-channel input (same as sam_controller.py lines 108-114)
    if in_chans == 8:
        pixel_mean = 2*[123.675, 116.28, 103.53, 123.675]
        pixel_std = 2*[58.395, 57.12, 57.375, 58.395]
        sam.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        sam.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        print("Registered pixel mean/std for 8-channel input")

    sam.to(device=device)
    sam.eval()

    # Create mask generator with reasonable defaults
    generator = SamAutomaticMaskGenerator(
        sam,
        pred_iou_thresh=kwargs.get("pred_iou_thresh", kwargs.get("iou_thresh", 0.88)),
        stability_score_thresh=kwargs.get("stability_score_thresh", kwargs.get("stability_thresh", 0.95)),
        points_per_side=kwargs.get("points_per_side", 32),
        points_per_batch=kwargs.get("points_per_batch", 64),
        box_nms_thresh=kwargs.get("box_nms_thresh", 0.7),
        crop_n_layers=kwargs.get("crop_n_layers", 0),
        crop_nms_thresh=kwargs.get("crop_nms_thresh", 0.7),
        crop_overlap_ratio=kwargs.get("crop_overlap_ratio", 512/1500),
        min_mask_region_area=kwargs.get("min_mask_region_area", 0),
    )
    
    # Apply monkey-patching to mask generator (same as sam_controller.py lines 598-599)
    generator.predictor.model.mask_decoder.predict_masks = new_predict_masks.__get__(
        generator.predictor.model.mask_decoder, MaskDecoder
    )
    generator.predictor.set_image = new_set_image.__get__(generator.predictor, type(generator.predictor))

    # Dataset from ftw_tools for consistency
    from ftw_tools.training.datasets import FTW
    from ftw_tools.training.datamodules import preprocess
    
    countries = kwargs.get("countries", ["belgium"])
    if isinstance(countries, str):
        countries = [countries]

    split = kwargs.get("split", "test")
    temporal_opt = kwargs.get("temporal_options", "stacked")

    per_country_dets: Dict[str, List[Detections]] = {}

    # Sort countries to ensure consistent ordering (matches GT script)
    countries_sorted = sorted(countries)

    for country in countries_sorted:
        print(f"Processing country: {country}")
        ds = FTW(
            root=data_dir,
            countries=[country],
            split=split,
            transforms=preprocess,
            load_boundaries=False,
            temporal_options=temporal_opt,
        )
        
        # Sort filenames to ensure consistent ordering with evaluation
        _sort_dataset_filenames(ds)
    
        # Handle img_proj if enabled (same as sam_controller.py)
        img_proj_model = None
        if use_img_proj:
            if config_file and os.path.exists(config_file):
                img_model_params = config.get('img_proj', {}).get('img_model_params', [8, 3, 64, 4])
                img_ckpt = config.get('img_proj', {}).get('img_ckpt', False)
            else:
                img_model_params = kwargs.get('img_model_params', [8, 3, 64, 4])
                img_ckpt = kwargs.get('img_ckpt', False)
            
            from image_mlp import PixelMLP
            img_proj_model = PixelMLP(*img_model_params)
            if img_ckpt and img_ckpt is not False:
                ckpt = torch.load(img_ckpt)
                img_proj_model.load_state_dict(ckpt['state_dict'])
            img_proj_model.to(device)
            img_proj_model.eval()

        detections_list: List[Detections] = []
        for idx in tqdm(range(len(ds)), desc=f"SAM {country}"):
            sample = ds[idx]
            image = sample["image"]  # (C, H, W), typically 8-ch for stacked

            # Prepare image as tensor (1, C, H, W) on device, same as sam_controller.py lines 615-619
            if isinstance(image, torch.Tensor):
                image_tensor = image.to(device)
            else:
                image_tensor = torch.from_numpy(image).to(device)
            
            # Apply img_proj if enabled (same as sam_controller.py line 617)
            if use_img_proj:
                image_tensor = img_proj_model(image_tensor.unsqueeze(0))
            else:
                # Use model_in_chans channels (same as sam_controller.py line 619)
                image_tensor = image_tensor[:in_chans].unsqueeze(0)  # (1, C, H, W)

            # Generate masks (pass tensor directly, same as sam_controller.py line 622)
            outputs = generator.generate(image_tensor)
        
            # Convert via unified converters → InstanceOutput → Detections
            instance_output = convert_sam_output(outputs, image_id=idx)
            detections = instance_output.to_detections(
                min_area=kwargs.get("min_area", 0),
                score_threshold=kwargs.get("confidence_threshold", 0.0),
            )

            detections_list.append(detections)
    
        per_country_dets[country] = detections_list

    return per_country_dets


def run_delineate_anything_inference(
    data_dir: str,
    model_weights: str,
    output_dir: str,
    **kwargs
) -> List[Detections]:
    """
    Run inference using DelineateAnything model.
    
    ⚠️  NOTE: This function is NOT yet deprecated because DelineateAnything does not
    have a registry-based segmenter implementation yet. Once a registry segmenter is
    created for DelineateAnything, this function should be deprecated.
    
    Args:
        data_dir: Directory containing the dataset
        model_weights: Path to model weights (or model name from registry)
        output_dir: Directory to save outputs
        
    Returns:
        List of Detections objects
    """
    print("Running DelineateAnything inference...")
    
    # Import components
    from ftw_tools.training.datasets import FTW
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Determine model variant
    # Priority: 1) explicit model_variant, 2) infer from model_weights string
    model_variant = kwargs.get("delany_model_variant", None)
    
    if model_variant is None:
        # Auto-detect from model_weights
        if model_weights in ["DelineateAnything", "DelineateAnything-S"]:
            model_variant = model_weights  # Registry name
        elif "small" in model_weights.lower() or "-s" in model_weights.lower():
            model_variant = "DelineateAnything-S"
        else:
            model_variant = "DelineateAnything"  # Default to full model
    
    print(f"Using model variant: {model_variant}")
    
    # If model_weights is a registry name, let DelineateAnything handle download
    # Otherwise, check if local path exists
    if model_weights not in ["DelineateAnything", "DelineateAnything-S"]:
        if not os.path.exists(model_weights):
            print(f"Warning: Model weights not found at {model_weights}, will try downloading from registry")
            # Override to use registry
            model_weights = model_variant
    
    # Initialize DelineateAnything model
    # Use confidence_threshold for both model and post-filtering (no double filtering)
    # For DelineateAnything, default should be 0.05 (YOLO default), not 0.5
    conf_thresh = kwargs.get("confidence_threshold", 0.5)
    if conf_thresh == 0.5:  # User didn't specify, use DelineateAnything default
        conf_thresh = 0.05
        print(f"Using DelineateAnything default confidence threshold: {conf_thresh}")
    
    # Patch the checkpoints dict to use local weights if provided
    from ftw_tools.inference.models import DelineateAnything
    if model_weights not in ["DelineateAnything", "DelineateAnything-S"]:
        # User provided local path - override the registry
        original_checkpoints = DelineateAnything.checkpoints.copy()
        DelineateAnything.checkpoints[model_variant] = model_weights
        print(f"Using local weights: {model_weights}")
    
    model = DelineateAnything(
        model=model_variant,
        patch_size=kwargs.get("patch_size", 256),
        resize_factor=kwargs.get("resize_factor", 2),
        max_detections=kwargs.get("max_detections", 100),
        iou_threshold=kwargs.get("iou_threshold", 0.3),
        conf_threshold=conf_thresh,  # Use consistent threshold
        device=device,
    )
    
    # Restore original checkpoints if we modified them
    if model_weights not in ["DelineateAnything", "DelineateAnything-S"]:
        DelineateAnything.checkpoints = original_checkpoints
    
    # Create dataset - DelineateAnything uses RGB only
    # Force single window if user didn't explicitly specify (stacked doesn't make sense for RGB-only model)
    temporal_opt = kwargs.get("temporal_options", "stacked")
    if temporal_opt == "stacked":
        temporal_opt = "windowA"  # Override to single window for RGB model
        print(f"DelineateAnything requires single window - using 'windowA' instead of 'stacked'")
    
    # Sort countries to ensure consistent ordering (matches GT script)
    countries = kwargs.get("countries", ["belgium"])
    if isinstance(countries, str):
        countries = [countries]
    countries_sorted = sorted(countries)
    
    ds = FTW(
        root=data_dir,
        countries=countries_sorted,
        split=kwargs.get("split", "test"),
        transforms=None,  # DelineateAnything handles normalization internally
        load_boundaries=False,
        temporal_options=temporal_opt,  # Single window (4ch RGBN), model extracts RGB
    )
    
    # Sort filenames to ensure consistent ordering with evaluation
    _sort_dataset_filenames(ds)
    
    detections_list = []
    
    # Store logits if requested
    logits_list = [] if kwargs.get("save_logits", False) else None
    
    # Process images individually (DelineateAnything handles batching internally via YOLO)
    for i in tqdm(range(len(ds)), desc="Processing DelineateAnything"):
        sample = ds[i]
        image = sample["image"]  # Shape: (C, H, W)
        
        # Ensure image is a torch tensor
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image)
        
        # Run inference - returns list of Results
        results = model(image.unsqueeze(0))  # Add batch dimension
        
        # Extract first result (single image)
        result = results[0] if isinstance(results, list) else results
        
        # Convert to InstanceOutput using converter
        instance_output = convert_delineate_anything_output(
            result,
            image_id=i
        )
        
        # Store raw outputs if requested
        if logits_list is not None:
            logits_list.append(instance_output)
        
        # Convert to Detections (no additional score filtering - already done by YOLO)
        detections = instance_output.to_detections(
            min_area=kwargs.get("min_area", 0),
            score_threshold=0.0  # Don't filter again - model already applied conf_threshold
        )
        
        detections_list.append(detections)
    
    if logits_list is not None:
        return detections_list, logits_list
    return detections_list


def run_registry_inference(
    model_name: str,
    data_dir: str,
    model_weights: str,
    output_dir: str,
    **kwargs
) -> List[Detections]:
    """
    Run inference using registry-based segmenters.
    
    This is a unified inference path that works with all registry models.
    """
    print(f"Running {model_name} inference using registry...")
    
    # Prepare segmenter arguments
    segmenter_kwargs = {
        "model_weights": model_weights,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    
    # Add model-specific arguments
    if model_name == "sam":
        segmenter_kwargs["model_type"] = kwargs.get("sam_model_type", "vit_h")
        segmenter_kwargs["in_chans"] = kwargs.get("in_chans", 8)
        if kwargs.get("config_file"):
            import yaml
            with open(kwargs["config_file"], "r") as f:
                config = yaml.safe_load(f)
            segmenter_kwargs["config"] = config
    elif model_name == "decode":
        if kwargs.get("config_file"):
            segmenter_kwargs["config"] = kwargs["config_file"]
    
    # Create segmenter
    segmenter = create_segmenter(model_name, **segmenter_kwargs)
    
    # Load dataset (same approach as legacy runners)
    from ftw_tools.training.datasets import FTW
    from ftw_tools.training.datamodules import preprocess
    
    countries = kwargs.get("countries", ["belgium"])
    if isinstance(countries, str):
        countries = [countries]
    countries_sorted = sorted(countries)
    
    # Create dataset
    ds = FTW(
        root=data_dir,
        countries=countries_sorted,
        split=kwargs.get("split", "test"),
        transforms=preprocess if model_name != "sam" else None,  # SAM handles preprocessing differently
        load_boundaries=False,
        temporal_options=kwargs.get("temporal_options", "stacked") if model_name in ["ftw", "decode"] else "stacked",
    )
    
    # Sort filenames
    _sort_dataset_filenames(ds)
    
    # Create dataloader
    batch_size = kwargs.get("batch_size", 16)
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)
    
    detections_list = []
    logits_list = [] if kwargs.get("save_logits", False) else None
    
    # Run inference
    with torch.inference_mode():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Processing {model_name} model")):
            # Extract images from batch
            if isinstance(batch, dict):
                images = batch.get("image", batch.get("images"))
            elif isinstance(batch, (list, tuple)):
                images = batch[0] if len(batch) > 0 else batch
            else:
                images = batch
            
            # Run segmenter inference
            outputs = list(segmenter.predict(images))
            
            # Process each output
            for i, output in enumerate(outputs):
                # Store raw outputs if requested
                if logits_list is not None:
                    logits_list.append(output)
                
                # Convert to Detections
                if isinstance(output, SemanticOutput):
                    detections = output.to_detections(
                        field_class_id=1,
                        min_area=kwargs.get("min_area", 0)
                    )
                elif isinstance(output, InstanceOutput):
                    detections = output.to_detections(
                        min_area=kwargs.get("min_area", 0),
                        score_threshold=kwargs.get("confidence_threshold", 0.5)
                    )
                else:
                    raise ValueError(f"Unexpected output type: {type(output)}")
                
                detections_list.append(detections)
    
    if logits_list is not None:
        return detections_list, logits_list
    return detections_list


# Model registry
# ⚠️  DEPRECATED: This dictionary maps to legacy inference functions.
# Once registry-based inference is fully verified, these functions will be removed.
# Use `run_registry_inference()` with `--use_registry` flag instead.
MODEL_RUNNERS = {
    "ftw": run_ftw_inference,
    "ftw-cli": run_ftw_inference,
    "decode": run_decode_inference,
    "sam": run_sam_inference,
    "delineate_anything": run_delineate_anything_inference,
    "da": run_delineate_anything_inference
}


def main():
    parser = argparse.ArgumentParser(
        description="Run model inference and convert to unified Detections format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    model_choices = list(MODEL_RUNNERS.keys())
    if REGISTRY_AVAILABLE:
        # Add registry models to choices
        registry_models = list(available_models().keys())
        model_choices = list(set(model_choices + registry_models))
    
    parser.add_argument(
        "--model", 
        type=str, 
        required=True,
        choices=model_choices,
        help="Model type to run inference with"
    )
    parser.add_argument(
        "--use_registry",
        action="store_true",
        default=False,
        help="Use new registry-based segmenters instead of legacy MODEL_RUNNERS (experimental)"
    )
    parser.add_argument(
        "--data_dir", 
        type=str, 
        required=True,
        help="Directory containing the dataset"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True,
        help="Directory to save inference outputs"
    )
    
    # Model-specific arguments
    parser.add_argument(
        "--model_weights", 
        type=str,
        help="Path to model weights/checkpoint"
    )
    parser.add_argument(
        "--config_file", 
        type=str,
        help="Path to model config file"
    )
    
    # Dataset arguments
    parser.add_argument(
        "--countries", 
        nargs="+", 
        default=["belgium"],
        help="Countries to process"
    )
    parser.add_argument(
        "--split", 
        type=str, 
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to use"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=16,
        help="Batch size for processing"
    )
    
    # Output arguments
    parser.add_argument(
        "--output_name", 
        type=str, 
        default=None,
        help="Name of the output file (defaults to {model}_detections.pkl)"
    )
    parser.add_argument(
        "--save_logits", 
        action="store_true",
        default=False,
        help="Save raw model logits alongside detections for debugging"
    )
    parser.add_argument(
        "--min_area", 
        type=int, 
        default=0,
        help="Minimum area (pixels) to keep instances during conversion"
    )
    parser.add_argument(
        "--confidence_threshold", 
        type=float, 
        default=0.5,
        help="Confidence threshold for filtering (default: 0.5 for most models, 0.05 for DelineateAnything)"
    )
    
    # Organize arguments by model type for clarity
    sam_group = parser.add_argument_group('SAM-specific options', 'Arguments only used when --model sam')
    sam_group.add_argument(
        "--sam_model_type", 
        type=str, 
        default="vit_h",
        help="SAM model type (vit_b, vit_l, vit_h). Default: vit_h for fine-tuned checkpoints."
    )
    sam_group.add_argument(
        "--sam_in_chans",
        type=int,
        default=8,
        help="Input channels for SAM (3 for RGB, 8 for stacked Sentinel-2). Default: 8"
    )
    
    decode_group = parser.add_argument_group('DECODE-specific options', 'Arguments only used when --model decode')
    # config_file is already defined above, but note it's mainly for DECODE
    
    ftw_group = parser.add_argument_group('FTW-specific options', 'Arguments only used when --model ftw')
    ftw_group.add_argument(
        "--temporal_options", 
        type=str, 
        default="stacked",
        choices=["stacked", "window_a", "window_b"],
        help="Temporal processing option for FTW (default: stacked)"
    )
    
    delany_group = parser.add_argument_group('DelineateAnything-specific options', 'Arguments only used when --model delineate_anything or --model da')
    delany_group.add_argument(
        "--delany_model_variant",
        type=str,
        default=None,
        choices=["DelineateAnything", "DelineateAnything-S"],
        help="DelineateAnything model variant (auto-detected from weights path if not specified)"
    )
    delany_group.add_argument(
        "--patch_size",
        type=int,
        default=256,
        help="Patch size for DelineateAnything (default: 256)"
    )
    delany_group.add_argument(
        "--resize_factor",
        type=int,
        default=2,
        help="Resize factor for DelineateAnything (default: 2)"
    )
    delany_group.add_argument(
        "--max_detections",
        type=int,
        default=100,
        help="Max detections per image for DelineateAnything (default: 100)"
    )
    delany_group.add_argument(
        "--iou_threshold",
        type=float,
        default=0.3,
        help="IoU threshold for DelineateAnything NMS (default: 0.3)"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory {args.data_dir} does not exist")
        sys.exit(1)
    
    # Expand 'all' countries to full list
    if args.countries and len(args.countries) == 1 and args.countries[0].lower() == "all":
        args.countries = list(ALL_COUNTRIES)
        print(f"Expanded countries to ALL ({len(args.countries)}): {args.countries}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set default output name if not provided
    if args.output_name is None:
        suffix = "all" if len(args.countries) > 10 else "-".join(args.countries)
        args.output_name = f"{args.model}_detections_{suffix}.pkl"
    
    # Prepare base arguments (common to all models)
    runner_args = {
        "data_dir": args.data_dir,
        "output_dir": args.output_dir,
        "countries": args.countries,
        "split": args.split,
        "batch_size": args.batch_size,
        "save_logits": args.save_logits,
        "min_area": args.min_area,
        "confidence_threshold": args.confidence_threshold,
    }
    
    # Add model weights and config (if provided)
    if args.model_weights:
        runner_args["model_weights"] = args.model_weights
    if args.config_file:
        runner_args["config_file"] = args.config_file
    
    # Add model-specific arguments based on selected model
    if args.model in ["ftw", "ftw-cli"]:
        runner_args["temporal_options"] = args.temporal_options
    elif args.model == "sam":
        runner_args["sam_model_type"] = args.sam_model_type
        runner_args["in_chans"] = getattr(args, "sam_in_chans", 8)  # Default to 8 for stacked Sentinel-2
    elif args.model in ["delineate_anything", "da"]:
        runner_args["delany_model_variant"] = args.delany_model_variant
        runner_args["patch_size"] = args.patch_size
        runner_args["resize_factor"] = args.resize_factor
        runner_args["max_detections"] = args.max_detections
        runner_args["iou_threshold"] = args.iou_threshold
    # DECODE uses config_file (already added above)
    
    # DelineateAnything can use built-in registry if no weights provided
    if not args.model_weights and args.model not in ["delineate_anything", "da"]:
        print(f"Error: --model_weights is required for {args.model}")
        sys.exit(1)
    
    # For DelineateAnything, set default weights path if not provided
    if args.model in ["delineate_anything", "da"] and not args.model_weights:
        # Use model variant to determine which checkpoint to use
        if args.delany_model_variant == "DelineateAnything-S":
            args.model_weights = "DelineateAnything-S"  # Will use registry
        else:
            args.model_weights = "DelineateAnything"  # Will use registry
        print(f"Using DelineateAnything built-in checkpoint: {args.model_weights}")
        runner_args["model_weights"] = args.model_weights
    
    try:
        print(f"Starting inference with {args.model} model...")
        start_time = time.time()
        
        # Try registry-based approach if requested and available
        if args.use_registry and REGISTRY_AVAILABLE and args.model in available_models():
            print(f"Using registry-based segmenter for {args.model}")
            result = run_registry_inference(
                model_name=args.model,
                **runner_args
            )
        else:
            # ⚠️  DEPRECATED: Legacy MODEL_RUNNERS approach
            # This path will be removed once registry-based inference is fully verified.
            # Use `--use_registry` flag to use the new registry-based path.
            import warnings
            warnings.warn(
                f"Using deprecated legacy inference path for {args.model}. "
                "Use --use_registry flag to use the new registry-based segmenters. "
                "Legacy path will be removed in a future version.",
                DeprecationWarning,
                stacklevel=2
            )
            runner = MODEL_RUNNERS.get(args.model)
            if runner is None:
                raise ValueError(f"Model {args.model} not found in MODEL_RUNNERS. Available: {list(MODEL_RUNNERS.keys())}")
            result = runner(**runner_args)
        
        # Saving logic
        if args.model == "sam" and isinstance(result, dict):
            # Aggregate all SAM detections into a single list and save once (baseline-style)
            # IMPORTANT: preserve dataset evaluation order (countries then images) so metrics align
            ordered_countries = None
            try:
                # Use canonical FTW country ordering if available (imported at module scope)
                if isinstance(getattr(args, "countries", None), list) and args.countries and args.countries[0] != "all":
                    # Respect user-specified order
                    ordered_countries = [c for c in args.countries if c in result]
                else:
                    ordered_countries = [c for c in ALL_COUNTRIES if c in result]
            except Exception:
                # Fallback to user-specified countries order, else dict key order
                if isinstance(getattr(args, "countries", None), list) and args.countries:
                    if args.countries[0] == "all":
                        ordered_countries = list(result.keys())
                    else:
                        ordered_countries = [c for c in args.countries if c in result]
                else:
                    ordered_countries = list(result.keys())

            detections_list = []
            for country in ordered_countries:
                detections_list.extend(result.get(country, []))
            logits_list = None
            output_path = os.path.join(args.output_dir, args.output_name)
            with open(output_path, 'wb') as f:
                pickle.dump(detections_list, f)
            print(f"Detections saved to: {output_path}")
        else:
            # Fallback to original saving behavior for non-SAM models
            if isinstance(result, tuple):
                detections_list, logits_list = result
            else:
                detections_list = result
                logits_list = None
            
            output_path = os.path.join(args.output_dir, args.output_name)
            with open(output_path, 'wb') as f:
                pickle.dump(detections_list, f)
                print(f"Detections saved to: {output_path}")

        inference_time = time.time() - start_time
        print(f"\nInference completed in {inference_time:.2f} seconds")
        print(f"Total detections lists: {len(detections_list) if isinstance(detections_list, list) else 'N/A'}")
        
        # Save logits if available in non-SAM path
        if 'logits_list' in locals() and logits_list is not None:
            logits_path = output_path.replace('.pkl', '_logits.pkl')
            with open(logits_path, 'wb') as f:
                pickle.dump(logits_list, f)
            print(f"Logits saved to: {logits_path}")
        
        # Optionally write metadata as before (skip for brevity)
        
    except Exception as e:
        print(f"Error running inference: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
