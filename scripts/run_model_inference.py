#!/usr/bin/env python3
"""
Script to run inference on different models using registry-based segmenters.

All models are now accessed through the unified model registry, which provides
a consistent interface across FTW, SAM, DECODE, and DelineateAnything models.
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
except ImportError as e:
    REGISTRY_AVAILABLE = False
    print(f"ERROR: Model registry not available: {e}")
    print("Please ensure the project is properly installed with registry support.")


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
        # Match FTW dataset structure:
        # .../country/s2_images/window_a/aoi_id.tif
        # .../country/s2_images/window_b/aoi_id.tif
        # .../country/label_masks/instance/aoi_id.tif
        # .../country/label_masks/semantic_2class/aoi_id.tif
        # .../country/label_masks/semantic_3class/aoi_id.tif
        match = re.search(r'/([^/]+)/(?:s2_images/(?:window_a|window_b)|label_masks/(?:instance|semantic_2class|semantic_3class))/([^/]+)\.tif$', path)
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
    elif model_name in ["delineate_anything", "da"]:
        # Load DA config to get model-specific defaults
        import yaml
        config_file = kwargs.get("config_file")
        if not config_file:
            # Use default config
            config_file = Path(__file__).parent.parent / "configs" / "delineate_anything" / "default.yaml"
        
        # Load config (either user-provided or default)
        if Path(config_file).exists():
            with open(config_file, "r") as f:
                config = yaml.safe_load(f)
                # Extract inference parameters from config (only if not provided by user)
                if "inference" in config:
                    inference_config = config["inference"]
                    
                    # Apply config values only if user didn't provide them (CLI > config > defaults)
                    if kwargs.get("confidence_threshold") is None and "confidence_threshold" in inference_config:
                        kwargs["confidence_threshold"] = inference_config["confidence_threshold"]
                        print(f"Using confidence_threshold from config: {kwargs['confidence_threshold']}")
                    
                    if kwargs.get("iou_threshold") is None and "iou_threshold" in inference_config:
                        kwargs["iou_threshold"] = inference_config["iou_threshold"]
                        print(f"Using iou_threshold from config: {kwargs['iou_threshold']}")
                    
                    if kwargs.get("max_detections") is None and "max_detections" in inference_config:
                        kwargs["max_detections"] = inference_config["max_detections"]
                        print(f"Using max_detections from config: {kwargs['max_detections']}")
                    
                    if kwargs.get("patch_size") is None and "patch_size" in inference_config:
                        kwargs["patch_size"] = inference_config["patch_size"]
                        print(f"Using patch_size from config: {kwargs['patch_size']}")
                    
                    if kwargs.get("resize_factor") is None and "resize_factor" in inference_config:
                        kwargs["resize_factor"] = inference_config["resize_factor"]
                        print(f"Using resize_factor from config: {kwargs['resize_factor']}")
        else:
            print(f"Warning: Config file not found: {config_file}")
        
        # Apply hardcoded defaults for any remaining None values
        if kwargs.get("patch_size") is None:
            kwargs["patch_size"] = 256
        if kwargs.get("resize_factor") is None:
            kwargs["resize_factor"] = 2
        if kwargs.get("max_detections") is None:
            kwargs["max_detections"] = 100
        if kwargs.get("iou_threshold") is None:
            kwargs["iou_threshold"] = 0.3
    
    # Create segmenter
    segmenter = create_segmenter(model_name, **segmenter_kwargs)
    
    # Load dataset (same approach as legacy runners)
    from ftw_tools.training.datasets import FTW
    from ftw_tools.training.datamodules import preprocess
    
    countries = kwargs.get("countries", ["belgium"])
    if isinstance(countries, str):
        countries = [countries]
    countries_sorted = sorted(countries)
    
    # Determine temporal_options based on model
    # FTW/SAM/DECODE need stacked (8 channels), DA needs single window (4 channels)
    if "temporal_options" in kwargs and kwargs["temporal_options"] is not None:
        temporal_options = kwargs["temporal_options"]
        print(f"Using user-specified temporal_options: {temporal_options}")
    else:
        # Auto-select based on model
        if model_name in ["ftw", "sam", "decode"]:
            temporal_options = "stacked"  # 8 channels (windowB + windowA)
        elif model_name in ["delineate_anything", "da"]:
            temporal_options = "windowA"  # 4 channels (RGBN), extract RGB
        else:
            temporal_options = "stacked"  # Default
        print(f"Auto-selected temporal_options: {temporal_options} (model: {model_name})")
    
    # Create dataset
    # Note: SAM and DA handle their own preprocessing/normalization internally
    use_transforms = None if model_name in ["sam", "delineate_anything", "da"] else preprocess
    ds = FTW(
        root=data_dir,
        countries=countries_sorted,
        split=kwargs.get("split", "test"),
        transforms=use_transforms,
        load_boundaries=False,
        temporal_options=temporal_options,
    )
    
    # Sort filenames
    _sort_dataset_filenames(ds)
    
    # Create dataloader
    batch_size = kwargs.get("batch_size", 16)
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)
    
    detections_list = []
    logits_list = [] if kwargs.get("save_logits", False) else None
    
    # Set default confidence threshold if not set (after config loading)
    if kwargs.get("confidence_threshold") is None:
        kwargs["confidence_threshold"] = 0.5  # Default for non-DA models
    
    # Get filenames from dataset for tracking
    filenames = None
    if hasattr(ds, 'filenames') and ds.filenames:
        filenames = ds.filenames
    elif hasattr(ds, 'file_list') and ds.file_list:
        filenames = ds.file_list
    
    # Run inference
    sample_idx = 0
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
                        score_threshold=kwargs["confidence_threshold"]
                    )
                else:
                    raise ValueError(f"Unexpected output type: {type(output)}")
                
                # Store image filename if available
                if filenames is not None and sample_idx < len(filenames):
                    filename_dict = filenames[sample_idx]
                    # Extract filename based on temporal_options to match what was actually used
                    if temporal_options == "windowA":
                        image_path = filename_dict.get("window_a") or ""
                    elif temporal_options == "windowB":
                        image_path = filename_dict.get("window_b") or ""
                    elif temporal_options == "stacked":
                        # For stacked, prefer windowB as primary
                        image_path = filename_dict.get("window_b") or filename_dict.get("window_a") or ""
                    else:
                        # Fallback: try various keys
                        image_path = (filename_dict.get("window_b") or 
                                      filename_dict.get("window_a") or 
                                      filename_dict.get("mask") or 
                                      filename_dict.get("image_b") or
                                      filename_dict.get("image_a") or
                                      "")
                    detections.image_filename = image_path
                
                detections_list.append(detections)
                sample_idx += 1
    
    if logits_list is not None:
        return detections_list, logits_list
    return detections_list


def main():
    parser = argparse.ArgumentParser(
        description="Run model inference using registry-based segmenters and convert to unified Detections format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments - use only registry models
    if REGISTRY_AVAILABLE:
        model_choices = list(available_models().keys())
    else:
        raise RuntimeError("Model registry not available - please ensure the registry module is properly installed")
    
    parser.add_argument(
        "--model", 
        type=str, 
        required=True,
        choices=model_choices,
        help="Model type to run inference with"
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
    parser.add_argument(
        "--temporal_options",
        type=str,
        default=None,  # Auto-select based on model
        choices=["stacked", "windowA", "windowB", "rgb", "median", "random_window"],
        help=(
            "Temporal processing option for multi-temporal data. "
            "Default: 'stacked' for FTW/SAM/DECODE (8 channels), 'windowA' for DA (4 channels). "
            "Options: stacked (windowB+windowA), windowA (4ch), windowB (4ch), rgb (6ch)"
        )
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
        default=None,
        help="Confidence threshold for filtering detections (default: 0.5 for most models, 0.05 for DelineateAnything from config)"
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
    # temporal_options is now a general argument above
    
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
        default=None,
        help="Patch size for DelineateAnything (default: from config or 256)"
    )
    delany_group.add_argument(
        "--resize_factor",
        type=int,
        default=None,
        help="Resize factor for DelineateAnything (default: from config or 2)"
    )
    delany_group.add_argument(
        "--max_detections",
        type=int,
        default=None,
        help="Max detections per image for DelineateAnything (default: from config or 100)"
    )
    delany_group.add_argument(
        "--iou_threshold",
        type=float,
        default=None,
        help="IoU threshold for DelineateAnything NMS (default: from config or 0.3)"
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
        runner_args["temporal_options"] = args.temporal_options  # Add temporal_options for DA
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
        
        # Use registry-based segmenter
        if not REGISTRY_AVAILABLE:
            raise RuntimeError("Model registry not available")
        if args.model not in available_models():
            raise ValueError(f"Model {args.model} not found in registry. Available: {list(available_models().keys())}")
        
        print(f"Using registry-based segmenter for {args.model}")
        result = run_registry_inference(
            model_name=args.model,
            **runner_args
        )
        
        # Save results
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
