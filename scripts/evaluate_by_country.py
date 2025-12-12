#!/usr/bin/env python3
"""
Evaluate model detections against ground truth with country-level breakdown.
"""

import argparse
import json
import os
import pickle
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from detections import Detections
from evaluator import Evaluator


def load_detections_from_file(file_path: str) -> List[Detections]:
    """Load detections from pickle file."""
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def load_gt_instance_masks_from_dataset(
    data_dir: str, 
    countries: List[str], 
    split: str = "test"
) -> Tuple[List[np.ndarray], List[str], List[Tuple[str, str]], List[int]]:
    """
    Load ground truth instance masks from FTW dataset for COCO metrics.
    
    Uses the SAME filtering and ordering logic as generate_ground_truth.py to ensure
    perfect alignment.
    
    Args:
        data_dir: Path to FTW dataset root
        countries: List of countries to process
        split: Dataset split (test/val/train)
        
    Returns:
        Tuple of (instance_masks_list, country_list, country_aoi_list, image_ids_list)
    """
    import geopandas as gpd
    import rasterio
    from pathlib import Path
    
    instance_masks = []
    countries_list = []
    country_aoi_list = []
    image_ids = []
    
    # Sort countries to ensure deterministic ordering
    countries_sorted = sorted(countries)
    
    for country in countries_sorted:
        print(f"Loading GT instance masks for {country}...")
        
        chips_file = os.path.join(data_dir, country, f"chips_{country}.parquet")
        if not os.path.exists(chips_file):
            print(f"Warning: Chips file not found for {country}")
            continue
            
        chips_gdf = gpd.read_parquet(chips_file)
        split_chips = chips_gdf[chips_gdf["split"] == split]
        aoi_ids = sorted(split_chips["aoi_id"].tolist())
        
        # Apply SAME filtering logic as FTW dataset class
        valid_aoi_ids = []
        for aoi_id in aoi_ids:
            window_b_fn = Path(os.path.join(data_dir, country, "s2_images/window_b", f"{aoi_id}.tif"))
            window_a_fn = Path(os.path.join(data_dir, country, "s2_images/window_a", f"{aoi_id}.tif"))
            masks_2c_fn = Path(os.path.join(data_dir, country, "label_masks/semantic_2class", f"{aoi_id}.tif"))
            masks_3c_fn = Path(os.path.join(data_dir, country, "label_masks/semantic_3class", f"{aoi_id}.tif"))
            
            if not (window_b_fn.exists() and window_a_fn.exists() and masks_2c_fn.exists() and masks_3c_fn.exists()):
                continue
            
            semantic_2_file = os.path.join(data_dir, country, "label_masks", "semantic_2class", f"{aoi_id}.tif")
            semantic_3_file = os.path.join(data_dir, country, "label_masks", "semantic_3class", f"{aoi_id}.tif")
            if not (os.path.exists(semantic_2_file) or os.path.exists(semantic_3_file)):
                continue
            
            valid_aoi_ids.append(aoi_id)
        
        # Process valid AOI IDs
        for aoi_id in valid_aoi_ids:
            instance_mask_path = os.path.join(data_dir, country, "label_masks", "instance", f"{aoi_id}.tif")
            
            if os.path.exists(instance_mask_path):
                try:
                    with rasterio.open(instance_mask_path) as src:
                        instance_mask = src.read(1)
                    instance_masks.append(instance_mask)
                    countries_list.append(country)
                    aoi_id_str = str(aoi_id)
                    country_aoi_list.append((country, aoi_id_str))
                    image_ids.append(len(instance_masks) - 1)
                except Exception as e:
                    print(f"Error loading instance mask for {country}/{aoi_id}: {str(e)}")
                    # Add empty mask to maintain alignment
                    # Get shape from semantic mask
                    semantic_2_file = os.path.join(data_dir, country, "label_masks", "semantic_2class", f"{aoi_id}.tif")
                    semantic_3_file = os.path.join(data_dir, country, "label_masks", "semantic_3class", f"{aoi_id}.tif")
                    try:
                        if os.path.exists(semantic_2_file):
                            with rasterio.open(semantic_2_file) as src:
                                mask_shape = src.shape
                        elif os.path.exists(semantic_3_file):
                            with rasterio.open(semantic_3_file) as src:
                                mask_shape = src.shape
                        else:
                            mask_shape = (256, 256)
                    except:
                        mask_shape = (256, 256)
                    instance_masks.append(np.zeros(mask_shape, dtype=np.uint8))
                    countries_list.append(country)
                    aoi_id_str = str(aoi_id)
                    country_aoi_list.append((country, aoi_id_str))
                    image_ids.append(len(instance_masks) - 1)
            else:
                # Instance mask missing - add empty mask to maintain alignment
                semantic_2_file = os.path.join(data_dir, country, "label_masks", "semantic_2class", f"{aoi_id}.tif")
                semantic_3_file = os.path.join(data_dir, country, "label_masks", "semantic_3class", f"{aoi_id}.tif")
                try:
                    if os.path.exists(semantic_2_file):
                        with rasterio.open(semantic_2_file) as src:
                            mask_shape = src.shape
                    elif os.path.exists(semantic_3_file):
                        with rasterio.open(semantic_3_file) as src:
                            mask_shape = src.shape
                    else:
                        mask_shape = (256, 256)
                except:
                    mask_shape = (256, 256)
                instance_masks.append(np.zeros(mask_shape, dtype=np.uint8))
                countries_list.append(country)
                aoi_id_str = str(aoi_id)
                country_aoi_list.append((country, aoi_id_str))
                image_ids.append(len(instance_masks) - 1)
    
    print(f"Loaded {len(instance_masks)} ground truth instance masks")
    return instance_masks, countries_list, country_aoi_list, image_ids


def build_country_aoi_mapping(
    data_dir: str, 
    countries: List[str], 
    split: str = "test"
) -> Dict[Tuple[str, str], int]:
    """
    Build a mapping from (country, aoi_id) to list index.
    
    Uses the SAME filtering and ordering logic as the FTW dataset class
    to ensure consistent mapping.
    
    Args:
        data_dir: Path to FTW dataset root
        countries: List of countries to process
        split: Dataset split (test/val/train)
        
    Returns:
        Dictionary mapping (country, aoi_id) tuples to list indices
    """
    import geopandas as gpd
    from pathlib import Path
    
    mapping = {}
    index = 0
    
    # Sort countries to ensure deterministic ordering (matches generate_ground_truth.py)
    countries_sorted = sorted(countries)
    
    for country in countries_sorted:
        # Load chips file to get AOI IDs for this split
        chips_file = os.path.join(data_dir, country, f"chips_{country}.parquet")
        if not os.path.exists(chips_file):
            print(f"Warning: Chips file not found for {country}")
            continue
            
        chips_gdf = gpd.read_parquet(chips_file)
        split_chips = chips_gdf[chips_gdf["split"] == split]
        # Sort AOI IDs to ensure deterministic ordering (matches FTW dataset class)
        aoi_ids = sorted(split_chips["aoi_id"].tolist())
        
        # Apply SAME filtering logic as FTW dataset class
        for aoi_id in aoi_ids:
            # Check for all required files (same as FTW dataset class)
            window_b_fn = Path(os.path.join(data_dir, country, "s2_images/window_b", f"{aoi_id}.tif"))
            window_a_fn = Path(os.path.join(data_dir, country, "s2_images/window_a", f"{aoi_id}.tif"))
            masks_2c_fn = Path(os.path.join(data_dir, country, "label_masks/semantic_2class", f"{aoi_id}.tif"))
            masks_3c_fn = Path(os.path.join(data_dir, country, "label_masks/semantic_3class", f"{aoi_id}.tif"))
            
            # Skip the image AOI's which does not have all four corresponding files
            if not (
                window_b_fn.exists()
                and window_a_fn.exists()
                and masks_2c_fn.exists()
                and masks_3c_fn.exists()
            ):
                continue
            
            # Also check semantic mask existence (same as FTW dataset class)
            # Prefer 2-class mask, fall back to 3-class if available
            semantic_2_file = os.path.join(data_dir, country, "label_masks", "semantic_2class", f"{aoi_id}.tif")
            semantic_3_file = os.path.join(data_dir, country, "label_masks", "semantic_3class", f"{aoi_id}.tif")
            if not (os.path.exists(semantic_2_file) or os.path.exists(semantic_3_file)):
                continue
            
            # Add to mapping
            # Normalize aoi_id to string for consistency
            aoi_id_str = str(aoi_id)
            mapping[(country, aoi_id_str)] = index
            index += 1
    
    return mapping


def load_gt_masks_from_dataset(
    data_dir: str, 
    countries: List[str], 
    split: str = "test"
) -> Tuple[List[np.ndarray], List[str], List[Tuple[str, str]], List[int]]:
    """
    Load ground truth binary masks from FTW dataset for pixel-level metrics.
    
    Uses the SAME filtering and ordering logic as the FTW dataset class to ensure
    perfect alignment.
    
    Args:
        data_dir: Path to FTW dataset root
        countries: List of countries to process
        split: Dataset split (test/val/train)
        
    Returns:
        Tuple of (masks_list, country_list, country_aoi_list, image_ids_list)
        where country_aoi_list contains (country, aoi_id) tuples for each mask
    """
    import geopandas as gpd
    import rasterio
    from pathlib import Path
    
    masks = []
    countries_list = []
    country_aoi_list = []  # List of (country, aoi_id) tuples
    image_ids = []
    
    # Sort countries to ensure deterministic ordering (matches generate_ground_truth.py)
    countries_sorted = sorted(countries)
    
    for country in countries_sorted:
        print(f"Loading GT masks for {country}...")
        
        # Load chips file to get AOI IDs for this split
        chips_file = os.path.join(data_dir, country, f"chips_{country}.parquet")
        if not os.path.exists(chips_file):
            print(f"Warning: Chips file not found for {country}")
            continue
            
        chips_gdf = gpd.read_parquet(chips_file)
        split_chips = chips_gdf[chips_gdf["split"] == split]
        # Sort AOI IDs to ensure deterministic ordering (matches FTW dataset class)
        aoi_ids = sorted(split_chips["aoi_id"].tolist())
        
        print(f"Found {len(aoi_ids)} AOIs for {country} {split}")
        
        # Apply SAME filtering logic as FTW dataset class
        valid_aoi_ids = []
        for aoi_id in aoi_ids:
            # Check for all required files (same as FTW dataset class)
            window_b_fn = Path(os.path.join(data_dir, country, "s2_images/window_b", f"{aoi_id}.tif"))
            window_a_fn = Path(os.path.join(data_dir, country, "s2_images/window_a", f"{aoi_id}.tif"))
            masks_2c_fn = Path(os.path.join(data_dir, country, "label_masks/semantic_2class", f"{aoi_id}.tif"))
            masks_3c_fn = Path(os.path.join(data_dir, country, "label_masks/semantic_3class", f"{aoi_id}.tif"))
            
            # Skip the image AOI's which does not have all four corresponding files
            # (same logic as generate_ground_truth.py)
            if not (
                window_b_fn.exists()
                and window_a_fn.exists()
                and masks_2c_fn.exists()
                and masks_3c_fn.exists()
            ):
                continue
            
            # Also check semantic mask existence (same as FTW dataset class)
            # Prefer 2-class mask, fall back to 3-class if available
            semantic_2_file = os.path.join(data_dir, country, "label_masks", "semantic_2class", f"{aoi_id}.tif")
            semantic_3_file = os.path.join(data_dir, country, "label_masks", "semantic_3class", f"{aoi_id}.tif")
            
            if os.path.exists(semantic_2_file):
                mask_fn = semantic_2_file
                mask_type = "2class"
            elif os.path.exists(semantic_3_file):
                mask_fn = semantic_3_file
                mask_type = "3class"
            else:
                # This should not happen due to filtering above, but handle gracefully
                print(f"Warning: Semantic mask not found for {country}/{aoi_id} (this should not happen)")
                continue
            
            valid_aoi_ids.append(aoi_id)
        
        print(f"  After filtering: {len(valid_aoi_ids)} valid AOIs (skipped {len(aoi_ids) - len(valid_aoi_ids)})")
        
        # Process valid AOI IDs in sorted order (matches FTW dataset class)
        # NOTE: We process ALL valid_aoi_ids (those that pass filtering)
        # to maintain alignment, even if instance masks don't exist
        for aoi_id in valid_aoi_ids:
            # Use same mask selection logic as FTW dataset class
            # Prefer 2-class mask, fall back to 3-class if available
            semantic_2_file = os.path.join(data_dir, country, "label_masks", "semantic_2class", f"{aoi_id}.tif")
            semantic_3_file = os.path.join(data_dir, country, "label_masks", "semantic_3class", f"{aoi_id}.tif")
            
            if os.path.exists(semantic_2_file):
                mask_path = semantic_2_file
                mask_type = "2class"
            elif os.path.exists(semantic_3_file):
                mask_path = semantic_3_file
                mask_type = "3class"
            else:
                # This should not happen due to filtering above, but handle gracefully
                print(f"Warning: Semantic mask not found for {country}/{aoi_id} (this should not happen)")
                continue
            
            try:
                # Load mask
                with rasterio.open(mask_path) as src:
                    mask = src.read(1)
                    
                    # Convert to binary field mask
                    if mask_type == "3class":
                        # Class 1 = field, Class 2 = boundary, Class 0 = background
                        binary_mask = (mask == 1).astype(np.uint8)
                    else:  # 2class
                        # Class 1 = field, Class 0 = background
                        binary_mask = (mask == 1).astype(np.uint8)
                    
                    masks.append(binary_mask)
                    countries_list.append(country)
                    # Normalize aoi_id to string for consistency
                    aoi_id_str = str(aoi_id)
                    country_aoi_list.append((country, aoi_id_str))
                    image_ids.append(len(masks) - 1)  # Use index as image_id
            except Exception as e:
                print(f"Error loading mask for {country}/{aoi_id}: {str(e)}")
                # Skip this mask to maintain alignment - this should be rare
                continue
    
    print(f"Loaded {len(masks)} ground truth masks")
    return masks, countries_list, country_aoi_list, image_ids


def match_predictions_by_country_aoi(
    pred_detections: List[Detections],
    pred_mapping: Dict[Tuple[str, str], int],
    country_aoi_list: List[Tuple[str, str]]
) -> List[Detections]:
    """
    Match prediction detections by (country, aoi_id) to canonical ordering.
    
    Args:
        pred_detections: List of prediction detections (indexed by pred_mapping)
        pred_mapping: Mapping from (country, aoi_id) to prediction detection index
        country_aoi_list: List of (country, aoi_id) tuples for canonical ordering
        
    Returns:
        List of matched prediction detections in same order as country_aoi_list
    """
    matched_pred = []
    missing_pred = 0
    
    for country, aoi_id in country_aoi_list:
        key = (country, aoi_id)
        
        # Get prediction detection
        if key in pred_mapping:
            pred_idx = pred_mapping[key]
            if pred_idx < len(pred_detections):
                matched_pred.append(pred_detections[pred_idx])
            else:
                print(f"Warning: Pred index {pred_idx} out of range for {country}/{aoi_id}")
                matched_pred.append(Detections(xyxy=np.empty((0, 4))))
                missing_pred += 1
        else:
            # Prediction not found - use empty detections
            matched_pred.append(Detections(xyxy=np.empty((0, 4))))
            missing_pred += 1
    
    if missing_pred > 0:
        print(f"Warning: {missing_pred} predictions not found in mapping")
    
    return matched_pred


def evaluate_by_country(
    model_detections: Dict[str, List[Detections]],
    gt_masks: List[np.ndarray],
    gt_instance_masks: Optional[List[np.ndarray]],
    countries_list: List[str],
    country_aoi_list: List[Tuple[str, str]],
    image_ids: List[int],
    iou_threshold: float = 0.5,
    metrics: List[str] = ["object"]
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Evaluate models with country-level breakdown using GT masks directly.
    
    Matches FTW baseline methodology: uses semantic masks for object metrics
    (extracts connected components, same as ftw_tools.training.metrics).
    
    Args:
        model_detections: Dictionary mapping model names to their detection lists (already matched)
        gt_masks: List of ground truth semantic masks (for pixel/object metrics)
        gt_instance_masks: Optional list of GT instance masks (for COCO metrics only)
        countries_list: List of country names for each sample
        country_aoi_list: List of (country, aoi_id) tuples for each sample
        image_ids: List of image IDs
        iou_threshold: IoU threshold for matching
        metrics: List of metrics to compute
        
    Returns:
        Dictionary with structure: {country: {model: {metric: value}}}
    """
    # Get unique countries
    unique_countries = list(set(countries_list))
    print(f"Evaluating across {len(unique_countries)} countries: {unique_countries}")
    
    # Create country-level results
    country_results = {}
    
    for country in unique_countries:
        print(f"\nEvaluating {country}...")
        
        # Get indices for this country
        country_indices = [i for i, c in enumerate(countries_list) if c == country]
        print(f"  Found {len(country_indices)} samples for {country}")
        
        if len(country_indices) == 0:
            print(f"  Warning: No samples found for {country}")
            continue
        
        # Extract masks and detections for this country
        country_gt_masks = [gt_masks[i] for i in country_indices]
        country_image_ids = [image_ids[i] for i in country_indices]
        
        # For COCO metrics, need GT instance masks converted to Detections
        # For pixel/object metrics with semantic mask mode, GT Detections not used (gt_masks used instead)
        country_gt_detections = []
        if "coco" in metrics and gt_instance_masks is not None:
            country_gt_instance_masks = [gt_instance_masks[i] for i in country_indices]
            # Convert GT instance masks to Detections for COCO format
            for instance_mask in country_gt_instance_masks:
                gt_dets = Detections.from_gt(instance_mask, min_area=0)
                country_gt_detections.append(gt_dets)
        else:
            # Create empty detections (not used when use_semantic_masks_for_object_metrics=True and coco not in metrics)
            country_gt_detections = [Detections(xyxy=np.empty((0, 4))) for _ in country_indices]
        
        country_model_detections = {}
        for model_name, model_det_list in model_detections.items():
            country_model_detections[model_name] = [model_det_list[i] for i in country_indices]
        
        # Create evaluator for this country
        # Use semantic masks for object metrics to match FTW baseline (ftw_tools.training.metrics)
        # This extracts connected components from semantic masks, same as get_object_level_metrics in metrics.py
        evaluator = Evaluator(
            iou_threshold=iou_threshold,
            metrics=metrics,
            gt_masks=country_gt_masks,
            image_ids=country_image_ids,
            use_semantic_masks_for_object_metrics=True  # Match FTW baseline: extract connected components from semantic masks
        )
        
        # Evaluate each model for this country
        country_model_results = {}
        for model_name, pred_detections in country_model_detections.items():
            print(f"  Evaluating {model_name} on {country}...")
            
            # When use_semantic_masks_for_object_metrics=True:
            # - Pixel metrics: uses gt_masks directly (y_true not used)
            # - Object metrics: uses gt_masks directly, extracts connected components (y_true not used)
            # - COCO metrics: needs y_true Detections (from GT instance masks)
            # So we pass country_gt_detections which are either from instance masks (for COCO) or empty (for pixel/object)
            results = evaluator.evaluate(y_true=country_gt_detections, y_pred=pred_detections)
            country_model_results[model_name] = results
        
        country_results[country] = country_model_results
    
    return country_results


def convert_to_json_serializable(obj: Any) -> Any:
    """
    Convert numpy types to JSON-serializable Python types.
    
    Args:
        obj: Object to convert
        
    Returns:
        JSON-serializable version of obj
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_json_serializable(item) for item in obj)
    else:
        return obj


def save_country_results(
    country_results: Dict[str, Dict[str, Dict[str, Any]]],
    output_dir: str,
    output_name: str = "country_evaluation_results.json",
    save_csv: bool = False
):
    """Save country-level evaluation results to JSON and optionally CSV."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to JSON-serializable format
    serializable_results = convert_to_json_serializable(country_results)
    
    # Save JSON
    json_path = os.path.join(output_dir, output_name)
    with open(json_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    print(f"\nResults saved to: {json_path}")
    
    # Save CSV if requested
    if save_csv:
        import csv
        csv_path = os.path.join(output_dir, output_name.replace('.json', '.csv'))
        
        # Get all countries and models
        countries = sorted(serializable_results.keys())
        if not countries:
            print("No results to save to CSV")
            return
        
        models = sorted(serializable_results[countries[0]].keys())
        if not models:
            print("No model results to save to CSV")
            return
        
        # Write CSV
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            header = ["country", "model"]
            # Add all metric keys from first result
            if countries and models:
                first_result = serializable_results[countries[0]][models[0]]
                metric_keys = sorted(first_result.keys())
                header.extend(metric_keys)
                writer.writerow(header)
                
                # Write data rows
                for country in countries:
                    for model in models:
                        if model in serializable_results[country]:
                            row = [country, model]
                            results = serializable_results[country][model]
                            row.extend([results.get(key, 0) for key in metric_keys])
                            writer.writerow(row)
        
        print(f"CSV results saved to: {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model detections against ground truth with country-level breakdown",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "--model_detections", 
        type=str, 
        required=True,
        help="Path to model detections JSON file or JSON string mapping model names to detection files"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True,
        help="Directory to save evaluation results"
    )
    
    # Evaluation parameters
    parser.add_argument(
        "--iou_threshold", 
        type=float, 
        default=0.5,
        help="IoU threshold for matching detections"
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        choices=["pixel", "object", "coco", "all"],
        default=["object"],
        help="Metrics to compute: pixel (pixel-level), object (object-level), coco (COCO-style), or all"
    )
    
    # Ground truth data - loaded directly from dataset (no GT Detections pickle needed)
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to FTW dataset root directory (needed for loading GT masks and building country/AOI mappings)"
    )
    parser.add_argument(
        "--countries",
        type=str,
        nargs="+",
        default=["all"],
        help="List of countries to evaluate (or 'all' for all countries)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate"
    )
    # Output options
    parser.add_argument(
        "--output_name",
        type=str,
        default="country_evaluation_results.json",
        help="Name of the output file"
    )
    parser.add_argument(
        "--save_csv",
        action="store_true",
        help="Also save results as CSV file"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output"
    )
    
    args = parser.parse_args()
    
    # Expand "all" countries
    if args.countries and len(args.countries) == 1 and args.countries[0].lower() == "all":
        # Import ALL_COUNTRIES from ftw_tools.settings
        ftw_baselines_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'models', 'ftw')
        if ftw_baselines_path not in sys.path:
            sys.path.insert(0, ftw_baselines_path)
        from ftw_tools.settings import ALL_COUNTRIES
        args.countries = list(ALL_COUNTRIES)
        print(f"Expanded countries to ALL ({len(args.countries)}): {args.countries}")
    
    # Expand metrics
    if "all" in args.metrics:
        args.metrics = ["pixel", "object", "coco"]
    
    # Build mapping for consistent ordering (same logic as FTW dataset class)
    print("Building country/AOI mapping...")
    mapping = build_country_aoi_mapping(
        args.data_dir, args.countries, args.split
    )
    print(f"  Built mapping for {len(mapping)} samples")
    
    # Load ground truth semantic masks from dataset (for pixel/object metrics)
    # This matches FTW baseline: uses semantic masks directly, no Detections needed
    print("Loading ground truth semantic masks from dataset...")
    print("  (Object metrics will use semantic masks with connected components, matching FTW baseline)")
    gt_masks, countries_list, country_aoi_list, image_ids = load_gt_masks_from_dataset(
        args.data_dir, args.countries, args.split
    )
    
    # Load ground truth instance masks for COCO metrics if needed
    gt_instance_masks = None
    if "coco" in args.metrics:
        print("Loading ground truth instance masks from dataset for COCO metrics...")
        gt_instance_masks, _, _, _ = load_gt_instance_masks_from_dataset(
            args.data_dir, args.countries, args.split
        )
        if len(gt_instance_masks) != len(gt_masks):
            print(f"Warning: GT instance masks ({len(gt_instance_masks)}) != GT semantic masks ({len(gt_masks)})")
            # Align by truncating to shorter length
            min_len = min(len(gt_instance_masks), len(gt_masks))
            gt_instance_masks = gt_instance_masks[:min_len]
            gt_masks = gt_masks[:min_len]
            countries_list = countries_list[:min_len]
            country_aoi_list = country_aoi_list[:min_len]
            image_ids = image_ids[:min_len]
    
    # Parse model detections
    print("Loading model detections...")
    if args.model_detections.endswith('.json'):
        with open(args.model_detections, 'r') as f:
            model_detections_paths = json.load(f)
    else:
        # Try to parse as JSON string
        try:
            model_detections_paths = json.loads(args.model_detections)
        except json.JSONDecodeError:
            print("Error: model_detections must be a valid JSON string or path to JSON file")
            print(f"Received: {args.model_detections}")
            sys.exit(1)
    
    # Load model detections and match to canonical ordering
    model_detections = {}
    for model_name, detections_path in model_detections_paths.items():
        print(f"Loading detections for {model_name}...")
        pred_detections = load_detections_from_file(detections_path)
        
        # Build mapping for predictions (assumes they were generated with same ordering logic)
        print(f"  Building mapping for {model_name}...")
        pred_mapping = build_country_aoi_mapping(
            args.data_dir, args.countries, args.split
        )
        print(f"  Built mapping for {len(pred_mapping)} predictions")
        
        # Match predictions to canonical ordering (same as GT masks)
        print(f"  Matching predictions to canonical ordering...")
        matched_pred = match_predictions_by_country_aoi(
            pred_detections, pred_mapping, country_aoi_list
        )
        model_detections[model_name] = matched_pred
        print(f"    Matched {len(matched_pred)} predictions")
        
        # Diagnostic: Check total instances
        total_pred_instances = sum(len(det) for det in matched_pred)
        print(f"  Total pred instances: {total_pred_instances}")
        if total_pred_instances == 0:
            print(f"  WARNING: No predictions found in {detections_path}!")
    
    # Run country-level evaluation
    print(f"\nRunning country-level evaluation with metrics: {args.metrics}")
    print(f"IoU threshold: {args.iou_threshold}")
    print("  Object metrics use semantic masks with connected components (matching FTW baseline)")
    country_results = evaluate_by_country(
        model_detections, gt_masks, gt_instance_masks, countries_list, country_aoi_list, image_ids,
        iou_threshold=args.iou_threshold, metrics=args.metrics
    )
    
    # Save results
    save_country_results(country_results, args.output_dir, args.output_name, args.save_csv)


if __name__ == "__main__":
    main()
