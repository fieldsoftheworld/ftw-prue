#!/usr/bin/env python3
"""Evaluate model detections against ground truth with country-level breakdown."""

import argparse
import json
import os
import pickle
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from prue_eval.detections import Detections
from prue_eval.evaluator import Evaluator


def load_detections_from_file(path: str) -> List[Detections]:
    with open(path, "rb") as f:
        return pickle.load(f)


def _get_valid_aois(data_dir: str, country: str, split: str) -> List:
    """Get sorted AOI IDs that have all required files for a country/split."""
    import geopandas as gpd
    from pathlib import Path

    chips_file = os.path.join(data_dir, country, f"chips_{country}.parquet")
    if not os.path.exists(chips_file):
        print(f"Warning: chips file not found for {country}")
        return []

    chips = gpd.read_parquet(chips_file)
    aoi_ids = sorted(chips[chips["split"] == split]["aoi_id"].tolist())

    valid = []
    for aoi_id in aoi_ids:
        required = [
            Path(data_dir, country, "s2_images/window_b", f"{aoi_id}.tif"),
            Path(data_dir, country, "s2_images/window_a", f"{aoi_id}.tif"),
            Path(data_dir, country, "label_masks/semantic_2class", f"{aoi_id}.tif"),
            Path(data_dir, country, "label_masks/semantic_3class", f"{aoi_id}.tif"),
        ]
        if all(p.exists() for p in required):
            valid.append(aoi_id)

    return valid


def _read_mask_safe(path: str, fallback_shape=(256, 256)) -> np.ndarray:
    """Read a single-band raster mask, returning zeros on failure."""
    import rasterio
    try:
        with rasterio.open(path) as src:
            return src.read(1)
    except Exception as e:
        print(f"Warning: failed to read {path}: {e}")
        return np.zeros(fallback_shape, dtype=np.uint8)


def load_gt_masks(
    data_dir: str,
    countries: List[str],
    split: str = "test",
    mask_type: str = "semantic",
) -> Tuple[List[np.ndarray], List[str], List[Tuple[str, str]], List[int]]:
    """
    Load GT masks from FTW dataset.

    mask_type="semantic" → binary field masks (for pixel/object metrics)
    mask_type="instance" → instance ID masks (for COCO metrics)
    """
    masks, countries_list, country_aoi_list, image_ids = [], [], [], []

    for country in sorted(countries):
        valid_aois = _get_valid_aois(data_dir, country, split)
        print(f"  {country}: {len(valid_aois)} valid AOIs")

        for aoi_id in valid_aois:
            if mask_type == "instance":
                path = os.path.join(data_dir, country, "label_masks/instance", f"{aoi_id}.tif")
                mask = _read_mask_safe(path)
            else:
                # Prefer 2-class, fall back to 3-class
                path_2c = os.path.join(data_dir, country, "label_masks/semantic_2class", f"{aoi_id}.tif")
                path_3c = os.path.join(data_dir, country, "label_masks/semantic_3class", f"{aoi_id}.tif")
                if os.path.exists(path_2c):
                    mask = (_read_mask_safe(path_2c) == 1).astype(np.uint8)
                elif os.path.exists(path_3c):
                    mask = (_read_mask_safe(path_3c) == 1).astype(np.uint8)
                else:
                    continue

            masks.append(mask)
            countries_list.append(country)
            country_aoi_list.append((country, str(aoi_id)))
            image_ids.append(len(masks) - 1)

    print(f"Loaded {len(masks)} GT {mask_type} masks")
    return masks, countries_list, country_aoi_list, image_ids


def extract_country_aoi(filename: str) -> Tuple[str, str]:
    match = re.search(
        r"/([^/]+)/(?:s2_images/(?:window_a|window_b)|label_masks/(?:instance|semantic_2class|semantic_3class))/([^/]+)\.tif$",
        filename,
    )
    return (match.group(1), match.group(2)) if match else ("", "")


def match_predictions_by_filename(pred_detections: List[Detections], country_aoi_list: List[Tuple[str, str]]) -> List[Detections]:
    """Reorder predictions to match GT ordering using image filenames."""
    if not pred_detections or not getattr(pred_detections[0], "image_filename", None):
        raise ValueError("Detections must have image_filename set. Re-run inference with updated script.")

    lookup = {}
    for det in pred_detections:
        if det.image_filename:
            key = extract_country_aoi(det.image_filename)
            if key[0]:
                lookup[key] = det

    matched = []
    missing = 0
    for key in country_aoi_list:
        if key in lookup:
            matched.append(lookup[key])
        else:
            matched.append(Detections(xyxy=np.empty((0, 4))))
            missing += 1

    if missing:
        print(f"Warning: {missing}/{len(country_aoi_list)} predictions missing")
    return matched


def evaluate_by_country(
    model_detections: Dict[str, List[Detections]],
    gt_masks: List[np.ndarray],
    gt_instance_masks: Optional[List[np.ndarray]],
    countries_list: List[str],
    country_aoi_list: List[Tuple[str, str]],
    image_ids: List[int],
    iou_threshold: float = 0.5,
    metrics: List[str] = ["object"],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Per-country evaluation using semantic masks for object metrics."""
    unique_countries = sorted(set(countries_list))
    print(f"Evaluating {len(unique_countries)} countries...")
    country_results = {}

    for country in unique_countries:
        idx = [i for i, c in enumerate(countries_list) if c == country]
        if not idx:
            continue

        c_gt_masks = [gt_masks[i] for i in idx]
        c_image_ids = [image_ids[i] for i in idx]

        if "coco" in metrics and gt_instance_masks is not None:
            c_gt_dets = [Detections.from_gt(gt_instance_masks[i]) for i in idx]
        else:
            c_gt_dets = [Detections(xyxy=np.empty((0, 4))) for _ in idx]

        evaluator = Evaluator(
            iou_threshold=iou_threshold,
            metrics=metrics,
            gt_masks=c_gt_masks,
            image_ids=c_image_ids,
            use_semantic_masks_for_object_metrics=True,
        )

        country_results[country] = {}
        for model_name, det_list in model_detections.items():
            c_preds = [det_list[i] for i in idx]
            print(f"  {model_name} / {country} ({len(idx)} samples)")
            country_results[country][model_name] = evaluator.evaluate(y_true=c_gt_dets, y_pred=c_preds)

    return country_results


def _to_json_safe(obj):
    """Recursively convert numpy types to native Python for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_json_safe(x) for x in obj)
    return obj


def save_country_results(results: Dict, output_dir: str, output_name: str = "country_evaluation_results.json", save_csv: bool = False):
    os.makedirs(output_dir, exist_ok=True)
    safe = _to_json_safe(results)

    json_path = os.path.join(output_dir, output_name)
    with open(json_path, "w") as f:
        json.dump(safe, f, indent=2)
    print(f"Results → {json_path}")

    if save_csv:
        import csv

        countries = sorted(safe.keys())
        if not countries:
            return
        models = sorted(safe[countries[0]].keys())
        if not models:
            return

        csv_path = json_path.replace(".json", ".csv")
        metric_keys = sorted(safe[countries[0]][models[0]].keys())
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["country", "model"] + metric_keys)
            for c in countries:
                for m in models:
                    if m in safe[c]:
                        w.writerow([c, m] + [safe[c][m].get(k, 0) for k in metric_keys])
        print(f"CSV → {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate detections per country", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--model_detections", type=str, required=True, help="JSON file or string: {model_name: detections_path}")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True, help="FTW dataset root")
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    parser.add_argument("--metrics", nargs="+", choices=["pixel", "object", "coco", "all"], default=["all"])
    parser.add_argument("--countries", nargs="+", default=["all"])
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--output_name", type=str, default="country_evaluation_results.json")
    parser.add_argument("--save_csv", action="store_true")

    args = parser.parse_args()

    if args.countries == ["all"]:
        from ftw_tools.settings import ALL_COUNTRIES
        args.countries = list(ALL_COUNTRIES)

    if "all" in args.metrics:
        args.metrics = ["pixel", "object", "coco"]

    # Load GT
    print("Loading GT semantic masks...")
    gt_masks, countries_list, country_aoi_list, image_ids = load_gt_masks(args.data_dir, args.countries, args.split, "semantic")

    gt_instance_masks = None
    if "coco" in args.metrics:
        print("Loading GT instance masks for COCO...")
        gt_instance_masks, _, _, _ = load_gt_masks(args.data_dir, args.countries, args.split, "instance")
        # Align lengths
        n = min(len(gt_masks), len(gt_instance_masks))
        gt_masks, gt_instance_masks = gt_masks[:n], gt_instance_masks[:n]
        countries_list, country_aoi_list, image_ids = countries_list[:n], country_aoi_list[:n], image_ids[:n]

    # Load predictions
    if args.model_detections.endswith(".json"):
        with open(args.model_detections) as f:
            det_paths = json.load(f)
    else:
        try:
            det_paths = json.loads(args.model_detections)
        except json.JSONDecodeError:
            print(f"Error: invalid JSON: {args.model_detections}")
            sys.exit(1)

    model_detections = {}
    for name, path in det_paths.items():
        preds = load_detections_from_file(path)
        matched = match_predictions_by_filename(preds, country_aoi_list)
        model_detections[name] = matched
        total = sum(len(d) for d in matched)
        print(f"  {name}: {len(matched)} matched, {total} instances")

    # Evaluate
    results = evaluate_by_country(
        model_detections, gt_masks, gt_instance_masks, countries_list, country_aoi_list, image_ids,
        iou_threshold=args.iou_threshold, metrics=args.metrics,
    )
    save_country_results(results, args.output_dir, args.output_name, args.save_csv)


if __name__ == "__main__":
    main()
