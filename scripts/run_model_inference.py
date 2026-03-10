#!/usr/bin/env python3
"""Run inference on models using registry-based segmenters."""

import argparse
import os
import pickle
import re
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from prue_eval.detections import Detections
from prue_eval.intermediate_formats import SemanticOutput, InstanceOutput, PanopticOutput
from ftw_tools.settings import ALL_COUNTRIES

from prue_eval.models.registry import create_segmenter, available_models
import prue_eval.models  # noqa: F401 — triggers adapter registration


def _sort_dataset_filenames(ds):
    """Sort dataset filenames by (country, aoi_id) for deterministic ordering."""

    def _sort_key(filename_dict):
        path = filename_dict.get("window_b") or filename_dict.get("window_a") or filename_dict.get("mask") or ""
        match = re.search(
            r"/([^/]+)/(?:s2_images/(?:window_a|window_b)|label_masks/(?:instance|semantic_2class|semantic_3class))/([^/]+)\.tif$",
            path,
        )
        if match:
            country = match.group(1)
            try:
                aoi_id = int(match.group(2))
            except ValueError:
                aoi_id = match.group(2)
            return (country, aoi_id)
        return ("", "")

    if hasattr(ds, "filenames") and ds.filenames:
        ds.filenames = sorted(ds.filenames, key=_sort_key)
    elif hasattr(ds, "file_list") and ds.file_list:
        ds.file_list = sorted(ds.file_list, key=_sort_key)


# Temporal options per model family
_TEMPORAL_DEFAULTS = {
    "ftw": "stacked",
    "sam": "stacked",
    "decode": "stacked",
    "delineate_anything": "windowA",
    "da": "windowA",
}

# Models that handle their own preprocessing (no preprocess transform)
_RAW_INPUT_MODELS = {"sam", "delineate_anything", "da", "mask2former"}


def run_registry_inference(model_name: str, data_dir: str, model_weights: str, output_dir: str, **kwargs) -> List[Detections]:
    """Run inference using a registry-based segmenter."""
    print(f"Running {model_name} inference...")

    segmenter_kwargs = {
        "model_weights": model_weights,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    # Model-specific kwargs
    if model_name == "sam":
        segmenter_kwargs["model_type"] = kwargs.get("sam_model_type", "vit_h")
        segmenter_kwargs["in_chans"] = kwargs.get("in_chans", 8)
        if kwargs.get("config_file"):
            import yaml
            with open(kwargs["config_file"]) as f:
                segmenter_kwargs["config"] = yaml.safe_load(f)
    elif model_name == "decode" and kwargs.get("config_file"):
        segmenter_kwargs["config"] = kwargs["config_file"]
    elif model_name == "mask2former" and kwargs.get("config_file"):
        segmenter_kwargs["config_file"] = kwargs["config_file"]
    elif model_name in ("delineate_anything", "da"):
        _apply_da_config(kwargs, segmenter_kwargs)

    segmenter = create_segmenter(model_name, **segmenter_kwargs)

    # Dataset
    from ftw_tools.torchgeo.datasets import FTW
    from ftw_tools.torchgeo.datamodules import preprocess

    countries = sorted(kwargs.get("countries", ["belgium"]))
    temporal = kwargs.get("temporal_options") or _TEMPORAL_DEFAULTS.get(model_name, "stacked")
    use_transforms = None if model_name in _RAW_INPUT_MODELS else preprocess

    ds = FTW(
        root=data_dir,
        countries=countries,
        split=kwargs.get("split", "test"),
        transforms=use_transforms,
        load_boundaries=False,
        temporal_options=temporal,
    )
    _sort_dataset_filenames(ds)

    dataloader = DataLoader(ds, batch_size=kwargs.get("batch_size", 16), shuffle=False, num_workers=4)
    filenames = getattr(ds, "filenames", None) or getattr(ds, "file_list", None)
    conf_threshold = kwargs.get("confidence_threshold") or 0.5

    detections_list = []
    sample_idx = 0

    with torch.inference_mode():
        for batch in tqdm(dataloader, desc=f"Inference [{model_name}]"):
            images = batch.get("image", batch.get("images")) if isinstance(batch, dict) else batch[0] if isinstance(batch, (list, tuple)) else batch
            outputs = list(segmenter.predict(images))

            for output in outputs:
                if isinstance(output, SemanticOutput):
                    det = output.to_detections(field_class_id=1, min_area=kwargs.get("min_area", 0))
                elif isinstance(output, InstanceOutput):
                    det = output.to_detections(min_area=kwargs.get("min_area", 0), score_threshold=conf_threshold)
                elif isinstance(output, PanopticOutput):
                    det = output.to_detections(min_area=kwargs.get("min_area", 0))
                else:
                    raise ValueError(f"Unexpected output type: {type(output)}")

                if filenames and sample_idx < len(filenames):
                    fd = filenames[sample_idx]
                    if temporal == "stacked":
                        det.image_filename = fd.get("window_b") or fd.get("window_a") or ""
                    else:
                        det.image_filename = fd.get(f"window_{temporal[-1].lower()}") or fd.get("window_a") or ""

                detections_list.append(det)
                sample_idx += 1

    return detections_list


def _apply_da_config(kwargs, segmenter_kwargs):
    """Load DelineateAnything config and apply defaults."""
    import yaml

    config_file = kwargs.get("config_file") or Path(__file__).parent.parent / "configs" / "delineate_anything" / "default.yaml"
    if Path(config_file).exists():
        with open(config_file) as f:
            config = yaml.safe_load(f)
        if "inference" in config:
            inf = config["inference"]
            for key in ("confidence_threshold", "iou_threshold", "max_detections", "patch_size", "resize_factor"):
                if kwargs.get(key) is None and key in inf:
                    kwargs[key] = inf[key]

    # Hard defaults for remaining None values
    kwargs.setdefault("patch_size", 256)
    kwargs.setdefault("resize_factor", 2)
    kwargs.setdefault("max_detections", 100)
    kwargs.setdefault("iou_threshold", 0.3)


def main():
    parser = argparse.ArgumentParser(description="Run model inference", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--model", type=str, required=True, choices=list(available_models().keys()))
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_weights", type=str)
    parser.add_argument("--config_file", type=str)
    parser.add_argument("--countries", nargs="+", default=["belgium"])
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--temporal_options", type=str, default=None, choices=["stacked", "windowA", "windowB", "rgb", "median", "random_window"])
    parser.add_argument("--output_name", type=str, default=None)
    parser.add_argument("--min_area", type=int, default=0)
    parser.add_argument("--confidence_threshold", type=float, default=None)

    sam_group = parser.add_argument_group("SAM options")
    sam_group.add_argument("--sam_model_type", type=str, default="vit_h")
    sam_group.add_argument("--sam_in_chans", type=int, default=8)

    da_group = parser.add_argument_group("DelineateAnything options")
    da_group.add_argument("--patch_size", type=int, default=None)
    da_group.add_argument("--resize_factor", type=int, default=None)
    da_group.add_argument("--max_detections", type=int, default=None)
    da_group.add_argument("--iou_threshold", type=float, default=None)

    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        print(f"Error: {args.data_dir} does not exist")
        sys.exit(1)

    if args.countries == ["all"]:
        args.countries = list(ALL_COUNTRIES)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.output_name is None:
        suffix = "all" if len(args.countries) > 10 else "-".join(args.countries)
        args.output_name = f"{args.model}_detections_{suffix}.pkl"

    # Validate weights
    if not args.model_weights and args.model not in ("delineate_anything", "da"):
        print(f"Error: --model_weights required for {args.model}")
        sys.exit(1)

    if args.model in ("delineate_anything", "da") and not args.model_weights:
        args.model_weights = "DelineateAnything"

    runner_args = {
        "data_dir": args.data_dir,
        "output_dir": args.output_dir,
        "countries": args.countries,
        "split": args.split,
        "batch_size": args.batch_size,
        "min_area": args.min_area,
        "confidence_threshold": args.confidence_threshold,
        "temporal_options": args.temporal_options,
        "model_weights": args.model_weights,
    }
    if args.config_file:
        runner_args["config_file"] = args.config_file
    if args.model == "sam":
        runner_args["sam_model_type"] = args.sam_model_type
        runner_args["in_chans"] = args.sam_in_chans
    elif args.model in ("delineate_anything", "da"):
        for k in ("patch_size", "resize_factor", "max_detections", "iou_threshold"):
            runner_args[k] = getattr(args, k)

    try:
        start = time.time()
        detections_list = run_registry_inference(model_name=args.model, **runner_args)

        output_path = os.path.join(args.output_dir, args.output_name)
        with open(output_path, "wb") as f:
            pickle.dump(detections_list, f)

        print(f"\nDone in {time.time() - start:.1f}s — {len(detections_list)} samples → {output_path}")

    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
