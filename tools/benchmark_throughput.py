"""
Benchmark FTW model throughput in km²/second.

Usage (multiple models from logs directory):
    python benchmark_throughput.py \
        --logs-dir logs/ \
        --sample-image data/ftw/austria/s2_images/window_a/g77_00002_10.tif \
        [--batch-size 64] \
        [--num-batches 50] \
        [--device cuda:0] \
        [--output-csv throughput_benchmark_results.csv] \
        [--num-workers 4]

Usage (single model from checkpoint):
    python benchmark_throughput.py \
        --weights path/to/model.ckpt \
        --sample-image data/ftw/austria/s2_images/window_a/g77_00002_10.tif \
        [--config-file path/to/config.yaml] \
        [--batch-size 64] \
        [--num-batches 50] \
        [--device cuda:0] \
        [--output-csv throughput_benchmark_results.csv] \
        [--num-workers 4]

    # For DelineateAnything models (.pt files), config-file is not needed:
    python benchmark_throughput.py \
        --weights path/to/delineate_anything_rgb_yolo11x-88ede029.pt \
        --sample-image data/ftw/austria/s2_images/window_a/g77_00002_10.tif \
        [--batch-size 64] \
        [--num-batches 50]
"""

import torch
from torch.utils.data import DataLoader, Dataset
import os
import sys
import yaml
import time
import argparse
from pathlib import Path
from einops import rearrange
import rasterio
import fiona.transform
import shapely.geometry
import pandas as pd

# Add project paths to sys.path so we can import ftw_tools
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root / "src" / "models" / "ftw"))
sys.path.insert(0, str(project_root / "src" / "models"))

from ftw_tools.training.trainers import CustomSemanticSegmentationTask
from ftw_tools.inference.models import DelineateAnything, load_model_from_checkpoint


def get_pixel_resolution_from_image(image_path: str) -> tuple:
    """Get pixel resolution in meters from a GeoTIFF image."""
    with rasterio.open(image_path) as f:
        geom = shapely.geometry.mapping(shapely.geometry.box(*f.bounds))
        geom_6933 = fiona.transform.transform_geom(f.crs, "EPSG:6933", geom)
        minx, miny, maxx, maxy = shapely.geometry.shape(geom_6933).bounds
        res_x = (maxx - minx) / f.width
        res_y = (maxy - miny) / f.height
        pixel_size_meters = (res_x + res_y) / 2
        print(f"Pixel resolution in EPSG:6933: {res_x:.2f} x {res_y:.2f} meters (avg: {pixel_size_meters:.2f})")
        return res_x, res_y, pixel_size_meters


def find_checkpoints(logs_dir: str):
    """
    Find all checkpoints and their configs in the logs directory.
    
    Returns:
        list of tuples: (checkpoint_path, config_path, model_type, backbone, num_filters)
    """
    checkpoints = []
    for root, dirs, files in os.walk(logs_dir):
        for file in files:
            if file.endswith("last.ckpt"):
                parent_dir = os.path.dirname(root)
                config_file_path = os.path.join(parent_dir, "config.yaml")
                checkpoint_path = os.path.join(root, file)
                if os.path.isfile(config_file_path):
                    with open(config_file_path, "r") as conf_file:
                        config_data = yaml.safe_load(conf_file)

                    trainer_init = config_data.get("model").get("init_args")
                    checkpoints.append((
                        checkpoint_path,
                        config_file_path,
                        trainer_init["model"],
                        trainer_init["backbone"],
                        trainer_init["num_filters"]
                    ))
                else:
                    print(f"Missing config for checkpoint {root}")
    return checkpoints


class DummyDataset(Dataset):
    def __init__(self, num_samples=1000, channels=8, height=256, width=256):
        self.num_samples = num_samples
        self.channels = channels
        self.height = height
        self.width = width

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sample = torch.randn(self.channels, self.height, self.width)
        return sample


def benchmark_model(
    model,
    model_type,
    dataloader,
    device,
    pixel_size_meters,
    image_height,
    image_width,
    is_delineate_anything=False
):
    """Benchmark a single model."""
    if not is_delineate_anything:
        model = model.eval().to(device)
    
    tic = time.time()
    for images in dataloader:
        if is_delineate_anything:
            # DelineateAnything expects (B, C, H, W) format
            images = images.to(device)
            with torch.inference_mode():
                # DelineateAnything.__call__ returns list[Results], we just need to call it
                _ = model(images)
        else:
            # FTW models
            if model_type in ["fcsiamdiff", "fcsiamconc", "fcsiamavg"]:
                images = rearrange(images, "b (t c) h w -> b t c h w", t=2)
            images = images.to(device)
            with torch.inference_mode():
                outputs = model(images)
    toc = time.time()
    elapsed_time = toc - tic

    total_images = len(dataloader.dataset)
    throughput = total_images / elapsed_time

    area_per_image_m2 = (image_height * pixel_size_meters) * (image_width * pixel_size_meters)
    area_per_image_km2 = area_per_image_m2 / 1e6
    throughput_km2_per_sec = throughput * area_per_image_km2
    
    print(f"Throughput: {throughput_km2_per_sec:.2f} km²/second")
    print(f"  Images/sec: {throughput:.2f}")
    print(f"  Time: {elapsed_time:.2f} seconds")
    
    return {
        "throughput_images_per_sec": throughput,
        "throughput_km2_per_sec": throughput_km2_per_sec,
        "elapsed_time": elapsed_time
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark FTW model throughput")
    
    # Two modes: either logs-dir OR weights (with optional config-file)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--logs-dir", help="Directory containing logs/checkpoints (e.g., 'logs/')")
    group.add_argument("--weights", help="Path to model checkpoint file (.ckpt or .pt)")
    
    parser.add_argument("--config-file", help="Path to config.yaml file (optional, only needed for some .ckpt files)")
    parser.add_argument("--sample-image", required=True, help="Path to sample GeoTIFF to extract pixel resolution")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--num-batches", type=int, default=50, help="Number of batches to process (default: 50)")
    parser.add_argument("--device", default="cuda:0", help="Device to run on (default: cuda:0)")
    parser.add_argument("--output-csv", default="throughput_benchmark_results.csv", help="Output CSV file (default: throughput_benchmark_results.csv)")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of dataloader workers (default: 4)")
    parser.add_argument("--channels", type=int, default=8, help="Number of input channels (default: 8)")
    parser.add_argument("--height", type=int, default=256, help="Image height in pixels (default: 256)")
    parser.add_argument("--width", type=int, default=256, help="Image width in pixels (default: 256)")
    parser.add_argument("--num-classes", type=int, default=3, help="Number of classes (default: 3)")
    args = parser.parse_args()
    
    # Validate arguments
    if args.weights and not args.logs_dir:
        # When using --weights, we're in single model mode
        pass
    
    # Get pixel resolution
    res_x, res_y, pixel_size_meters = get_pixel_resolution_from_image(args.sample_image)
    
    # Setup device
    device = torch.device(args.device)
    
    # Create dataset and dataloader
    ds = DummyDataset(
        num_samples=args.batch_size * args.num_batches,
        channels=args.channels,
        height=args.height,
        width=args.width
    )
    dataloader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    rows = []
    
    if args.logs_dir and not args.weights:
        # Mode 1: Find checkpoints in logs directory
        print(f"\nSearching for checkpoints in {args.logs_dir}...")
        checkpoints = find_checkpoints(args.logs_dir)
        print(f"Found {len(checkpoints)} checkpoints")
        
        if len(checkpoints) == 0:
            print("No checkpoints found. Exiting.")
            return
        
        # Get unique model configurations
        unique_configs = {}
        for checkpoint_path, config_path, model_type, backbone, num_filters in checkpoints:
            key = (model_type, backbone, num_filters)
            if key not in unique_configs:
                unique_configs[key] = (checkpoint_path, config_path, model_type, backbone, num_filters)
        
        print(f"Found {len(unique_configs)} unique model configurations")
        for model_type, backbone, num_filters in unique_configs.keys():
            print(f"  - Model: {model_type}, Backbone: {backbone}, Filters: {num_filters}")
        
        # Benchmark each unique model configuration
        for checkpoint_path, config_path, model_type, backbone, num_filters in unique_configs.values():
            print(f"\n{'='*60}")
            print(f"Benchmarking: Model={model_type}, Backbone={backbone}, Filters={num_filters}")
            print(f"  Checkpoint: {checkpoint_path}")
            print(f"{'='*60}")
            
            # Load model from checkpoint
            task = CustomSemanticSegmentationTask.load_from_checkpoint(
                checkpoint_path,
                map_location="cpu"
            )
            model = task.model
            
            results = benchmark_model(
                model, model_type, dataloader, device,
                pixel_size_meters, args.height, args.width
            )
            
            rows.append({
                "model": model_type,
                "backbone": backbone,
                "num_filters": num_filters,
                "checkpoint": checkpoint_path,
                **results
            })
    
    if args.weights:
        # Mode 2: Load single model from checkpoint file
        print(f"\n{'='*60}")
        print(f"Loading model from checkpoint: {args.weights}")
        print(f"{'='*60}")
        
        # Determine model type based on file extension and content
        weights_path = Path(args.weights)
        is_delineate_anything = False
        model_type = "unknown"
        backbone = "unknown"
        num_filters = "unknown"
        
        if weights_path.suffix == ".pt":
            # Check if it's a DelineateAnything model (YOLO-based)
            try:
                import ultralytics
                
                # Determine which variant based on filename
                if "delineate_anything_s" in weights_path.name.lower() or "yolo11n" in weights_path.name.lower():
                    model_name = "DelineateAnything-S"
                else:
                    model_name = "DelineateAnything"
                
                print(f"Loading DelineateAnything model: {model_name}")
                
                # Load YOLO model directly from file path
                yolo_model = ultralytics.YOLO(str(weights_path))
                yolo_model.to(device)
                yolo_model.eval()
                yolo_model.fuse()
                
                # Create DelineateAnything wrapper
                model = DelineateAnything(
                    model=model_name,
                    patch_size=(args.height, args.width),
                    resize_factor=2,
                    device=args.device
                )
                # Replace the model with our loaded one
                model.model = yolo_model
                
                is_delineate_anything = True
                model_type = model_name
                backbone = "yolo11"
                num_filters = "N/A"
                
            except Exception as e:
                print(f"Failed to load as DelineateAnything: {e}")
                raise
        elif weights_path.suffix == ".ckpt":
            # Load FTW Lightning checkpoint
            try:
                task = CustomSemanticSegmentationTask.load_from_checkpoint(
                    args.weights,
                    map_location="cpu"
                )
                model = task.model
                
                # Get model type from checkpoint or config
                if args.config_file and Path(args.config_file).exists():
                    with open(args.config_file, "r") as f:
                        config_data = yaml.safe_load(f)
                    trainer_init = config_data.get("model", {}).get("init_args", {})
                    model_type = trainer_init.get("model", "unknown")
                    backbone = trainer_init.get("backbone", "unknown")
                    num_filters = trainer_init.get("num_filters", "unknown")
                else:
                    # Try to get from checkpoint hyper_parameters
                    ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
                    if "hyper_parameters" in ckpt:
                        hparams = ckpt["hyper_parameters"]
                        model_type = hparams.get("model", "unknown")
                        backbone = hparams.get("backbone", "unknown")
                        num_filters = hparams.get("num_filters", "unknown")
                
            except Exception as e:
                print(f"Failed to load as Lightning checkpoint: {e}")
                # Try loading with load_model_from_checkpoint as fallback
                print("Trying alternative loading method...")
                model, model_type = load_model_from_checkpoint(args.weights)
                backbone = "unknown"
                num_filters = "unknown"
        else:
            raise ValueError(f"Unsupported model file format: {weights_path.suffix}. Expected .ckpt or .pt")
        
        print(f"Model: {model_type}, Backbone: {backbone}, Filters: {num_filters}")
        
        results = benchmark_model(
            model, model_type, dataloader, device,
            pixel_size_meters, args.height, args.width,
            is_delineate_anything=is_delineate_anything
        )
        
        rows.append({
            "model": model_type,
            "backbone": backbone,
            "num_filters": num_filters,
            "checkpoint": args.weights,
            **results
        })

    # Save results
    df = pd.DataFrame(rows)
    df.to_csv(args.output_csv, index=False)
    print(f"\nResults saved to {args.output_csv}")


if __name__ == "__main__":
    main()
