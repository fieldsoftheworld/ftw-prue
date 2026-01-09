#!/usr/bin/env python3
"""
Compute embeddings for all Sentinel-2 images (window_a & window_b)
across all FTW countries using CLAY, TerraFM, or DINOv3 encoders.
Supports batched inference and saves float16 embeddings for compact storage.
"""

import os
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
from math import ceil

from .model_utils import (
    load_image,
    prepare_clay_batch,
    get_model_and_preprocess,
)
from ..path_config import get_data_root, get_metadata_path
import numpy as np

# ============================================================
# 1️⃣ ARGUMENTS
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Compute embeddings for FTW Sentinel-2 tiles (batched).")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["clay", "terrafm", "dinov3", "croma", "decur", "dofa", "prithvi", "satlas", "softcon", "galileo"],
        help="Model type for embedding extraction",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Base FTW data directory (defaults to FTW_DATA_ROOT env var or ./data/ftw)",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default=None,
        help="Path to metadata YAML (defaults to FTW_METADATA_PATH env var or ./configs/metadata.yaml)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save computed embeddings (.npz files)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for batched embedding computation",
    )
    return parser.parse_args()


# ============================================================
# 2️⃣ EMBEDDING COMPUTATION
# ============================================================
def compute_embeddings(model_name: str, data_path: str = None, metadata_path: str = None, output_dir: str = None, batch_size: int = 32):
    """
    Compute embeddings for FTW dataset using specified model.
    
    Args:
        model_name: Name of the model to use
        data_path: Path to FTW data directory (defaults to path_config.get_data_root())
        metadata_path: Path to metadata YAML (defaults to path_config.get_metadata_path())
        output_dir: Output directory for embeddings (defaults to ./precomputed_feats/{model_name})
        batch_size: Batch size for processing
    """
    if data_path is None:
        data_path = str(get_data_root())
    
    if metadata_path is None:
        metadata_path = str(get_metadata_path())
    
    if output_dir is None:
        output_dir = f"./precomputed_feats/{model_name}"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, preprocess, gsd, waves = get_model_and_preprocess(model_name, device, metadata_path)

    countries_root = Path(data_path)
    # latvia_root = Path("/projects/benq/ftw-data/latvia")  # fixed path for Latvia

    # Gather all countries
    country_dirs = sorted([p for p in countries_root.iterdir() if p.is_dir()])
    # if latvia_root.exists():
    #     country_dirs.append(latvia_root)

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n🚀 Starting embedding computation for {model_name.upper()}")
    print(f"📂 Data root: {countries_root}")
    print(f"💾 Output dir: {output_dir}")
    print(f"🌍 Found {len(country_dirs)} countries (including Latvia if present).")

    # Iterate over countries
    for country in tqdm(country_dirs, desc="🌍 Countries"):
        s2_dir = country / "s2_images"
        if not s2_dir.exists():
            print(f"⚠️ Skipping {country.name}: no s2_images folder")
            continue

        for window in ["window_a", "window_b"]:
            window_dir = s2_dir / window
            if not window_dir.exists():
                continue

            tifs = sorted(window_dir.glob("*.tif"))
            if not tifs:
                continue

            # Create output directories
            country_dir = Path(output_dir) / country.name / window
            country_dir.mkdir(parents=True, exist_ok=True)

            n_batches = ceil(len(tifs) / batch_size)
            for b in tqdm(range(n_batches), desc=f"{country.name}/{window}", leave=False):
                batch_paths = tifs[b * batch_size : (b + 1) * batch_size]

                try:
                    if model_name == "clay":
                        sample = prepare_clay_batch(
                            image_paths=[str(p) for p in batch_paths],
                            device=device,
                            preprocess=preprocess,
                            gsd=gsd,
                            waves=waves,
                        )
                        with torch.no_grad():
                            emb_batch = encoder(sample).detach().cpu().to(torch.float16)
                    else:
                        images = []
                        for img_path in batch_paths:
                            select_rgb = model_name == "dinov3"
                            image, _, _ = load_image(str(img_path), select_rgb=select_rgb)
                            sample = {"image": image}
                            sample = preprocess(sample)
                            images.append(sample["image"])

                        batch_tensor = torch.stack(images).to(device)
                        with torch.no_grad():
                            emb_batch = encoder(batch_tensor).detach().cpu().to(torch.float16)

                    for img_path, emb in zip(batch_paths, emb_batch):
                        out_path = country_dir / f"{model_name}_{img_path.stem}.npz"
                        emb_np = emb.numpy()  # Convert to NumPy array
                        # import code; code.interact(local=dict(globals(), **locals()))
                        np.savez_compressed(out_path, embedding=emb_np)

                except Exception as e:
                    print(f"❌ Batch failed for {country.name}/{window}: {e}")
                    continue

    print(f"\n✅ Done! Float16 embeddings saved under: {output_dir}")


# ============================================================
# 3️⃣ MAIN
# ============================================================
if __name__ == "__main__":
    args = parse_args()
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = f"./precomputed_feats/{args.model}"
    else:
        output_dir = os.path.join(output_dir, args.model)
    compute_embeddings(args.model, args.data_path, args.metadata, output_dir, args.batch_size)


#python -m models.compute_feats --output_dir /u/subashk/storage/ftw-prue/logs/precomputed_feats --model clay