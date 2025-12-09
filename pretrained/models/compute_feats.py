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
        choices=["clay", "terrafm", "dinov3"],
        help="Model type for embedding extraction (clay, terrafm, dinov3)",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/projects/benq/ftw-data/data/ftw",
        help="Base FTW data directory containing country folders (except Latvia)",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default="/u/subashk/storage/ftw-ablation/FTW-Bakeoff/ftw-baselines-2/configs/metadata.yaml",
        help="Path to metadata YAML for CLAY",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/projects/benq/ftw-data/precomputed_feats",
        help="Directory to save computed embeddings (.pt files)",
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
def compute_embeddings(model_name: str, data_path: str, metadata_path: str, output_dir: str, batch_size: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, preprocess, gsd, waves = get_model_and_preprocess(model_name, device, metadata_path)

    countries_root = Path(data_path)
    latvia_root = Path("/projects/benq/ftw-data/latvia")  # fixed path for Latvia

    # Gather all countries
    country_dirs = sorted([p for p in countries_root.iterdir() if p.is_dir()])
    if latvia_root.exists():
        country_dirs.append(latvia_root)

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
                    # ============================================================
                    # CLAY — use metadata + temporal context
                    # ============================================================
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

                    # ============================================================
                    # TerraFM / DINOv3 — standard tensor batching
                    # ============================================================
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

                    # ============================================================
                    # Save embeddings per sample
                    # ============================================================
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
    output_dir = os.path.join(args.output_dir, args.model)
    compute_embeddings(args.model, args.data_path, args.metadata, output_dir, args.batch_size)
