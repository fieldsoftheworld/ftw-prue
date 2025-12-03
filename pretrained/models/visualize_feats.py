#!/usr/bin/env python3
"""
Sanity check: randomly sample three Sentinel-2 tiles from a random FTW country,
compute embeddings from CLAY, TerraFM, and DINOv3,
reshape transformer outputs to 2D, upsample to input size, take PCA across channels,
and visualize PCA-RGB maps side-by-side with the original images.
"""

import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
import torch.nn.functional as F
from einops import rearrange
import math

from .model_utils import (
    load_image,
    prepare_clay_batch,
    get_model_and_preprocess,
)

# ============================================================
# 1️⃣ CONFIGURATION
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
metadata_path = "/u/subashk/storage/ftw-ablation/FTW-Bakeoff/ftw-baselines-2/configs/metadata.yaml"
data_root = Path("/projects/benq/ftw-data/data/ftw")
latvia_root = Path("/projects/benq/ftw-data/latvia")
save_path = Path("features_visualization.png")

# gather all available country dirs
country_dirs = sorted([p for p in data_root.iterdir() if p.is_dir()])
if latvia_root.exists():
    country_dirs.append(latvia_root)

# randomly pick one country and window (a/b)
country = random.choice(country_dirs)
s2_dir = country / "s2_images"
window = random.choice(["window_a", "window_b"])
window_dir = s2_dir / window

# pick 3 random tifs
tifs = sorted(window_dir.glob("*.tif"))
if not tifs:
    raise RuntimeError(f"No TIFF files found in {window_dir}")
num_samples = min(3, len(tifs))
sample_paths = random.sample(tifs, num_samples)

print(f"🌍 Country: {country.name}")
print(f"🪟 Window: {window}")
print(f"🧩 Selected {num_samples} random tiles:")
for p in sample_paths:
    print(f"   - {p.name}")

# ============================================================
# 2️⃣ HELPER FUNCTIONS
# ============================================================
def compute_pca_rgb(feats: torch.Tensor):
    """Compute PCA across channels → 3 components (for visualization)."""
    B, C, H, W = feats.shape
    x = feats[0].permute(1, 2, 0).reshape(-1, C).cpu().numpy()
    pca = PCA(n_components=3)
    x_pca = pca.fit_transform(x)
    x_pca = (x_pca - x_pca.min(0)) / (x_pca.max(0) - x_pca.min(0) + 1e-8)
    return x_pca.reshape(H, W, 3)


def normalize_emb_output(emb: torch.Tensor, model_name: str, input_hw):
    """
    Normalize transformer output [B, N, D] → [B, D, H, W],
    using einops.rearrange and handling non-square token grids.
    """
    print(f"Original embedding shape from {model_name}: {emb.shape}")
    feat_resolution = emb.shape[1] ** 0.5
    emb = rearrange(emb, "b (h w) d -> b h w d", h=int(feat_resolution), w=int(feat_resolution))
    emb = emb.permute(0, 3, 1, 2)  # [B, D, h, w]
    emb_up = F.interpolate(emb, size=(256, 256), mode="bilinear", align_corners=False)
    return emb_up


# ============================================================
# 3️⃣ LOAD MODELS AND COMPUTE EMBEDDINGS
# ============================================================
models = ["clay", "terrafm", "dinov3"]

# Store results for each sample and model
results = []

for image_path in sample_paths:
    print(f"\n🖼️ Processing {image_path.name} ...")

    # Load RGB for visualization
    image_tensor, _, _ = load_image(str(image_path))
    rgb_img = image_tensor[:3].numpy()
    rgb_img = np.clip(rgb_img / 3000.0, 0, 1).transpose(1, 2, 0)

    sample_outputs = {"rgb": rgb_img}

    # Run all models
    for model_name in models:
        print(f"🚀 Running {model_name.upper()}...")
        encoder, preprocess, gsd, waves = get_model_and_preprocess(model_name, device, metadata_path)

        with torch.no_grad():
            if model_name == "clay":
                sample = prepare_clay_batch(
                    [str(image_path)],
                    device=device,
                    preprocess=preprocess,
                    gsd=gsd,
                    waves=waves,
                )
                emb = encoder(sample)
            else:
                select_rgb = model_name == "dinov3"
                image, _, _ = load_image(str(image_path), select_rgb=select_rgb)
                s = preprocess({"image": image})
                image = s["image"].unsqueeze(0).to(device)
                emb = encoder(image)

        # Normalize embedding shape and upsample
        emb_up = normalize_emb_output(emb, model_name, image_tensor.shape[1:])
        sample_outputs[model_name] = emb_up

    results.append(sample_outputs)

# ============================================================
# 4️⃣ PCA VISUALIZATION
# ============================================================
fig, axes = plt.subplots(num_samples, len(models) + 1, figsize=(20, 6 * num_samples))

for row_idx, sample in enumerate(results):
    axes[row_idx, 0].imshow(sample["rgb"])
    axes[row_idx, 0].set_title(f"Original RGB ({country.name})")
    axes[row_idx, 0].axis("off")

    for col_idx, model_name in enumerate(models, start=1):
        emb_rgb = compute_pca_rgb(sample[model_name])
        axes[row_idx, col_idx].imshow(emb_rgb)
        axes[row_idx, col_idx].set_title(f"{model_name.upper()} PCA RGB")
        axes[row_idx, col_idx].axis("off")

plt.tight_layout()
plt.savefig(save_path, dpi=200)
plt.show()

print(f"\n✅ Saved PCA visualization: {save_path}")
