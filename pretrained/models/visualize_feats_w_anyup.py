#!/usr/bin/env python3
"""
Sanity check: randomly sample three Sentinel-2 tiles from a (user-specified or random) FTW country,
compute embeddings from CLAY, TerraFM, and DINOv3,
reshape transformer outputs to 2D, upsample using both bilinear interpolation
and a pretrained AnyUp upsampler, take PCA across channels,
and visualize PCA-RGB maps side-by-side with the original images.
"""

import os
import random
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")  # ✅ ensures it works on headless servers
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
import torch.nn.functional as F
from einops import rearrange
import argparse

from .model_utils import (
    load_image,
    prepare_clay_batch,
    get_model_and_preprocess,
)

# ============================================================
# 🌱 REPRODUCIBILITY
# ============================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🔒 Random seed fixed at {seed}")

set_seed(42)

# ============================================================
# 1️⃣ CONFIGURATION
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
metadata_path = "/u/subashk/storage/ftw-ablation/FTW-Bakeoff/ftw-baselines-2/configs/metadata.yaml"
data_root = Path("/projects/benq/ftw-data/data/ftw")
latvia_root = Path("/projects/benq/ftw-data/latvia")

# ============================================================
# 🌍 COUNTRY AND TILE SELECTION (via argparse + reproducibility)
# ============================================================
parser = argparse.ArgumentParser(description="Visualize PCA embeddings for Sentinel-2 tiles")
parser.add_argument("--country", type=str, default="croatia", help="Country name to visualize (default: random)")
parser.add_argument("--num_samples", type=int, default=3, help="Number of random tiles to sample")
parser.add_argument("--window", type=str, choices=["window_a", "window_b"], default=None,
                    help="Select window_a or window_b (default: random)")
args = parser.parse_args()

# gather all available country dirs
country_dirs = sorted([p for p in data_root.iterdir() if p.is_dir()])
if latvia_root.exists():
    country_dirs.append(latvia_root)

# choose country (user-provided or random)
if args.country:
    matches = [p for p in country_dirs if p.name.lower() == args.country.lower()]
    if not matches:
        raise ValueError(f"❌ Country '{args.country}' not found. Available: {[p.name for p in country_dirs]}")
    country = matches[0]
else:
    country = random.choice(country_dirs)

# choose window
window = args.window or random.choice(["window_a", "window_b"])
s2_dir = country / "s2_images"
window_dir = s2_dir / window

tifs = sorted(window_dir.glob("*.tif"))
if not tifs:
    raise RuntimeError(f"No TIFF files found in {window_dir}")

num_samples = min(args.num_samples, len(tifs))
sample_paths = random.sample(tifs, num_samples)

save_path = Path(f"features_visualization_w_upsampling.png")

print(f"\n🌍 Country: {country.name}")
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
    pca = PCA(n_components=3, random_state=42)
    x_pca = pca.fit_transform(x)
    x_pca = (x_pca - x_pca.min(0)) / (x_pca.max(0) - x_pca.min(0) + 1e-8)
    return x_pca.reshape(H, W, 3)


def normalize_emb_output(emb: torch.Tensor, model_name: str, input_hw, upsampler=None, hr_image=None, norm_const=3000.0):
    """
    Normalize transformer output [B, N, D] → [B, D, H, W],
    and upsample using both bilinear and AnyUp (if available).
    For Sentinel-2 inputs, only approximate RGB bands are normalized to ImageNet stats.
    """
    print(f"Original embedding shape from {model_name}: {emb.shape}")
    feat_resolution = int(emb.shape[1] ** 0.5)
    emb = rearrange(emb, "b (h w) d -> b d h w", h=feat_resolution, w=feat_resolution)

    emb_up_bilinear = F.interpolate(emb, size=input_hw, mode="bilinear", align_corners=False)

    emb_up_anyup = None
    if upsampler is not None and hr_image is not None:
        print("🔼 Using AnyUp upsampler with Sentinel-2 pseudo-RGB normalization...")
        rgb = torch.clamp(hr_image[:, :3, :, :] / norm_const, 0, 1)
        mean = torch.tensor([0.485, 0.456, 0.406], device=hr_image.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=hr_image.device).view(1, 3, 1, 1)
        hr_image_norm = (rgb - mean) / std
        with torch.no_grad():
            emb_up_anyup = upsampler(hr_image_norm, emb,q_chunk_size=256)

    return emb_up_bilinear, emb_up_anyup

# ============================================================
# 3️⃣ LOAD MODELS AND COMPUTE EMBEDDINGS
# ============================================================
print("📦 Loading AnyUp upsampler...")
upsampler = torch.hub.load('wimmerth/anyup', 'anyup', trust_repo=True).to(device)
upsampler.eval()

models = ["clay", "terrafm", "dinov3"]
results = []

for image_path in sample_paths:
    print(f"\n🖼️ Processing {image_path.name} ...")

    image_tensor, _, _ = load_image(str(image_path))
    rgb_img = image_tensor[:3].numpy()
    rgb_img = np.clip(rgb_img / 3000.0, 0, 1).transpose(1, 2, 0)
    sample_outputs = {"rgb": rgb_img}

    for model_name in models:
        print(f"🚀 Running {model_name.upper()}...")
        encoder, preprocess, gsd, waves = get_model_and_preprocess(model_name, device, metadata_path)
        with torch.no_grad():
            if model_name == "clay":
                sample = prepare_clay_batch([str(image_path)], device=device,
                                            preprocess=preprocess, gsd=gsd, waves=waves)
                emb = encoder(sample)
            else:
                select_rgb = model_name == "dinov3"
                image, _, _ = load_image(str(image_path), select_rgb=select_rgb)
                s = preprocess({"image": image})
                image = s["image"].unsqueeze(0).to(device)
                emb = encoder(image)

        emb_up_bilinear, emb_up_anyup = normalize_emb_output(
            emb, model_name, image_tensor.shape[1:], upsampler=upsampler,
            hr_image=image_tensor.unsqueeze(0).to(device)
        )

        sample_outputs[f"{model_name}_bilinear"] = emb_up_bilinear
        if emb_up_anyup is not None:
            sample_outputs[f"{model_name}_anyup"] = emb_up_anyup

    results.append(sample_outputs)

# ============================================================
# 4️⃣ PCA VISUALIZATION
# ============================================================
fig, axes = plt.subplots(num_samples, 1 + len(models) * 2, figsize=(24, 6 * num_samples))
col_titles = ["Original RGB"] + [f"{m.upper()} (Bilinear)" for m in models] + [f"{m.upper()} (AnyUp)" for m in models]

for row_idx, sample in enumerate(results):
    for col_idx, title in enumerate(col_titles):
        ax = axes[row_idx, col_idx]
        if col_idx == 0:
            ax.imshow(sample["rgb"])
        else:
            key = title.split()[0].lower() + "_" + title.split()[-1].lower().strip("()")
            if key in sample:
                emb_rgb = compute_pca_rgb(sample[key])
                ax.imshow(emb_rgb)
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=12)
        ax.set_title(title)
        ax.axis("off")

plt.tight_layout()
plt.savefig(save_path, dpi=200)
print(f"\n✅ Saved PCA visualization: {save_path}")
