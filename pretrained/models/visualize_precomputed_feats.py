#!/usr/bin/env python3

import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
import torch.nn.functional as F
import rasterio
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--country_names",
    type=str,
    default="austria;brazil;belgium;rwanda;cambodia;france",
    help="Semicolon-separated FTW country names"
)
args = parser.parse_args()

seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

ftw_root = Path("/projects/benq/ftw-data/data/ftw")
precomputed_dir = Path("/projects/benq/ftw-data/precomputed_feats")

# MODEL KEYS IN FILESYSTEM
model_names = [
    "clay", "galileo", "dinov3", "prithvi", "terramind",
    "softcon", "satlas", "decur", "dofa", "terrafm", "croma"
]

# DISPLAY NAMES (your screenshot formatting)
MODEL_TITLES = {
    "clay": "Clay",
    "galileo": "Galileo",
    "dinov3": "DINOv3",
    "prithvi": "Prithvi 2.0",
    "terramind": "TerraMind",
    "softcon": "SoftCon",
    "satlas": "Satlas",
    "decur": "DeCUR",
    "dofa": "DOFA-v1",
    "terrafm": "TerraFM",
    "croma": "CROMA",
}

MODEL_CONFIGS = {
    "croma":{"embed_hw":15},"galileo":{"embed_hw":64},"decur":{"embed_hw":14},
    "dofa":{"embed_hw":14},"prithvi":{"embed_hw":14},"satlas":{"embed_hw":16},
    "softcon":{"embed_hw":16},"clay":{"embed_hw":32},"dinov3":{"embed_hw":16},
    "terrafm":{"embed_hw":16},"terramind":{"embed_hw":16},
}

def load_rgb(path):
    with rasterio.open(path) as src:
        img = src.read().astype(np.float32)
    rgb = img[:3]
    rgb = np.clip(rgb / 3000.0, 0, 1)
    return rgb.transpose(1, 2, 0)

def load_embedding(model, country, window, stem):
    f = precomputed_dir / model / country / window / f"{model}_{stem}.npz"
    return np.load(f)["embedding"]

def reshape_embedding(emb, model):
    hw = MODEL_CONFIGS[model]["embed_hw"]
    if emb.ndim == 2:
        L, D = emb.shape
        emb = torch.tensor(emb).reshape(hw, hw, D).permute(2, 0, 1).unsqueeze(0).float()
    else:
        emb = torch.tensor(emb).unsqueeze(0).float()
    return F.interpolate(emb, (256, 256), mode="bilinear", align_corners=False)

def compute_pca_rgb(feats):
    _, D, H, W = feats.shape
    X = feats[0].permute(1, 2, 0).reshape(-1, D).numpy()
    X = PCA(3).fit_transform(X)
    X = (X - X.min(0)) / (X.max(0) - X.min(0) + 1e-8)
    return X.reshape(H, W, 3)

# ===========================================================
# SAMPLE ONE IMAGE FROM EACH COUNTRY
# ===========================================================

requested_countries = [c.strip() for c in args.country_names.split(";")]

sample_paths = []

for cname in requested_countries:
    cdir = ftw_root / cname
    if not cdir.exists():
        raise RuntimeError(f"Country folder not found: {cdir}")

    window = random.choice(["window_a", "window_b"])
    tifs = sorted((cdir / "s2_images" / window).glob("*.tif"))
    if len(tifs) == 0:
        raise RuntimeError(f"No tif images in {cdir}/s2_images/{window}")

    img_path = random.choice(tifs)
    sample_paths.append((cname, window, img_path))

# ===========================================================
# MAKE GRID
# ===========================================================

num_rows = len(sample_paths)          # countries
num_cols = 1 + len(model_names)       # image + models

fig, axes = plt.subplots(
    num_rows, num_cols,
    figsize=(2.4 * num_cols, 2.4 * num_rows),
    constrained_layout=False
)

# CRUCIAL: REDUCE VERTICAL SPACE
plt.subplots_adjust(
    top=0.93,
    left=0.18,
    right=0.99,
    bottom=0.02,
    wspace=0.02,
    hspace=-0.35, 
)

# ===========================================================
# BUILD THE COLLAGE
# ===========================================================

for row_idx, (cname, window, tif_path) in enumerate(sample_paths):

    rgb = load_rgb(tif_path)
    stem = tif_path.stem

    # COLUMN 0 — Raw Image
    ax = axes[row_idx, 0]
    ax.imshow(rgb)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    # Capitalize country names
    ax.set_ylabel(cname.capitalize(), fontsize=18)

    if row_idx == 0:
        ax.set_title("Image", fontsize=18)

    # MODEL COLUMNS
    for col_idx, model in enumerate(model_names, start=1):

        emb = load_embedding(model, cname, window, stem)
        emb_up = reshape_embedding(emb, model)
        pca_rgb = compute_pca_rgb(emb_up)

        ax = axes[row_idx, col_idx]
        ax.imshow(pca_rgb)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

        if row_idx == 0:
            ax.set_title(MODEL_TITLES[model], fontsize=18)

# ===========================================================
# SAVE
# ===========================================================

out_path = "pca_features.pdf"
fig.savefig(out_path, dpi=200, bbox_inches="tight")
plt.close(fig)

print("Saved:", out_path)
