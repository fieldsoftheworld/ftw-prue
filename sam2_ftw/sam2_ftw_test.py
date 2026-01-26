#!/usr/bin/env python3
"""
SAM-2 Temporal Test Script (COMPATIBLE WITH TRAINING)

- Uses forward_image (not video API)
- Uses window_a -> window_b temporal memory
- Uses points + mask prompts
- Uses high-res features
"""

import os
import sys
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
from tqdm import tqdm

# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

sam2_repo_path = Path("/u/gmuhawenayo/projects/sam2")
sys.path.insert(0, str(sam2_repo_path))

from sam2.build_sam import build_sam2_video_predictor

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
DATA_ROOT = "/u/gmuhawenayo/datasets/FTW-Dataset/ftw"
MODEL_CFG = "sam2_hiera_s.yaml"
BASE_CHECKPOINT = "/u/gmuhawenayo/projects/sam2/checkpoints/sam2.1_hiera_small.pt"
FINE_TUNED_DECODER = "sam2_ftw/mask_decoder_final.pt"

COUNTRIES = ["france"]
MAX_IMAGE_SIZE = 1024
OUTPUT_DIR = "sam2_ftw/results"
NUM_SAMPLES = 30

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------
def load_ftw_data(root, countries, split="test"):
    data = []
    for c in countries:
        base = os.path.join(root, c)
        df = gpd.read_parquet(os.path.join(base, f"chips_{c}.parquet"))
        df = df[df["split"] == split]
        for idx in df["aoi_id"].values:
            a = os.path.join(base, "s2_images/window_a", f"{idx}.tif")
            b = os.path.join(base, "s2_images/window_b", f"{idx}.tif")
            m = os.path.join(base, "label_masks/semantic_3class", f"{idx}.tif")
            if all(os.path.exists(x) for x in [a, b, m]):
                data.append({"a": a, "b": b, "m": m, "id": idx})
    return data


def read_sample(ent):
    def read_img(p):
        with rasterio.open(p) as src:
            x = src.read()[:3].transpose(1, 2, 0)
            return x.astype(np.uint8)

    img_a = read_img(ent["a"])
    img_b = read_img(ent["b"])

    with rasterio.open(ent["m"]) as src:
        mask = src.read(1)

    r = min(MAX_IMAGE_SIZE / img_a.shape[0], MAX_IMAGE_SIZE / img_a.shape[1])
    h, w = int(img_a.shape[0] * r), int(img_a.shape[1] * r)

    img_a = cv2.resize(img_a, (w, h))
    img_b = cv2.resize(img_b, (w, h))
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    field_mask = ((mask == 1) | (mask == 2)).astype(np.float32)
    return img_a, img_b, mask, field_mask


def sample_points(mask, n=3):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None, None

    idx = np.random.choice(len(xs), size=min(n, len(xs)), replace=False)
    pts = np.stack([xs[idx], ys[idx]], axis=1)
    lbls = np.ones(len(pts), dtype=np.int32)
    return pts, lbls


def compute_iou(pred, gt):
    pred = (pred > 0.5)
    gt = gt.astype(bool)
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    return inter / (union + 1e-8)

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_data = load_ftw_data(DATA_ROOT, COUNTRIES, "test")
    test_data = test_data[:NUM_SAMPLES]

    # Build model
    model = build_sam2_video_predictor(MODEL_CFG, checkpoint=None, device=device)

    base_ckpt = torch.load(BASE_CHECKPOINT, map_location="cpu")
    model.load_state_dict(base_ckpt["model"], strict=False)

    # Load fine-tuned decoder ONLY
    decoder_state = torch.load(FINE_TUNED_DECODER, map_location="cpu")
    model.sam_mask_decoder.load_state_dict(decoder_state, strict=False)

    model.eval()

    ious = []

    for ent in tqdm(test_data):
        img_a, img_b, gt_3c, gt_bin = read_sample(ent)
        pts, lbls = sample_points(gt_bin)

        if pts is None:
            continue

        img_a = torch.from_numpy(img_a).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255
        img_b = torch.from_numpy(img_b).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255
        gt_bin_t = torch.from_numpy(gt_bin).unsqueeze(0).to(device)

        with torch.no_grad():
            # Temporal memory
            _ = model.forward_image(img_a)

            feats = model.forward_image(img_b)

            H, W = img_b.shape[-2:]
            pts_n = pts.astype(np.float32)
            pts_n[:, 0] /= W
            pts_n[:, 1] /= H
            pts_n *= model.image_size

            pts_t = torch.tensor(pts_n).unsqueeze(0).to(device)
            lbls_t = torch.tensor(lbls).unsqueeze(0).to(device)

            mask_prompt = F.interpolate(
                gt_bin_t.unsqueeze(0),
                size=(model.image_size // 4, model.image_size // 4),
                mode="nearest",
            )

            sparse, dense = model.sam_prompt_encoder(
                points=(pts_t, lbls_t),
                boxes=None,
                masks=mask_prompt,
            )

            high_res = feats["backbone_fpn"][:2]

            low_res, _, _, _ = model.sam_mask_decoder(
                image_embeddings=feats["vision_features"],
                image_pe=model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense,
                multimask_output=False,
                repeat_image=False,
                high_res_features=high_res,
            )

            pred = torch.sigmoid(low_res[:, 0])
            pred = F.interpolate(pred.unsqueeze(0),
                                  size=gt_bin.shape,
                                  mode="bilinear").squeeze().cpu().numpy()

        iou = compute_iou(pred, gt_bin)
        ious.append(iou)

        # Visualization
        plt.figure(figsize=(15, 4))
        plt.subplot(1, 4, 1); plt.title("Window B"); plt.imshow(img_b[0].permute(1,2,0).cpu()); plt.axis("off")
        plt.subplot(1, 4, 2); plt.title("GT"); plt.imshow(gt_bin, cmap="gray"); plt.axis("off")
        plt.subplot(1, 4, 3); plt.title(f"Pred (IoU={iou:.3f})"); plt.imshow(pred, cmap="gray"); plt.axis("off")
        plt.subplot(1, 4, 4); plt.title("Overlay"); 
        plt.imshow(img_b[0].permute(1,2,0).cpu()); 
        plt.imshow(pred > 0.5, alpha=0.5, cmap="Reds"); 
        plt.axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{ent['id']}.png"), dpi=150)
        plt.close()

    print(f"\nMean IoU: {np.mean(ious):.4f}")


if __name__ == "__main__":
    main()
