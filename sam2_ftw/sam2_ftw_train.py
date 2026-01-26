#!/usr/bin/env python3
"""
SAM-2 Temporal Training on FTW
- Uses window_a -> window_b as temporal sequence
- Passes BOTH points and masks to prompt encoder
- Fine-tunes ONLY the mask decoder
"""

import os
import sys
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import geopandas as gpd
from tqdm import tqdm
import rasterio

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
CHECKPOINT_PATH = "/u/gmuhawenayo/projects/sam2/checkpoints/sam2.1_hiera_small.pt"
MODEL_CFG = "sam2_hiera_s.yaml"

COUNTRIES = ["france"]
OUTPUT_DIR = "sam2_ftw"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_IMAGE_SIZE = 1024
NO_OF_STEPS = 300
ACCUMULATION_STEPS = 4
LR = 1e-4
WEIGHT_DECAY = 1e-4

# ---------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------
def load_ftw_data(data_root, countries, split="train"):
    data = []
    for country in countries:
        root = os.path.join(data_root, country)
        chips = os.path.join(root, f"chips_{country}.parquet")
        df = gpd.read_parquet(chips)
        df = df[df["split"] == split]

        for idx in df["aoi_id"].values:
            a = os.path.join(root, "s2_images/window_a", f"{idx}.tif")
            b = os.path.join(root, "s2_images/window_b", f"{idx}.tif")
            m = os.path.join(root, "label_masks/semantic_3class", f"{idx}.tif")
            if all(os.path.exists(x) for x in [a, b, m]):
                data.append({"a": a, "b": b, "m": m})
    return data


def read_temporal_sample(data):
    ent = data[np.random.randint(len(data))]

    def read_img(p):
        with rasterio.open(p) as src:
            x = src.read()
            x = x[:3].transpose(1, 2, 0)
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

    # Binary field mask (class 1 + 2)
    field_mask = ((mask == 1) | (mask == 2)).astype(np.float32)

    # Positive point sampling
    ys, xs = np.where(field_mask > 0)
    if len(xs) == 0:
        return None

    idx = np.random.randint(len(xs))
    points = np.array([[xs[idx], ys[idx]]], dtype=np.float32)
    labels = np.array([1], dtype=np.int32)

    return img_a, img_b, field_mask, points, labels


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_data = load_ftw_data(DATA_ROOT, COUNTRIES, "train")
    assert len(train_data) > 0

    # Build predictor
    predictor = build_sam2_video_predictor(
        MODEL_CFG, checkpoint=None, device=device
    )

    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
    state = ckpt["model"] if "model" in ckpt else ckpt
    predictor.load_state_dict(state, strict=False)

    model = predictor

    # -----------------------------------------------------------------
    # FREEZE EVERYTHING EXCEPT MASK DECODER
    # -----------------------------------------------------------------
    for p in model.image_encoder.parameters():
        p.requires_grad = False
    for p in model.sam_prompt_encoder.parameters():
        p.requires_grad = False
    for p in model.sam_mask_decoder.parameters():
        p.requires_grad = True

    model.image_encoder.eval()
    model.sam_prompt_encoder.eval()
    model.sam_mask_decoder.train()

    optimizer = torch.optim.AdamW(
        model.sam_mask_decoder.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    mean_iou = 0.0

    # -----------------------------------------------------------------
    # TRAIN LOOP
    # -----------------------------------------------------------------
    for step in tqdm(range(1, NO_OF_STEPS + 1)):
        sample = read_temporal_sample(train_data)
        if sample is None:
            continue

        img_a, img_b, gt_mask, pts, lbls = sample

        img_a = torch.from_numpy(img_a).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255
        img_b = torch.from_numpy(img_b).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255

        gt_mask = torch.from_numpy(gt_mask).unsqueeze(0).to(device)

        # ---- Temporal memory (NO grad) ----
        with torch.no_grad():
            _ = model.forward_image(img_a)

        # ---- Train on frame B ----
        with torch.cuda.amp.autocast(enabled=(device == "cuda")):
            feats = model.forward_image(img_b)

            H, W = img_b.shape[-2:]
            pts_norm = pts.copy()
            pts_norm[:, 0] /= W
            pts_norm[:, 1] /= H
            pts_norm *= model.image_size

            pts_t = torch.tensor(pts_norm).unsqueeze(0).to(device)
            lbls_t = torch.tensor(lbls).unsqueeze(0).to(device)

            # ---- MASK PROMPT (downsampled GT) ----
            mask_prompt = F.interpolate(
                gt_mask.unsqueeze(0),
                size=(model.image_size // 4, model.image_size // 4),
                mode="nearest",
            )

            sparse, dense = model.sam_prompt_encoder(
                points=(pts_t, lbls_t),
                boxes=None,
                masks=mask_prompt,
            )

            high_res_features = None
            if model.use_high_res_features_in_sam:
                # SAM-2 expects exactly two feature maps
                high_res_features = feats["backbone_fpn"][:2]

            low_res_masks, _, _, _ = model.sam_mask_decoder(
                image_embeddings=feats["vision_features"],
                image_pe=model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense,
                multimask_output=False,
                repeat_image=False,
                high_res_features=high_res_features,
            )

            pred = torch.sigmoid(low_res_masks[:, 0])
            pred = F.interpolate(pred.unsqueeze(0), size=gt_mask.shape[-2:], mode="bilinear").squeeze(0)

            loss = F.binary_cross_entropy(pred, gt_mask)

        scaler.scale(loss / ACCUMULATION_STEPS).backward()

        if step % ACCUMULATION_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        with torch.no_grad():
            bin_pred = (pred > 0.5).float()
            inter = (bin_pred * gt_mask).sum()
            union = bin_pred.sum() + gt_mask.sum() - inter
            iou = inter / (union + 1e-6)
            mean_iou = 0.99 * mean_iou + 0.01 * iou.item()

        if step % 50 == 0:
            print(f"Step {step} | Loss {loss.item():.4f} | mIoU {mean_iou:.4f}")

    torch.save(model.sam_mask_decoder.state_dict(),
               os.path.join(OUTPUT_DIR, "mask_decoder_final.pt"))

    print("✅ Training complete")


if __name__ == "__main__":
    main()
