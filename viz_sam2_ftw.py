#!/usr/bin/env python3
"""
Visualize SAM-2 predictions on FTW:
- window_a (input A)
- window_b (input B)
- ground truth binary mask (field)
- predicted mask
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ftw_tools.torchgeo.datasets import FTW
from ftw_tools.torchgeo.trainers import CustomSemanticSegmentationTask


def load_model(model_path: str, device: torch.device):
    """Load SAM-2 LightningModule and get underlying SAM-2 model."""
    print(f"Loading checkpoint: {model_path}")
    task = CustomSemanticSegmentationTask.load_from_checkpoint(
        model_path, map_location="cpu", strict=False
    )
    task.eval()
    task.to(device)

    if task.hparams.get("model") != "sam2":
        raise ValueError(f"Checkpoint model type is {task.hparams.get('model')}, expected 'sam2'")

    model = task.model.to(device).eval()
    return task, model


def build_dataset(data_root: str, country: str, split: str = "val"):
    """Build FTW dataset in SAM-2 mode for a single country."""
    ds = FTW(
        root=data_root,
        countries=[country],
        split=split,
        load_boundaries=False,         # 2-class masks for simplicity
        temporal_options="sam2",       # SAM-2 mode
        swap_order=False,
        input_type="images",
        feat_root=None,
        preprocessing="none",
        metadata_path=None,
        sam2_max_image_size=1024,
        sam2_num_points=3,             # more points for visualization
    )
    print(f"Dataset: {len(ds)} samples from {country} ({split})")
    return ds


@torch.no_grad()
def sam2_predict_on_batch(model, batch, device: torch.device):
    """
    Run SAM-2 on a batch from FTW SAM-2 dataset.
    Returns:
        window_a, window_b, field_mask, pred_prob
    Shapes:
        window_a, window_b: [B, 3, H, W] (float, 0-1)
        field_mask: [B, H, W] (0/1)
        pred_prob: [B, H, W] (0-1)
    """
    window_a = batch["window_a"].to(device) / 255.0
    window_b = batch["window_b"].to(device) / 255.0
    field_mask = batch["field_mask"].to(device)
    points = batch.get("points", None)
    point_labels = batch.get("point_labels", None)

    if points is not None:
        points = points.to(device)
    if point_labels is not None:
        point_labels = point_labels.to(device)

    model = model

    orig_H, orig_W = window_b.shape[-2:]
    target_size = model.image_size

    # Resize to SAM-2 resolution
    img_a_resized = F.interpolate(
        window_a,
        size=(target_size, target_size),
        mode="bilinear",
        align_corners=False,
    )
    img_b_resized = F.interpolate(
        window_b,
        size=(target_size, target_size),
        mode="bilinear",
        align_corners=False,
    )

    # Build temporal memory with window_a
    _ = model.forward_image(img_a_resized)

    # Forward on window_b
    feats = model.forward_image(img_b_resized)

    # Normalize points
    if points is None or points.shape[1] == 0:
        # Dummy all-background prediction
        pred = torch.zeros((field_mask.shape[0], orig_H, orig_W), device=device)
        return window_a, window_b, field_mask, pred

    points_norm = points.clone()
    scale_x = target_size / orig_W
    scale_y = target_size / orig_H
    points_norm[:, :, 0] *= scale_x
    points_norm[:, :, 1] *= scale_y
    points_norm[:, :, 0] /= target_size
    points_norm[:, :, 1] /= target_size
    points_norm *= model.image_size

    # Mask prompt
    field_mask_resized = F.interpolate(
        field_mask.unsqueeze(1),
        size=(target_size, target_size),
        mode="nearest",
    )
    mask_prompt = F.interpolate(
        field_mask_resized,
        size=(model.image_size // 4, model.image_size // 4),
        mode="nearest",
    )

    # Encode prompts
    sparse, dense = model.sam_prompt_encoder(
        points=(points_norm, point_labels),
        boxes=None,
        masks=mask_prompt,
    )

    # High-res features if available
    high_res_features = None
    if model.use_high_res_features_in_sam:
        high_res_features = feats["backbone_fpn"][:2]

    # Decode masks
    low_res_masks, _, _, _ = model.sam_mask_decoder(
        image_embeddings=feats["vision_features"],
        image_pe=model.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=False,
        repeat_image=False,
        high_res_features=high_res_features,
    )

    # Probabilities and upsampling
    pred = torch.sigmoid(low_res_masks[:, 0])
    pred = F.interpolate(
        pred.unsqueeze(1),
        size=(orig_H, orig_W),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)

    return window_a, window_b, field_mask, pred

def visualize_sample(window_a, window_b, field_mask, pred, idx: int = 0, save_path: str = None):
    """Plot window_a, window_b, GT mask, prediction mask, and prediction overlay."""
    wa = window_a[idx].cpu().permute(1, 2, 0).numpy()  # [H, W, 3]
    wb = window_b[idx].cpu().permute(1, 2, 0).numpy()
    gt = field_mask[idx].cpu().numpy()
    pr = pred[idx].cpu().numpy()
    pr_bin = (pr > 0.5).astype(np.float32)

    wa = np.clip(wa, 0.0, 1.0)
    wb = np.clip(wb, 0.0, 1.0)

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    axes[0].imshow(wa)
    axes[0].set_title("Window A")
    axes[0].axis("off")

    axes[1].imshow(wb)
    axes[1].set_title("Window B")
    axes[1].axis("off")

    axes[2].imshow(gt, cmap="gray")
    axes[2].set_title("GT field mask")
    axes[2].axis("off")

    # pr = pred[idx].cpu().numpy()
    # pr_bin = (pr > 0.5).astype(np.float32)

    # axes[3].imshow(pr, cmap="gray", vmin=0, vmax=1)
    # axes[3].set_title("Pred prob (0–1)")
    # axes[3].axis("off")

    # axes[4].imshow(pr_bin, cmap="gray")
    # axes[4].set_title("Pred mask (binary)")
    # axes[4].axis("off")

    axes[3].imshow(pr_bin, cmap="gray")
    axes[3].set_title("Pred mask (binary)")
    axes[3].axis("off")

    axes[4].imshow(wb)
    axes[4].imshow(pr_bin, cmap="Reds", alpha=0.5)
    axes[4].set_title("Prediction overlay")
    axes[4].axis("off")

    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize SAM-2 FTW predictions")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to SAM-2 FTW checkpoint (.ckpt)",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/u/gmuhawenayo/datasets/FTW-Dataset/ftw",
        help="Path to FTW dataset root",
    )
    parser.add_argument(
        "--country",
        type=str,
        default="germany",
        help="Country to visualize (e.g., germany, france)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        help="Split to use: train / val / test",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of samples to visualize",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="sam2_ftw/viz",
        help="Output directory for saved figures",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    task, model = load_model(args.model, device)
    ds = build_dataset(args.data_root, args.country, split=args.split)
    dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=4)

    for i, batch in enumerate(dl):
        if i >= args.num_samples:
            break

        window_a, window_b, field_mask, pred = sam2_predict_on_batch(model, batch, device)

        save_path = f"{args.out_dir}/{args.country}_{args.split}_sample_{i}.png"
        visualize_sample(window_a, window_b, field_mask, pred, idx=0, save_path=save_path)


if __name__ == "__main__":
    main()

# python viz_sam2_ftw.py   --model /u/gmuhawenayo/projects/PRUE-CVPR/ftw-prue/logs/sam2-ftw-rebuttal/FTW-project/5294stag/checkpoints/last.ckpt   --data_root /u/gmuhawenayo/datasets/FTW-Dataset/ftw   --country cambodia   --split test   --num_samples 5   --out_dir sam2_ftw/viz