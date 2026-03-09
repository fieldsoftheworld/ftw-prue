import argparse
import random
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
import os

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent))

from ftw_tools.torchgeo.datasets import FTW
from ftw_tools.torchgeo.trainers import CustomSemanticSegmentationTask
from ftw_tools.models.baseline_eval import prepare_input


def denormalize(img_tensor, mean, std):
    """
    Denormalize image tensor: (img * std) + mean
    Expects img_tensor of shape [C, H, W]
    mean, std are tensors of shape [C]
    """
    # Ensure mean/std are on same device and proper shape
    if not isinstance(mean, torch.Tensor):
        mean = torch.tensor(mean)
    if not isinstance(std, torch.Tensor):
        std = torch.tensor(std)

    mean = mean.to(img_tensor.device).view(-1, 1, 1)
    std = std.to(img_tensor.device).view(-1, 1, 1)

    return (img_tensor * std) + mean


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Clay Model Predictions")
    parser.add_argument("--country", type=str, required=True, help="Country to visualize")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data_dir", type=str, default="./data/ftw", help="Path to FTW dataset")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=str, default=None, help="Output plot filename")

    args = parser.parse_args()

    device = torch.device(args.device)
    country_name = args.country
    if args.out is None:
        args.out = f"{country_name}_preds.png"

    print(f"Loading checkpoint: {args.ckpt_path}")
    print(f"Device: {device}")

    # Load Model
    # We use CustomSemanticSegmentationTask to load the full lightning module
    task = CustomSemanticSegmentationTask.load_from_checkpoint(args.ckpt_path, map_location=device, strict=False)
    task.eval()
    task.to(device)

    encoder = None
    decoder = task.model

    if hasattr(task, "backbone"):
        encoder = task.backbone
        print("Using encoder from checkpoint")
    else:
        print("Warning: No backbone found in checkpoint. Assuming decoder-only or handled internally.")

    # Prepare Dataset
    # We need metadata mainly for normalization params
    repo_root = Path(__file__).resolve().parent
    metadata_path = str(repo_root / "configs" / "metadata.yaml")

    print(f"Initializing Dataset for {country_name}...")
    ds = FTW(
        root=args.data_dir,
        countries=[country_name],
        split="test",
        preprocessing="clay",
        metadata_path=metadata_path,
        temporal_options="stacked",
        load_boundaries=True,  # Load 3-class masks (BG, Field, Boundary)
    )

    # Get Normalization Params
    preprocessor = ds.preprocessor
    mean = preprocessor.mean
    std = preprocessor.std

    print(f"Dataset size: {len(ds)}")

    # Sample 5 random indices
    if len(ds) < 5:
        indices = list(range(len(ds)))
    else:
        indices = random.sample(range(len(ds)), 5)

    display_indices = []
    print(f"Visualizing indices: {indices}")

    # Custom Colormap for 3 Classes + Ignore
    from matplotlib.colors import ListedColormap, BoundaryNorm

    # 0: Background (Blue), 1: Field (Orange), 2: Boundary (Green), 3: Ignore (Black)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#000000"]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm([0, 1, 2, 3, 4], cmap.N)

    # Visualization Loop
    fig, axes = plt.subplots(5, 4, figsize=(15, 20))
    # Columns: Window A, Window B, GT, Pred

    cols = ["Window A", "Window B", "Ground Truth", "Prediction"]
    for ax, col in zip(axes[0], cols):
        ax.set_title(col)

    for i, idx in enumerate(indices):
        sample = ds[idx]

        # Print paths
        paths = ds.img_filenames[idx]
        print(f"Sample {idx} Paths:")
        print(f"  Window A: {paths['window_a']}")
        print(f"  Window B: {paths['window_b']}")
        print(f"  Mask:     {paths['mask']}")

        # Check unique mask values
        mask_unique = torch.unique(sample["mask"]).numpy()
        print(f"Sample {idx} Mask Unique Values: {mask_unique}")
        display_indices.append(idx)

        # Prepare input for model
        batch = {}
        for k, v in sample.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.unsqueeze(0)
            else:
                batch[k] = [v]

        x = prepare_input(batch, "images", device)

        # Inference
        with torch.no_grad():
            if encoder:
                feats = encoder(x)
                logits = decoder(feats)
            else:
                logits = decoder(x)

            # Logits: [1, NumClasses, H, W]
            preds = torch.argmax(logits, dim=1).cpu().squeeze(0).numpy()

        # Get Images and Masks for Vis
        full_img = sample["image"]  # [8, H, W]
        C_half = full_img.shape[0] // 2

        win_b_norm = full_img[:C_half]
        win_a_norm = full_img[C_half:]

        win_b = denormalize(win_b_norm, mean, std)
        win_a = denormalize(win_a_norm, mean, std)

        def to_vis(t):
            t = t[:3].permute(1, 2, 0).cpu().numpy()
            return t

        def robust_norm(arr):
            p2, p98 = np.percentile(arr, (2, 98))
            return np.clip((arr - p2) / (p98 - p2), 0, 1)

        win_a_vis = robust_norm(to_vis(win_a))
        win_b_vis = robust_norm(to_vis(win_b))

        gt = sample["mask"].cpu().numpy()

        # Plot
        # Window A
        axes[i, 0].imshow(win_a_vis)
        axes[i, 0].axis("off")

        # Window B
        axes[i, 1].imshow(win_b_vis)
        axes[i, 1].axis("off")

        # GT
        # 3 classes: 0, 1, 2. Ignore index 3 handled by not being in map?
        # Usually masks have 0,1,2.
        axes[i, 2].imshow(gt, cmap=cmap, norm=norm, interpolation="nearest")
        axes[i, 2].axis("off")

        # Pred
        axes[i, 3].imshow(preds, cmap=cmap, norm=norm, interpolation="nearest")
        axes[i, 3].axis("off")

    plt.tight_layout()
    os.makedirs("./predictions", exist_ok=True)
    plt.savefig("./predictions/" + args.out)
    print(f"Saved visualization to {args.out}")
