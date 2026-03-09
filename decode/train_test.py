import os
import csv
import yaml
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib.pyplot as plt

import torchmetrics
import rasterio.features
import shapely.geometry


from fractal_resunet.models.semanticsegmentation.FracTAL_ResUNet import FracTAL_ResUNet_cmtsk as decode_model
from data_module import FTWMultiCountryDataset
from fractal_resunet.nn.loss.custom_aux_loss import MultiTaskLoss

with open("base_config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

experiment_name = cfg["experiment_name"]
save_dir = Path(cfg["save_dir"]) / experiment_name
save_dir.mkdir(parents=True, exist_ok=True)

with open(save_dir / "config.yaml", "w") as f:
    yaml.dump(cfg, f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = decode_model(
    nfilters_init=cfg["model"]["nfilters_init"],
    NClasses=cfg["data"]["n_classes"],
    depth=cfg["model"]["depth"],
    ftdepth=cfg["model"]["ftdepth"],
    psp_depth=cfg["model"]["psp_depth"],
    norm_type=cfg["model"]["norm_type"],
    norm_groups=cfg["model"]["norm_groups"],
    nheads_start=cfg["model"]["nheads_start"],
    in_channels=cfg["data"]["in_channels"],
).to(device)


def make_loader(split):
    dataset = FTWMultiCountryDataset(
        root_dir=cfg["data"]["root_dir"],
        countries=cfg["data"]["countries"],
        split=split,
        load_boundaries=False,
        temporal_option=cfg["data"]["temporal_option"],
        crop_size=tuple(cfg["data"]["crop_size"]),
        num_samples=cfg["data"]["num_samples"],
    )
    return DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=(split == "train"),
        num_workers=cfg["train"]["num_workers"],
    )


train_loader = make_loader("train")
val_loader = make_loader("val")
test_loader = make_loader("test")

criterion = MultiTaskLoss(
    depth=cfg["model"]["depth"],
    seg_weight=cfg["loss"]["seg_weight"],
    bound_weight=cfg["loss"]["bound_weight"],
    dist_weight=cfg["loss"]["dist_weight"],
)

optimizer = optim.Adam(model.parameters(), lr=cfg["train"]["lr"])
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["train"]["num_epochs"])
lrs = []


def to_one_hot(tensor, num_classes, presence_only):
    """
    Convert mask/boundary to one-hot + valid mask.
    - If presence_only=True → pixels == 3 are ignored (mask=0 in valid_mask).
    - If presence_only=False → pixels == 3 are treated as background (class 0).
    """
    valid_mask = (tensor != 3).float() if presence_only else torch.ones_like(tensor, dtype=torch.float32)

    tensor_proc = tensor.clone()
    tensor_proc[tensor_proc == 3] = 0
    if tensor_proc.ndim == 3:
        tensor_proc = tensor_proc.unsqueeze(1)

    one_hot = torch.zeros(
        tensor_proc.size(0),
        num_classes,
        tensor_proc.size(2),
        tensor_proc.size(3),
        dtype=torch.float32,
        device=tensor_proc.device,
    )
    one_hot.scatter_(1, tensor_proc.long(), 1)
    return one_hot, valid_mask.unsqueeze(1)


train_csv = open(save_dir / "train_loss.csv", "w", newline="")
val_csv = open(save_dir / "val_loss.csv", "w", newline="")
train_writer, val_writer = csv.writer(train_csv), csv.writer(val_csv)
train_writer.writerow(["epoch", "total", "seg", "bound", "dist"])
val_writer.writerow(["epoch", "total", "seg", "bound", "dist"])

train_losses_all, val_losses_all = [], []
best_val = float("inf")
patience_counter = 0

try:
    for epoch in range(cfg["train"]["num_epochs"]):
        model.train()
        train_losses, seg_losses, bound_losses, dist_losses = [], [], [], []

        for win_a, win_b, images, mask, boundary, distance in train_loader:
            if cfg["data"]["in_channels"] == 4:
                images = win_a if np.random.randint(0, 10) % 2 == 0 else win_b

            images, mask, boundary, distance = (
                images.to(device),
                mask.to(device),
                boundary.to(device),
                distance.to(device),
            )

            one_hot_mask, valid_mask_segm = to_one_hot(
                mask, num_classes=cfg["data"]["n_classes"], presence_only=cfg["data"]["presence_only"]
            )
            one_hot_boundary, _ = to_one_hot(boundary, num_classes=2, presence_only=cfg["data"]["presence_only"])

            labels_list = [one_hot_mask, one_hot_boundary, distance]

            preds = model(images)
            loss, lseg, lbound, ldist = criterion(preds, labels_list, valid_mask_segm)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            seg_losses.append(lseg.item())
            bound_losses.append(lbound.item())
            dist_losses.append(ldist.item())

        train_loss = np.mean(train_losses)
        train_writer.writerow([epoch, train_loss, np.mean(seg_losses), np.mean(bound_losses), np.mean(dist_losses)])
        train_csv.flush()
        train_losses_all.append(train_loss)

        model.eval()
        val_losses, seg_losses, bound_losses, dist_losses = [], [], [], []
        with torch.no_grad():
            for win_a, win_b, images, mask, boundary, distance in val_loader:
                if cfg["data"]["in_channels"] == 4:
                    images = win_a if np.random.randint(0, 10) % 2 == 0 else win_b

                images, mask, boundary, distance = (
                    images.to(device),
                    mask.to(device),
                    boundary.to(device),
                    distance.to(device),
                )

                one_hot_mask, valid_mask_segm = to_one_hot(
                    mask, num_classes=cfg["data"]["n_classes"], presence_only=cfg["data"]["presence_only"]
                )
                one_hot_boundary, _ = to_one_hot(boundary, num_classes=2, presence_only=cfg["data"]["presence_only"])

                labels_list = [one_hot_mask, one_hot_boundary, distance]
                preds = model(images)

                loss, lseg, lbound, ldist = criterion(preds, labels_list, valid_mask_segm)

                val_losses.append(loss.item())
                seg_losses.append(lseg.item())
                bound_losses.append(lbound.item())
                dist_losses.append(ldist.item())

        val_loss = np.mean(val_losses)
        val_writer.writerow([epoch, val_loss, np.mean(seg_losses), np.mean(bound_losses), np.mean(dist_losses)])
        val_csv.flush()
        val_losses_all.append(val_loss)

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        lrs.append(current_lr)

        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_dir / "best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= cfg["train"]["patience"]:
                break
finally:
    train_csv.close()
    val_csv.close()

plt.figure()
plt.plot(np.arange(len(train_losses_all)), train_losses_all, label="train")
plt.plot(np.arange(len(val_losses_all)), val_losses_all, label="val")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig(save_dir / "loss_plot.png")
plt.close()

plt.figure()
plt.plot(np.arange(len(lrs)), lrs)
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.tight_layout()
plt.savefig(save_dir / "lr_plot.png")
plt.close()

#######################
#### Country wise test
#######################


def get_object_level_metrics(y_true, y_pred, iou_threshold=0.5):
    """Compute TP, FP, FN counts for object-level metrics."""
    if iou_threshold < 0.5:
        raise ValueError("iou_threshold must be >= 0.5")

    y_true_shapes = []
    for geom, val in rasterio.features.shapes(y_true.astype(np.int32)):
        if val == 1:
            y_true_shapes.append(shapely.geometry.shape(geom))

    y_pred_shapes = []
    for geom, val in rasterio.features.shapes(y_pred.astype(np.int32)):
        if val == 1:
            y_pred_shapes.append(shapely.geometry.shape(geom))

    tps, fns = 0, 0
    matched_js = set()

    for y_true_shape in y_true_shapes:
        matching_j = None
        for j, y_pred_shape in enumerate(y_pred_shapes):
            if y_true_shape.intersects(y_pred_shape):
                intersection = y_true_shape.intersection(y_pred_shape)
                union = y_true_shape.union(y_pred_shape)
                iou = intersection.area / union.area
                if iou > iou_threshold:
                    matching_j = j
                    matched_js.add(j)
                    break
        if matching_j is not None:
            tps += 1
        else:
            fns += 1

    fps = len(y_pred_shapes) - len(matched_js)
    return tps, fps, fns


def run_test(model, test_loaders, save_dir, presence_only_countries=None):
    """
    Evaluate per-country metrics and save results to CSV.
    """
    model.eval()
    presence_only_countries = presence_only_countries or []
    results_path = Path(save_dir) / "test_results.csv"

    with open(results_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Test country",
                "Pixel IoU",
                "Pixel precision",
                "Pixel recall",
                "Object precision",
                "Object recall",
            ]
        )

        for country, loader in test_loaders.items():
            metrics = torchmetrics.MetricCollection(
                {
                    "iou": torchmetrics.classification.MulticlassJaccardIndex(num_classes=2, average="none").to(device),
                    "precision": torchmetrics.classification.MulticlassPrecision(num_classes=2, average="none").to(
                        device
                    ),
                    "recall": torchmetrics.classification.MulticlassRecall(num_classes=2, average="none").to(device),
                }
            )

            all_tps, all_fps, all_fns = 0, 0, 0

            with torch.no_grad():
                for win_a, win_b, images, mask, boundary, distance in loader:
                    if cfg["data"]["in_channels"] == 4:
                        images = win_a if np.random.randint(0, 10) % 2 == 0 else win_b
                    images, mask = images.to(device), mask.to(device)

                    preds = model(images)
                    pred_mask = torch.argmax(preds[0], dim=1).cpu().numpy()
                    gt_mask = mask.squeeze(1).cpu().numpy()

                    if country in presence_only_countries:
                        valid_mask = gt_mask != 3
                        gt_mask = np.where(valid_mask, gt_mask, 0)
                        pred_mask = np.where(valid_mask, pred_mask, 0)

                    metrics.update(
                        torch.from_numpy(pred_mask).long().to(device),
                        torch.from_numpy(gt_mask).long().to(device),
                    )

                    for i in range(pred_mask.shape[0]):
                        tps, fps, fns = get_object_level_metrics(gt_mask[i], pred_mask[i])
                        all_tps += tps
                        all_fps += fps
                        all_fns += fns

            results = metrics.compute()
            pixel_iou = results["iou"][1].item()
            pixel_precision = results["precision"][1].item()
            pixel_recall = results["recall"][1].item()

            if all_tps + all_fps > 0:
                object_precision = all_tps / (all_tps + all_fps)
            else:
                object_precision = float("nan")
            if all_tps + all_fns > 0:
                object_recall = all_tps / (all_tps + all_fns)
            else:
                object_recall = float("nan")

            if country in presence_only_countries:
                row = [
                    country,
                    "nan",
                    "nan",
                    round(pixel_recall, 4),
                    "nan",
                    round(object_recall, 4),
                ]
            else:
                row = [
                    country,
                    round(pixel_iou, 4),
                    round(pixel_precision, 4),
                    round(pixel_recall, 4),
                    round(object_precision, 4) if not np.isnan(object_precision) else "nan",
                    round(object_recall, 4) if not np.isnan(object_recall) else "nan",
                ]

            writer.writerow(row)


test_loaders = {}


presence_only_countries = ["brazil", "india", "kenya", "rwanda"]

for country in cfg["data"]["countries"]:
    dataset = FTWMultiCountryDataset(
        root_dir=cfg["data"]["root_dir"],
        countries=[country],
        split="test",
        load_boundaries=False,
        temporal_option=cfg["data"]["temporal_option"],
        crop_size=tuple(cfg["data"]["crop_size"]),
        num_samples=-1,
    )
    test_loaders[country] = DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
    )

run_test(model, test_loaders, save_dir, presence_only_countries)
