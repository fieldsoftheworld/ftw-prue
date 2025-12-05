import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib.pyplot as plt

import torchmetrics
import rasterio.features
import shapely.geometry
import yaml
import numpy as np
import csv
import os



from fractal_resunet.models.semanticsegmentation.FracTAL_ResUNet import FracTAL_ResUNet_cmtsk as decode_model
from data_module import FTWMultiCountryDataset
from fractal_resunet.nn.loss.custom_aux_loss import MultiTaskLoss



with open("logs-decode/exp0910-2classes-2win/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

experiment_name = cfg["experiment_name"]
save_dir = Path(cfg["save_dir"]) / experiment_name

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint_path = save_dir / "best_model.pth"
in_channels = cfg["data"]["in_channels"]
n_classes = cfg["data"]["n_classes"]
batch_size = cfg["train"]["batch_size"]
countries = cfg["data"]["countries"]

model = decode_model(
    nfilters_init=cfg["model"]["nfilters_init"],
    NClasses=n_classes,
    depth=cfg["model"]["depth"],
    ftdepth=cfg["model"]["ftdepth"],
    psp_depth=cfg["model"]["psp_depth"],
    norm_type=cfg["model"]["norm_type"],
    norm_groups=cfg["model"]["norm_groups"],
    nheads_start=cfg["model"]["nheads_start"],
    in_channels=in_channels,
).to(device)

model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

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
    results_path = Path(save_dir) / "test_results_final_flipped.csv"

    with open(results_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Test country",
            "Pixel IoU",
            "Pixel precision",
            "Pixel recall",
            "Object precision",
            "Object recall",
        ])

        for country, loader in test_loaders.items():
            metrics = torchmetrics.MetricCollection({
                "iou": torchmetrics.classification.MulticlassJaccardIndex(num_classes=2, average="none").to(device),
                "precision": torchmetrics.classification.MulticlassPrecision(num_classes=2, average="none").to(device),
                "recall": torchmetrics.classification.MulticlassRecall(num_classes=2, average="none").to(device),
            })

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

presence_only_countries = ["brazil", "india", "kenya", "rwanda"]

test_loaders = {}
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

