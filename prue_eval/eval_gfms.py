from __future__ import annotations

import argparse
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchmetrics import JaccardIndex, MetricCollection, Precision, Recall
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import rasterio.features
import shapely.geometry

from ftw_tools.postprocess.metrics import get_object_level_metrics
from ftw_tools.torchgeo.datasets import FTW
from ftw_tools.torchgeo.trainers import CustomSemanticSegmentationTask
from prue_eval.detections import Detections

FULL_DATA_COUNTRIES = [
    "austria",
    "belgium",
    "cambodia",
    "corsica",
    "croatia",
    "denmark",
    "estonia",
    "finland",
    "france",
    "germany",
    "latvia",
    "lithuania",
    "luxembourg",
    "netherlands",
    "slovakia",
    "slovenia",
    "south_africa",
    "spain",
    "sweden",
    "vietnam",
    "portugal",
]

_COCO_AP_KEYS = [
    "coco_AP",
    "coco_AP50",
    "coco_AP75",
    "coco_APs",
    "coco_APm",
    "coco_APl",
    "coco_AR1",
    "coco_AR10",
    "coco_AR100",
    "coco_ARs",
    "coco_ARm",
    "coco_ARl",
]


def semantic_to_detections(
    mask: np.ndarray,
    score_map: np.ndarray | None = None,
    class_id: int = 0,
    min_area: int = 0,
) -> Detections:
    """Convert semantic mask + score map into instance-style Detections."""
    instance_masks: List[np.ndarray] = []
    xyxys: List[List[float]] = []
    confidences: List[float] = []
    class_ids: List[int] = []
    mask_uint8 = mask.astype(np.uint8)

    for geom, val in rasterio.features.shapes(mask_uint8):
        if val != 1:
            continue
        poly = shapely.geometry.shape(geom)
        if poly.area < min_area:
            continue

        inst_mask = rasterio.features.rasterize(
            [geom],
            out_shape=mask.shape,
            fill=0,
            default_value=1,
            dtype=np.uint8,
        )
        if inst_mask.sum() == 0:
            continue

        instance_masks.append(inst_mask)

        ys, xs = np.where(inst_mask > 0)
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        xyxys.append([float(x_min), float(y_min), float(x_max), float(y_max)])

        conf = float(score_map[inst_mask > 0].mean()) if score_map is not None and score_map[inst_mask > 0].size > 0 else 1.0
        confidences.append(conf)
        class_ids.append(class_id)

    if len(xyxys) == 0:
        return Detections(xyxy=np.empty((0, 4), dtype=np.float32))

    return Detections(
        xyxy=np.asarray(xyxys, dtype=np.float32),
        mask=np.asarray(instance_masks, dtype=np.uint8),
        confidence=np.asarray(confidences, dtype=np.float32),
        class_id=np.asarray(class_ids, dtype=np.int64),
    )


def compute_coco_segm_metrics(
    gt_dets_list: List[Detections],
    pred_dets_list: List[Detections],
    image_ids: List[int],
    image_size: Tuple[int, int] = (256, 256),
) -> Dict[str, float]:
    """COCO segm AP/AR from Detections, aligned with baseline_eval flow."""
    coco_predictions: List[Dict[str, Any]] = []
    coco_gt_annotations: List[Dict[str, Any]] = []
    next_ann_id = 1

    for gt_dets, pred_dets, image_id in zip(gt_dets_list, pred_dets_list, image_ids):
        preds = pred_dets.to_coco_format(image_id, next_ann_id)
        for p in preds:
            p["category_id"] = 0
        coco_predictions.extend(preds)
        next_ann_id += len(preds)

        gts = gt_dets.to_coco_format(image_id, next_ann_id)
        for g in gts:
            g["category_id"] = 0
        coco_gt_annotations.extend(gts)
        next_ann_id += len(gts)

    if not coco_predictions:
        return {}

    height, width = int(image_size[0]), int(image_size[1])
    categories = [{"id": 0, "name": "ag_field", "supercategory": "landcover"}]
    unique_image_ids = set(image_ids)
    for ann in coco_gt_annotations:
        unique_image_ids.add(ann["image_id"])

    images = [
        {"id": iid, "width": width, "height": height, "file_name": f"image_{iid}.png"}
        for iid in sorted(unique_image_ids)
    ]

    try:
        coco_gt = COCO()
        coco_gt.dataset = {
            "info": {"description": "FTW COCO GT", "version": "1.0", "year": 2025},
            "licenses": [],
            "images": images,
            "annotations": coco_gt_annotations,
            "categories": categories,
        }
        coco_gt.createIndex()

        valid_ids = set(coco_gt.imgs.keys())
        filtered = [p for p in coco_predictions if p.get("image_id") in valid_ids]
        coco_dt = coco_gt.loadRes(filtered)
        coco_eval = COCOeval(coco_gt, coco_dt, "segm")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        if not coco_eval.eval:
            return {}
        s = coco_eval.stats
        return {
            "coco_AP": float(s[0]) * 100,
            "coco_AP50": float(s[1]) * 100,
            "coco_AP75": float(s[2]) * 100,
            "coco_APs": float(s[3]) * 100,
            "coco_APm": float(s[4]) * 100,
            "coco_APl": float(s[5]) * 100,
            "coco_AR1": float(s[6]) * 100,
            "coco_AR10": float(s[7]) * 100,
            "coco_AR100": float(s[8]) * 100,
            "coco_ARs": float(s[9]) * 100,
            "coco_ARm": float(s[10]) * 100,
            "coco_ARl": float(s[11]) * 100,
        }
    except Exception:
        return {k: float("nan") for k in _COCO_AP_KEYS}


def prepare_input(batch, input_type, device):
    """Return model input based on run-time input_type, NOT model_type."""
    if input_type == "features":
        return batch["feat"].to(device)

    if input_type in ("images", "images_noaug"):
        if "time" in batch and "latlon" in batch:
            return {
                "platform": batch["platform"],
                "image": batch["image"].to(device),
                "time": batch["time"].to(device),
                "latlon": batch["latlon"].to(device),
                "gsd": batch["gsd"].to(device),
                "waves": batch["waves"].to(device),
            }
        return batch["image"].to(device)

    raise ValueError(f"Unsupported input_type={input_type}")


def run_gfm_eval(
    model_path: str,
    backbone: str | None,
    test_split: str,
    dir: str,
    gpu: int,
    countries: Union[Tuple[str, ...], Sequence[str]],
    iou_threshold: float,
    out: str | None,
    model_predicts_3_classes: bool,
    test_on_3_classes: bool,
    temporal_options: str,
    swap_order: bool,
    input_type: str = "images",
    feat_root: str | None = None,
    encoder_ckpt_path: str | None = None,
    metadata_path: str | None = None,
) -> None:
    """Evaluate checkpoint using a baseline_eval-style inference/metric loop."""
    print("Running GFM eval (baseline_eval-style)")
    if gpu is None:
        gpu = -1

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() and gpu >= 0 else "cpu")

    print("Loading model...")
    tic = time.time()
    trainer = CustomSemanticSegmentationTask.load_from_checkpoint(model_path, map_location="cpu", strict=False)
    trainer.eval()

    saved_model_type = trainer.hparams.get("model", "unet")
    saved_backbone = trainer.hparams.get("backbone", None)
    print(f"  → saved_model_type={saved_model_type}, saved_backbone={saved_backbone}")

    if input_type == "images_noaug":
        model_type = "gfm"
        backbone_name = backbone if backbone is not None else saved_backbone
        preprocessing = backbone
    elif input_type == "images":
        model_type = saved_model_type
        backbone_name = saved_backbone
        preprocessing = "ftw"
    elif input_type == "features":
        model_type = None
        backbone_name = None
        preprocessing = None
    else:
        raise ValueError(f"Unsupported input_type={input_type}")

    print(f"  → model_type={model_type}, backbone={backbone_name}")

    decoder = trainer.model.to(device).eval()
    encoder = None
    if input_type == "features":
        print("→ Feature mode: encoder NOT required.")
    elif model_type != "gfm":
        print("→ pretrained / baseline model: encoder NOT required.")
    else:
        if hasattr(trainer, "backbone"):
            encoder = trainer.backbone.to(device).eval()
            print("→ Using encoder stored inside checkpoint")
        else:
            print("→ Rebuilding encoder via pretrained_factory.get_encoder()")
            from pretrained.pretrained_factory import get_encoder

            encoder = get_encoder(
                model_name=backbone_name,
                device=device,
                weights_path=encoder_ckpt_path,
            )
            encoder.eval()
            for p in encoder.parameters():
                p.requires_grad = False
            print(f"→ Loaded encoder weights from: {encoder_ckpt_path}")
        if encoder is None:
            raise ValueError(
                "GFM images_noaug path requires an encoder. "
                "Provide a full finetuned checkpoint or pass --encoder_ckpt_path for decoder-only checkpoints."
            )

    print(f"Model loaded in {time.time() - tic:.2f}s")

    countries_tuple = tuple(countries)
    if countries_tuple == ("all",):
        countries_tuple = tuple(FULL_DATA_COUNTRIES)

    if metadata_path is None:
        metadata_path = str(Path(__file__).resolve().parents[1] / "configs" / "metadata.yaml")

    print("Creating dataloader...")
    tic = time.time()
    ds = FTW(
        root=dir,
        countries=list(countries_tuple),
        split=test_split,
        preprocessing=preprocessing,
        metadata_path=metadata_path,
        load_boundaries=test_on_3_classes,
        temporal_options=temporal_options,
        swap_order=swap_order,
        input_type=input_type,
        feat_root=feat_root,
    )
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=12)
    print(f"Created dataloader with {len(ds)} samples in {time.time() - tic:.2f}s")

    if test_on_3_classes:
        metrics = MetricCollection(
            [
                JaccardIndex(task="multiclass", average="none", num_classes=3, ignore_index=3),
                Precision(task="multiclass", average="none", num_classes=3, ignore_index=3),
                Recall(task="multiclass", average="none", num_classes=3, ignore_index=3),
            ]
        ).to(device)
    else:
        metrics = MetricCollection(
            [
                JaccardIndex(task="multiclass", average="none", num_classes=2, ignore_index=3),
                Precision(task="multiclass", average="none", num_classes=2, ignore_index=3),
                Recall(task="multiclass", average="none", num_classes=2, ignore_index=3),
            ]
        ).to(device)

    all_tps = 0
    all_fps = 0
    all_fns = 0
    num_classes = 3 if model_predicts_3_classes else 2

    all_gt_dets: List[Detections] = []
    all_pred_dets: List[Detections] = []
    image_ids: List[int] = []
    img_id_counter = 0

    for batch in tqdm(dl):
        x = prepare_input(batch, input_type, device)
        masks = batch["mask"].to(device)

        with torch.inference_mode():
            if model_type == "gfm":
                feats = encoder(x)
                logits = decoder(feats)
            elif model_type == "pretrained":
                logits = decoder(x)
            else:
                logits = decoder(x)
            logits = logits[:, :num_classes, :, :]
            probs = torch.softmax(logits, dim=1)
            outputs = probs.argmax(dim=1)

        if model_predicts_3_classes:
            new_outputs = torch.zeros(outputs.shape[0], outputs.shape[1], outputs.shape[2], device=device)
            new_outputs[outputs == 2] = 0
            new_outputs[outputs == 0] = 0
            new_outputs[outputs == 1] = 1
            outputs = new_outputs
        else:
            if test_on_3_classes:
                raise ValueError("Cannot test on 3 classes when the model was trained on 2 classes")

        metrics.update(outputs, masks)
        out_np = outputs.cpu().numpy().astype(np.uint8)
        mask_np = masks.cpu().numpy().astype(np.uint8)

        if probs.shape[1] > 1:
            crop_probs_np = probs[:, 1, :, :].cpu().numpy()
        else:
            crop_probs_np = probs[:, 0, :, :].cpu().numpy()

        batch_size = len(out_np)
        for i in range(batch_size):
            t, f, n = get_object_level_metrics(mask_np[i], out_np[i], iou_threshold=iou_threshold)
            all_tps += t
            all_fps += f
            all_fns += n

            gt_det = semantic_to_detections(mask_np[i], score_map=None, class_id=0)
            pred_det = semantic_to_detections(out_np[i], score_map=crop_probs_np[i], class_id=0)
            all_gt_dets.append(gt_det)
            all_pred_dets.append(pred_det)
            image_ids.append(img_id_counter)
            img_id_counter += 1

    results = metrics.compute()
    coco_results = compute_coco_segm_metrics(
        all_gt_dets, all_pred_dets, image_ids
    )

    pixel_iou = results["MulticlassJaccardIndex"][1].item()
    pixel_prec = results["MulticlassPrecision"][1].item()
    pixel_recall = results["MulticlassRecall"][1].item()

    object_precision = all_tps / (all_tps + all_fps) if (all_tps + all_fps) > 0 else float("nan")
    object_recall = all_tps / (all_tps + all_fns) if (all_tps + all_fns) > 0 else float("nan")
    object_f1 = (
        2 * object_precision * object_recall / (object_precision + object_recall)
        if not (np.isnan(object_precision) or np.isnan(object_recall)) and (object_precision + object_recall) > 0
        else float("nan")
    )

    print(f"\nPixel IoU (crop):        {pixel_iou:.4f}")
    print(f"Pixel Precision (crop):  {pixel_prec:.4f}")
    print(f"Pixel Recall (crop):     {pixel_recall:.4f}")
    print(f"Object Precision:        {object_precision:.4f}")
    print(f"Object Recall:           {object_recall:.4f}")
    print(f"Object F1:               {object_f1:.4f}")

    coco_map_50_95 = coco_results.get("coco_AP", float("nan"))
    coco_map_50 = coco_results.get("coco_AP50", float("nan"))

    print(f"COCO mAP@0.5:        {coco_map_50:.4f}")
    print(f"COCO mAP@0.5:0.95:   {coco_map_50_95:.4f}")

    if coco_results:
        print("\nCOCO Metrics (pycocotools COCOeval, iouType=segm):")
        for k, v in coco_results.items():
            if isinstance(v, float):
                print(f"{k:20}: {v:.2f}")
            else:
                print(f"{k:20}: {v}")

    country_str = ";".join(countries_tuple)
    if set(countries_tuple) == set(FULL_DATA_COUNTRIES):
        country_str = "all"

    if out is not None:
        header = (
            "train_checkpoint,test_countries,pixel_level_iou,"
            "pixel_level_precision,pixel_level_recall,"
            "object_level_precision,object_level_recall,object_level_f1,"
            "coco_map_50,coco_map_50_95\n"
        )
        file_exists = os.path.exists(out)

        with open(out, "a") as f:
            if not file_exists:
                f.write(header)

            f.write(
                f"{model_path},{country_str},"
                f"{round(pixel_iou, 3)},"
                f"{round(pixel_prec, 3)},"
                f"{round(pixel_recall, 3)},"
                f"{round(object_precision, 3)},"
                f"{round(object_recall, 3)},"
                f"{round(object_f1, 3)},"
                f"{round(coco_map_50, 3)},"
                f"{round(coco_map_50_95, 3)}\n"
            )


# Alias matching baseline_eval naming
test = run_gfm_eval
evaluate = run_gfm_eval


def main() -> None:
    parser = argparse.ArgumentParser(description="GFM-style FTW eval; COCO via pycocotools COCOeval (segm)")
    parser.add_argument("--model", "-m", required=True, help="Checkpoint path")
    parser.add_argument("--backbone", default=None)
    parser.add_argument("--test_split", default="test")
    parser.add_argument("--dir", default="./data/ftw", help="FTW dataset root")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--countries", "-c", nargs="+", required=True, help='Country names or a single "all"')
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    parser.add_argument("--out", "-o", default="metrics.csv")
    parser.add_argument("--model_predicts_3_classes", action="store_true")
    parser.add_argument("--test_on_3_classes", action="store_true")
    parser.add_argument("--temporal_options", default="stacked")
    parser.add_argument("--swap_order", action="store_true")
    parser.add_argument("--input_type", default="images")
    parser.add_argument("--feat_root", default=None)
    parser.add_argument("--encoder_ckpt_path", default=None)
    parser.add_argument("--metadata_path", default=None, help="Path to metadata.yaml (defaults to repo configs/metadata.yaml)")

    args = parser.parse_args()
    countries = args.countries
    if len(countries) == 1 and countries[0].lower() == "all":
        countries_t = ("all",)
    else:
        countries_t = tuple(countries)

    run_gfm_eval(
        model_path=args.model,
        backbone=args.backbone,
        test_split=args.test_split,
        dir=args.dir,
        gpu=args.gpu,
        countries=countries_t,
        iou_threshold=args.iou_threshold,
        out=args.out,
        model_predicts_3_classes=args.model_predicts_3_classes,
        test_on_3_classes=args.test_on_3_classes,
        temporal_options=args.temporal_options,
        swap_order=args.swap_order,
        input_type=args.input_type,
        feat_root=args.feat_root,
        encoder_ckpt_path=args.encoder_ckpt_path,
        metadata_path=args.metadata_path,
    )


if __name__ == "__main__":
    main()
