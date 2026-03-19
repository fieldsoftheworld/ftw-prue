import os
import time
from typing import List

import numpy as np
import torch
from lightning.pytorch.cli import LightningCLI
from torch.utils.data import DataLoader
from torchgeo.trainers import BaseTask
from torchmetrics import JaccardIndex, MetricCollection, Precision, Recall
from tqdm import tqdm
import rasterio.features
import shapely.geometry
from ftw_tools.postprocess.detections import Detections
from ftw_tools.postprocess.evaluator import Evaluator

from ftw_tools.postprocess.metrics import get_object_level_metrics
from ftw_tools.torchgeo.preprocess import preprocess
from ftw_tools.torchgeo.datasets import FTW
from ftw_tools.torchgeo.trainers import CustomSemanticSegmentationTask
from box import Box
import yaml
from lightning.pytorch.loggers import WandbLogger

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
    "portugal"
]



def prepare_input(batch, input_type, device):
    """Return model-ready input dict/tensor depending on modality."""
    # Image-only
    if input_type == "images":
        return batch["image"].to(device)
    
    # Feature-only
    if input_type == "features":
        return {"feat": batch["feat"].to(device)}
    
    # Image + Feature fusion
    if "images" in input_type and "features" in input_type:
        return {
            "image": batch["image"].to(device),
            "feat": batch["feat"].to(device),
        }
    
    # Clay model (includes spatiotemporal inputs)
    if "time" in batch and "latlon" in batch:
        res = {
            "platform": "sentinel-2-l2a",
            "image": batch["image"].to(device),
            "time": batch["time"].to(device),
            "latlon": batch["latlon"].to(device),
        }
        if "gsd" in batch:
            res["gsd"] = batch["gsd"].to(device)
        if "waves" in batch:
            res["waves"] = batch["waves"].to(device)
        return res

    raise ValueError(f"Unrecognized input_type: {input_type} or batch keys {batch.keys()}")


def extract_flag(cli_args, flag, default=None):
    """Extract a value like --flag value OR --flag=value"""
    if f"--{flag}" in cli_args:
        idx = cli_args.index(f"--{flag}")
        if idx + 1 < len(cli_args):
            val = cli_args[idx + 1]
            del cli_args[idx:idx + 2]
            return val
    for arg in cli_args:
        if arg.startswith(f"--{flag}="):
            val = arg.split("=", 1)[1]
            cli_args.remove(arg)
            return val
    return default

def semantic_to_detections(
    mask: np.ndarray,
    score_map: np.ndarray | None = None,
    class_id: int = 0,
    min_area: int = 0,
) -> Detections:
    """
    Convert semantic mask + (optional) score map to Detections instance.
    - mask: 2D binary (1 = foreground)
    - score_map: 2D softmax prob (same shape)
    - class_id: default 0 (field)
    """
    instance_masks = []
    xyxys = []
    confidences = []
    class_ids = []

    mask_uint8 = mask.astype(np.uint8)

    for geom, val in rasterio.features.shapes(mask_uint8):
        if val != 1:                 # Skip background
            continue

        poly = shapely.geometry.shape(geom)

        if poly.area < min_area:     # Filter small regions
            continue

        # Rasterize single polygon to instance mask
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

        # Compute bounding box
        ys, xs = np.where(inst_mask > 0)
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        xyxys.append([float(x_min), float(y_min), float(x_max), float(y_max)])

        # Confidence = mean score inside instance
        if score_map is not None:
            conf = float(score_map[inst_mask > 0].mean()) if score_map[inst_mask > 0].size > 0 else 0.0
        else:
            conf = 1.0
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

def fit(config, ckpt_path, cli_args):
    """Command to fit the model."""
    print("Running fit command")

    # Construct the arguments for PyTorch Lightning CLI
    cli_args = ["fit", f"--config={config}"] + list(cli_args)

    # If a checkpoint path is provided, append it to the CLI arguments
    if ckpt_path:
        cli_args += [f"--ckpt_path={ckpt_path}"]

    print(f"CLI arguments: {cli_args}")

    # Best practices for Rasterio environment variables
    rasterio_best_practices = {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "AWS_NO_SIGN_REQUEST": "YES",
        "GDAL_MAX_RAW_BLOCK_CACHE_SIZE": "200000000",
        "GDAL_SWATH_SIZE": "200000000",
        "VSI_CURL_CACHE_SIZE": "200000000",
    }
    os.environ.update(rasterio_best_practices)


    with open(config, "r") as f:
        yaml_cfg = Box(yaml.safe_load(f))

    default_root_dir = yaml_cfg.trainer.default_root_dir

    run_name = extract_flag(cli_args,"run_name", "debug")
    log_mode = extract_flag(cli_args,"log_mode", "disabled")
    project = extract_flag(cli_args,"project", "FTW-project")
    print(f" Project name: {project}, Run name: {run_name}, Log mode: {log_mode}")

    wandb_logger_config = {
    "class_path": "lightning.pytorch.loggers.wandb.WandbLogger",
    "init_args": {
        "project": project,
        "name": run_name,
        "save_dir": default_root_dir,
        "mode": log_mode,
        "log_model": False,
    },
    }

    cli = LightningCLI(
        model_class=BaseTask,
        seed_everything_default=0,
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_kwargs={"overwrite": True},
        trainer_defaults={
            "logger": wandb_logger_config
        },
        args=cli_args,
    )


def test(
    model_path,
    test_split,
    dir,
    gpu,
    countries,
    iou_threshold,
    out,
    model_predicts_3_classes,
    test_on_3_classes,
    temporal_options,
    swap_order,
    input_type="images",
    preprocess_type="ftw",
    feat_root=None,
    backbone=None,
    encoder_ckpt_path=None,
):
    """Command to test the model."""
    print("Running test command")
    if gpu is None:
        gpu = -1

    # Merge `test_model` function into this test command
    if torch.cuda.is_available() and gpu >= 0:
        device = torch.device(f"cuda:{gpu}")
    else:
        device = torch.device("cpu")

    print("Loading model")
    tic = time.time()
    
    kwargs = {}
    if backbone is not None:
        kwargs["backbone"] = backbone
    if encoder_ckpt_path is not None:
        kwargs["weights"] = encoder_ckpt_path
        
    trainer = CustomSemanticSegmentationTask.load_from_checkpoint(
        model_path, map_location="cpu", **kwargs
    )
    model = trainer.eval().to(device)
    print(f"Model loaded in {time.time() - tic:.2f}s")

    print("Creating dataloader")
    tic = time.time()
    
    if countries == ("all",):
        countries = FULL_DATA_COUNTRIES
    
    # import code;code.interact(local=dict(globals(), **locals()));
    ds = FTW(
        root=dir,
        countries=countries,
        split=test_split,
        preprocessing=None if input_type == "features" else (backbone if backbone is not None else preprocess_type),
        metadata_path="/u/subashk/storage/ftw-ablation/FTW-Bakeoff/ftw-baselines-2/configs/metadata.yaml",
        load_boundaries=test_on_3_classes,
        temporal_options=temporal_options,
        swap_order=swap_order,
        input_type=input_type,
        feat_root=feat_root,
    )
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=12)
    print(f"Created dataloader with {len(ds)} samples in {time.time() - tic:.2f}s")
    # import code;code.interact(local=dict(globals(), **locals()));

    if test_on_3_classes:
        metrics = MetricCollection(
            [
                JaccardIndex(
                    task="multiclass", average="none", num_classes=3, ignore_index=3
                ),
                Precision(
                    task="multiclass", average="none", num_classes=3, ignore_index=3
                ),
                Recall(
                    task="multiclass", average="none", num_classes=3, ignore_index=3
                ),
            ]
        ).to(device)
    else:
        metrics = MetricCollection(
            [
                JaccardIndex(
                    task="multiclass", average="none", num_classes=2, ignore_index=3
                ),
                Precision(
                    task="multiclass", average="none", num_classes=2, ignore_index=3
                ),
                Recall(
                    task="multiclass", average="none", num_classes=2, ignore_index=3
                ),
            ]
        ).to(device)

    all_tps = 0
    all_fps = 0
    all_fns = 0

    all_gt_dets: List[Detections] = []
    all_pred_dets: List[Detections] = []
    
    if model_predicts_3_classes:
        num_classes = 3
    else:
        num_classes = 2
    for batch in tqdm(dl):
        x = prepare_input(batch, input_type, device)
       
        masks = batch["mask"].to(device)

        with torch.inference_mode():
            if input_type == "features" and getattr(trainer, "hparams", {}).get("model") == "gfm":
                logits = trainer.model(x)[:, :num_classes, :, :]
            else:
                logits = model(x)[:, :num_classes, :, :]
            probs = torch.softmax(logits, dim=1)
            outputs = probs.argmax(dim=1)

        if model_predicts_3_classes:
            new_outputs = torch.zeros(
                outputs.shape[0], outputs.shape[1], outputs.shape[2], device=device
            )
            new_outputs[outputs == 2] = 0  # Boundary pixels
            new_outputs[outputs == 0] = 0  # Background pixels
            new_outputs[outputs == 1] = 1  # Crop pixels
            outputs = new_outputs
        else:
            if test_on_3_classes:
                raise ValueError(
                    "Cannot test on 3 classes when the model was trained on 2 classes"
                )

        metrics.update(outputs, masks)
        outputs = outputs.cpu().numpy().astype(np.uint8)
        masks = masks.cpu().numpy().astype(np.uint8)

        # Use class 1 (crop) probabilities as foreground scores
        if probs.shape[1] > 1:
            crop_probs_np = probs[:, 1, :, :].cpu().numpy()  # [B, H, W]
        else:
            # In case of a single-channel logits (rare), just use sigmoid-like
            crop_probs_np = probs[:, 0, :, :].cpu().numpy()

        batch_size = len(outputs)
        for i in range(batch_size):
            output = outputs[i]
            mask = masks[i]
            score_map = crop_probs_np[i]

            tps, fps, fns = get_object_level_metrics(
                mask, output, iou_threshold=iou_threshold
            )
            all_tps += tps
            all_fps += fps
            all_fns += fns

            gt_det = semantic_to_detections(mask, score_map=None, class_id=0)
            pred_det = semantic_to_detections(output, score_map=score_map, class_id=0)

            all_gt_dets.append(gt_det)
            all_pred_dets.append(pred_det)

    results = metrics.compute()
    pixel_level_iou = results["MulticlassJaccardIndex"][1].item()
    pixel_level_precision = results["MulticlassPrecision"][1].item()
    pixel_level_recall = results["MulticlassRecall"][1].item()

    if all_tps + all_fps > 0:
        object_precision = all_tps / (all_tps + all_fps)
    else:
        object_precision = float("nan")

    if all_tps + all_fns > 0:
        object_recall = all_tps / (all_tps + all_fns)
    else:
        object_recall = float("nan")
    
    if not (np.isnan(object_precision) or np.isnan(object_recall)) and (object_precision + object_recall) > 0:
        object_f1 = 2 * object_precision * object_recall / (object_precision + object_recall)
    else:
        object_f1 = float("nan")

    evalu = Evaluator(metrics=["coco"])
    eval_results = evalu.evaluate(all_gt_dets, all_pred_dets)
    coco_map_50_95 = eval_results.get("coco_AP", float("nan"))       # AP@[0.5:0.95]
    coco_map_50 = eval_results.get("coco_AP50", float("nan"))   # AP@0.5

    print(f"Pixel IoU (crop):        {pixel_level_iou:.4f}")
    print(f"Pixel Precision (crop):  {pixel_level_precision:.4f}")
    print(f"Pixel Recall (crop):     {pixel_level_recall:.4f}")
    print(f"Object Precision:        {object_precision:.4f}")
    print(f"Object Recall:           {object_recall:.4f}")
    print(f"Object F1:               {object_f1:.4f}")
    print(f"COCO mAP@0.5:        {coco_map_50:.4f}")
    print(f"COCO mAP@0.5:0.95:   {coco_map_50_95:.4f}")

    country_str = ";".join(countries)
    if set(countries) == set(FULL_DATA_COUNTRIES):
        country_str = "all"

    if out is not None:
        if not os.path.exists(out):
            with open(out, "w") as f:
                f.write(
                    "train_checkpoint,test_countries,pixel_level_iou,pixel_level_precision,pixel_level_recall,object_level_precision,object_level_recall,object_level_f1,coco_map_50,coco_map_50_95\n"
                )
        with open(out, "a") as f:
            f.write(
                f"{model_path},{country_str},{round(pixel_level_iou,3)},{round(pixel_level_precision,3)},{round(pixel_level_recall,3)},{round(object_precision,3)},{round(object_recall,3)},{round(object_f1,3)},{round(coco_map_50,3)},{round(coco_map_50_95,3)}\n"
            )
