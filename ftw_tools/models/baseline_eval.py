import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch.cli import LightningCLI
from torch.utils.data import DataLoader
from torchgeo.trainers import BaseTask
from torchmetrics import JaccardIndex, MetricCollection, Precision, Recall
from tqdm import tqdm

from ftw_tools.postprocess.metrics import get_object_level_metrics
from ftw_tools.torchgeo.datasets import FTW
from ftw_tools.torchgeo.trainers import CustomSemanticSegmentationTask
from box import Box
import yaml

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
    """Return model input based on run-time input_type, NOT model_type."""

    # Case 1 — Feature precomputed input
    if input_type == "features":
        return batch["feat"].to(device)

    # Case 2 — Any image mode: images / images_noaug / clay metadata style
    if input_type in ("images", "images_noaug"):
        # CLAY structured model (metadata-aware)
        if "time" in batch and "latlon" in batch:
            return {
                "platform": batch["platform"],
                "image": batch["image"].to(device),
                "time": batch["time"].to(device),
                "latlon": batch["latlon"].to(device),
                "gsd": batch["gsd"].to(device),
                "waves": batch["waves"].to(device),
            }
        # Other backbones: simple tensor input
        return batch["image"].to(device)

    raise ValueError(f"Unsupported input_type={input_type}")


# ----------------------------------------
    # 2️⃣ Extract run_name & log_mode
    # ----------------------------------------
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
    backbone,
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
    feat_root=None,
    encoder_ckpt_path=None,
):
    """Command to test the model (FINAL UPDATED VERSION)."""

    print("Running test command")
    if gpu is None:
        gpu = -1

    # -----------------------------
    # Device
    # -----------------------------
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() and gpu >= 0 else "cpu")

    # -----------------------------
    # Load checkpoint
    # -----------------------------
    print("Loading model...")
    tic = time.time()

    trainer = CustomSemanticSegmentationTask.load_from_checkpoint(
        model_path, map_location="cpu", strict=False
    )
    trainer.eval()

    saved_model_type = trainer.hparams.get("model", "unet")
    saved_backbone   = trainer.hparams.get("backbone", None)
    print(f"  → saved_model_type={saved_model_type}, saved_backbone={saved_backbone}")

    if input_type == "images_noaug": #This is for GFM experiments
        model_type = "gfm"
        backbone_name = backbone
        preprocessing = backbone
    elif input_type == "images": #This is for non-GFM experiments
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

    # -------------------------------------------------------
    # Decoder is always stored in trainer.model
    # -------------------------------------------------------
    decoder = trainer.model.to(device).eval()

    # -------------------------------------------------------
    # ENCODER 
    # -------------------------------------------------------
    encoder = None

    if input_type == "features":
        print("→ Feature mode: encoder NOT required.")
        encoder = None

    elif model_type != "gfm":
        print("→ pretrained / UNET / baseline model: encoder NOT required if experimented on precomputed features on images with unet.")
        encoder = None

    else:
        # -------------------------
        # CASE 1 — Images-trained GFM checkpoint with encoder stored
        # -------------------------
        if hasattr(trainer, "backbone"):
            encoder = trainer.backbone.to(device).eval()
            print("→ Using encoder stored inside checkpoint")

        # -------------------------
        # CASE 2 — Feature-trained decoder: must rebuild encoder
        # -------------------------
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

    print(f"Model loaded in {time.time() - tic:.2f}s")

    # ---------------------------------------------
    # Build FTW dataset
    # ---------------------------------------------
    if countries == ("all",):
        countries = FULL_DATA_COUNTRIES
    
    # CLAY metadata file
    metadata_path = None
    if preprocessing == "clay":
        repo_root = Path(__file__).resolve().parents[2]
        metadata_path = str(repo_root / "configs" / "metadata.yaml")
        print(f"Using CLAY metadata file: {metadata_path}")

    print("Creating dataloader...")
    tic = time.time()

    ds = FTW(
        root=dir,
        countries=countries,
        split=test_split,
        load_boundaries=test_on_3_classes,
        temporal_options=temporal_options,
        swap_order=swap_order,
        input_type=input_type,
        feat_root=feat_root,
        preprocessing=preprocessing,
        metadata_path=metadata_path,
    )

    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=12)
    print(f"  → Loaded {len(ds)} samples in {time.time() - tic:.2f}s")

    # ---------------------------------------------
    # Metrics
    # ---------------------------------------------
    if test_on_3_classes:
        metrics = MetricCollection([
            JaccardIndex(task="multiclass", average="none", num_classes=3, ignore_index=3),
            Precision(task="multiclass", average="none", num_classes=3, ignore_index=3),
            Recall(task="multiclass", average="none", num_classes=3, ignore_index=3),
        ]).to(device)
    else:
        metrics = MetricCollection([
            JaccardIndex(task="multiclass", average="none", num_classes=2, ignore_index=3),
            Precision(task="multiclass", average="none", num_classes=2, ignore_index=3),
            Recall(task="multiclass", average="none", num_classes=2, ignore_index=3),
        ]).to(device)

    # ---------------------------------------------
    # Eval loop
    # ---------------------------------------------
    all_tps = all_fps = all_fns = 0
    num_classes = 3 if model_predicts_3_classes else 2

    for batch in tqdm(dl):
        masks = batch["mask"].to(device)

        with torch.inference_mode():
            # SAM-2 special handling
            if saved_model_type == "sam2":
                # SAM-2 needs window_a, window_b, points, point_labels
                window_a = batch["window_a"].to(device) / 255.0  # [B, 3, H, W]
                window_b = batch["window_b"].to(device) / 255.0  # [B, 3, H, W]
                field_mask = batch["field_mask"].to(device)  # [B, H, W], binary
                points = batch.get("points", None)  # [B, N, 2] or None
                point_labels = batch.get("point_labels", None)  # [B, N] or None
                
                # Skip if no points (empty masks)
                if points is None or points.shape[1] == 0:
                    # Use dummy predictions (all background)
                    logits = torch.zeros((field_mask.shape[0], 2, *field_mask.shape[1:]), device=device)
                else:
                    # Use trainer's forward logic
                    model = decoder  # decoder is the SAM2VideoPredictor
                    
                    # Resize images to model.image_size
                    target_size = model.image_size
                    orig_H, orig_W = window_b.shape[-2:]
                    
                    window_a_resized = F.interpolate(
                        window_a, size=(target_size, target_size), mode="bilinear", align_corners=False
                    )
                    window_b_resized = F.interpolate(
                        window_b, size=(target_size, target_size), mode="bilinear", align_corners=False
                    )
                    
                    # Temporal memory (no grad)
                    with torch.no_grad():
                        _ = model.forward_image(window_a_resized)
                    
                    # Forward on window_b
                    feats = model.forward_image(window_b_resized)
                    
                    # Normalize points
                    points_norm = points.clone()
                    scale_x = target_size / orig_W
                    scale_y = target_size / orig_H
                    points_norm[:, :, 0] *= scale_x
                    points_norm[:, :, 1] *= scale_y
                    points_norm[:, :, 0] /= target_size
                    points_norm[:, :, 1] /= target_size
                    points_norm *= model.image_size
                    
                    # Prepare mask prompt
                    field_mask_resized = F.interpolate(
                        field_mask.unsqueeze(1), size=(target_size, target_size), mode="nearest"
                    )
                    mask_prompt = F.interpolate(
                        field_mask_resized, size=(model.image_size // 4, model.image_size // 4), mode="nearest"
                    )
                    
                    # Encode prompts
                    sparse, dense = model.sam_prompt_encoder(
                        points=(points_norm, point_labels), boxes=None, masks=mask_prompt
                    )
                    
                    # Get high-res features
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
                    
                    # Apply sigmoid and upsample
                    pred = torch.sigmoid(low_res_masks[:, 0])
                    pred = F.interpolate(
                        pred.unsqueeze(1), size=(orig_H, orig_W), mode="bilinear", align_corners=False
                    ).squeeze(1)
                    
                    # Convert binary probabilities to 2-class logits format
                    # pred is [B, H, W] with values 0-1
                    # Convert to logits: [B, 2, H, W]
                    # Class 0 (background): 1 - pred
                    # Class 1 (field): pred
                    logits = torch.stack([1 - pred, pred], dim=1)  # [B, 2, H, W]
                
                # Get class predictions
                outputs = logits.argmax(dim=1)  # [B, H, W] with values 0 or 1
                
                # Use binary field mask as target
                masks = field_mask.long()  # [B, H, W] with values 0 or 1
                
            else:
                # Original logic for other models
                x = prepare_input(batch, input_type, device)
                
                if model_type == "gfm":
                    feats = encoder(x)
                    logits = decoder(feats)
                elif model_type == "pretrained":
                    logits = decoder(x)
                else:
                    logits = decoder(x)

                logits = logits[:, :num_classes, :, :]
                outputs = logits.argmax(dim=1)

                # 3 → 2 class projection
                if model_predicts_3_classes:
                    mapped = torch.zeros_like(outputs)
                    mapped[outputs == 1] = 1
                    outputs = mapped
                else:
                    if test_on_3_classes:
                        raise ValueError("Model predicts 2 classes but test_on_3_classes=True")

        metrics.update(outputs, masks)

        # Object metrics
        out_np = outputs.cpu().numpy().astype(np.uint8)
        mask_np = masks.cpu().numpy().astype(np.uint8)

        for i in range(len(out_np)):
            t, f, n = get_object_level_metrics(mask_np[i], out_np[i], iou_threshold)
            all_tps += t
            all_fps += f
            all_fns += n

    # ---------------------------------------------
    # Aggregate results
    # ---------------------------------------------
    results   = metrics.compute()
    pixel_iou = results["MulticlassJaccardIndex"][1].item()
    pixel_prec = results["MulticlassPrecision"][1].item()
    pixel_recall = results["MulticlassRecall"][1].item()

    object_precision = all_tps / (all_tps + all_fps) if (all_tps + all_fps) > 0 else float("nan")
    object_recall    = all_tps / (all_tps + all_fns) if (all_tps + all_fns) > 0 else float("nan")
    object_f1 = (
        2 * object_precision * object_recall / (object_precision + object_recall)
        if not (np.isnan(object_precision) or np.isnan(object_recall))
        and (object_precision + object_recall) > 0
        else float("nan")
    )

    # ---------------------------------------------
    # Print results
    # ---------------------------------------------
    print(f"\nPixel IoU (crop):        {pixel_iou:.4f}")
    print(f"Pixel Precision (crop):  {pixel_prec:.4f}")
    print(f"Pixel Recall (crop):     {pixel_recall:.4f}")
    print(f"Object Precision:        {object_precision:.4f}")
    print(f"Object Recall:           {object_recall:.4f}")
    print(f"Object F1:               {object_f1:.4f}")

    # ---------------------------------------------
    # Save CSV row
    # ---------------------------------------------
    country_str = ";".join(countries)
    if set(countries) == set(FULL_DATA_COUNTRIES):
        country_str = "all"

    if out is not None:
        header = (
            "train_checkpoint,test_countries,pixel_level_iou,"
            "pixel_level_precision,pixel_level_recall,"
            "object_level_precision,object_level_recall,object_level_f1\n"
        )
        file_exists = os.path.exists(out)

        with open(out, "a") as f:
            if not file_exists:
                f.write(header)
            f.write(
                f"{model_path},{country_str},"
                f"{round(pixel_iou,3)},"
                f"{round(pixel_prec,3)},"
                f"{round(pixel_recall,3)},"
                f"{round(object_precision,3)},"
                f"{round(object_recall,3)},"
                f"{round(object_f1,3)}\n"
            )
