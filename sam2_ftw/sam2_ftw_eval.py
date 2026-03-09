#!/usr/bin/env python3
"""
SAM-2 Temporal Eval on FTW
- Uses window_a -> window_b as temporal sequence
- Auto-masks Window A (Frame 0)
- Propagates to Window B (Frame 1)
- Compares predictions on Window B with Ground Truth
"""

import os
import sys
from pathlib import Path
import cv2
import numpy as np
import torch
from tqdm import tqdm
from collections import OrderedDict
import pandas as pd
import warnings

# Filter SAM2 C extension warning
warnings.filterwarnings("ignore", message="cannot import name '_C' from 'sam2'")

# Add project root to path
sam2_repo_path = Path("/projects/benq/atwollam/FTW-Bakeoff/specialized_field_models/sam2/sam2")
sys.path.insert(0, str(sam2_repo_path))

# Patch SAM2 internal tqdm to be silent
import sam2.sam2_video_predictor
def silent_tqdm(iterable, *args, **kwargs):
    return iterable
sam2.sam2_video_predictor.tqdm = silent_tqdm

from build_sam_v2 import build_sam2, build_sam2_video_predictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from hydra.core.global_hydra import GlobalHydra
from hydra import initialize

# Local imports
from dataset_v2 import FTW
from metrics_v2 import get_object_level_metrics

DATA_ROOT = "/projects/benq/ftw-data/data/ftw"
CHECKPOINT_PATH = "/projects/benq/atwollam/FTW-Bakeoff/specialized_field_models/sam2/sam2/checkpoints/sam2.1_hiera_small.pt"
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_s.yaml"

OUTPUT_DIR = "sam2_ftw_eval"
os.makedirs(OUTPUT_DIR, exist_ok=True)
CSV_NAME = os.path.join(OUTPUT_DIR, "eval_results.csv")

class Sam2EvalTransform:
    def __init__(self, max_image_size=1024):
        self.max_image_size = max_image_size
        self.img_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.img_std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    def __call__(self, sample):
        # sample has 'window_a', 'window_b', 'mask'
        img_a = sample["window_a"].numpy().transpose(1, 2, 0) # H,W,C
        img_b = sample["window_b"].numpy().transpose(1, 2, 0)
        mask_gt = sample["mask"].numpy() # H,W (GT for Window B)

        orig_h, orig_w = img_a.shape[0], img_a.shape[1]

        # For auto-mask generator, we usually want the image in 0-255 uint8.
        # Preprocessing: dataset might return raw values. 
        # Previous logic: / 3000.0 clip 0-1, * 255 -> uint8.
        img_a_norm = np.clip(img_a / 3000.0, 0, 1)
        img_b_norm = np.clip(img_b / 3000.0, 0, 1) # Normalize img_b too for consistency
        
        img_a_uint8 = (img_a_norm * 255).astype(np.uint8)
        
        # Resize to MAX_IMAGE_SIZE
        r = min(self.max_image_size / orig_h, self.max_image_size / orig_w)
        h, w = int(orig_h * r), int(orig_w * r)
        
        img_a_res = cv2.resize(img_a_norm, (w, h))
        img_b_res = cv2.resize(img_b_norm, (w, h))
        
        # We also need img_a_uint8 resized if we want to batch it perfectly?
        # AutoMaskGenerator works on individual images.
        # But if we return it in batch, they must be same size.
        # So yes, resize img_a_uint8 as well.
        img_a_uint8_res = cv2.resize(img_a_uint8, (w, h))
        
        # Standardize for predictor input
        img_a_tensor = (img_a_res - self.img_mean) / self.img_std
        img_b_tensor = (img_b_res - self.img_mean) / self.img_std
        
        # Stack frames: (2, H, W, C)
        images_np = np.stack([img_a_tensor, img_b_tensor], axis=0)
        
        # Permute to (2, C, H, W)
        images_t = torch.from_numpy(images_np).permute(0, 3, 1, 2).float()
        
        # We need mask_gt resized? No, metrics against orig mask_gt?
        # Typically metrics are against original GT.
        # But DataLoader needs consistent size. 
        # So we resize GT to (w,h) for transport, OR we return indices and load GT later?
        # Or we return mask_gt padded?
        # Let's resize mask GT to (w,h) for transport, but we ideally compare at Orig size.
        # We can store orig_h, orig_w and resize preds back.
        # But mask_gt also needs to be batched.
        mask_gt_res = cv2.resize(mask_gt, (w, h), interpolation=cv2.INTER_NEAREST)
        
        return {
            "images_t": images_t, # (2, C, H, W)
            "img_a_uint8": torch.from_numpy(img_a_uint8_res), # (H, W, 3) -> Tensor for collate
            "mask_gt": torch.from_numpy(mask_gt_res), # (H, W)
            "orig_size": torch.tensor([orig_h, orig_w]),
            # "orig_mask_gt": mask_gt # Cannot batch variable size array easily without custom collate list
        }

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

def init_custom_state(predictor, images_tensor, height, width, device, offload_video_to_cpu=False, offload_state_to_cpu=False):
    """
    Manually initialize inference state with provided image tensor.
    images_tensor: (N, 3, H, W) normalized
    """
    inference_state = {}
    inference_state["images"] = images_tensor
    inference_state["num_frames"] = len(images_tensor)
    inference_state["offload_video_to_cpu"] = offload_video_to_cpu
    inference_state["offload_state_to_cpu"] = offload_state_to_cpu
    inference_state["video_height"] = height
    inference_state["video_width"] = width
    inference_state["device"] = device
    if offload_state_to_cpu:
        inference_state["storage_device"] = torch.device("cpu")
    else:
        inference_state["storage_device"] = device
    
    inference_state["point_inputs_per_obj"] = {}
    inference_state["mask_inputs_per_obj"] = {}
    inference_state["cached_features"] = {}
    inference_state["constants"] = {}
    inference_state["obj_id_to_idx"] = OrderedDict()
    inference_state["obj_idx_to_id"] = OrderedDict()
    inference_state["obj_ids"] = []
    inference_state["output_dict_per_obj"] = {}
    inference_state["temp_output_dict_per_obj"] = {}
    inference_state["frames_tracked_per_obj"] = {}
    
    # Warm up frame 0
    predictor._get_image_feature(inference_state, frame_idx=0, batch_size=1)
    
    return inference_state

def main():
    # Hyperparameters
    BATCH_SIZE = 1 
    NUM_WORKERS = 4
    MAX_IMAGE_SIZE = 1024
    
    # Countries to evaluate
    # Using a subset for demonstration/speed as per sam_eval.py example, or full list if desired.
    # User requested "iterate across countries", implying all or the standard subset.
    # sam_eval.py uses: ['slovenia', 'france', 'south_africa']
    COUNTRIES = ['slovenia', 'france', 'south_africa'] 
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize CSV with new columns
    with open(CSV_NAME, "w") as f:
        f.write("Country,IOU,Pixel Precision,Pixel Recall,Obj Precision,Obj Recall\n")

    # 1. Build Auto-mask generator
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    sam2_model = build_sam2(MODEL_CFG, CHECKPOINT_PATH, device=device, apply_postprocessing=False)
    mask_generator = SAM2AutomaticMaskGenerator(sam2_model)
    
    # 2. Build Video Predictor
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    predictor = build_sam2_video_predictor(MODEL_CFG, CHECKPOINT_PATH, device=device)

    transform = Sam2EvalTransform(max_image_size=MAX_IMAGE_SIZE)

    for country in COUNTRIES:
        print(f"Evaluating Country: {country}")
        
        dataset = FTW(
            root=DATA_ROOT, 
            countries=[country], 
            split="test", 
            load_boundaries="instance", 
            temporal_options="sam2",
            transforms=transform
        )
        
        if len(dataset) == 0:
            print(f"No samples found for {country}, skipping.")
            continue
        
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=False, 
            num_workers=NUM_WORKERS,
            collate_fn=collate_fn
        )
        
        # Metric Accumulators
        b_iou = []
        b_pxl_prec = []
        b_pxl_recall = []
        b_obj_prec = []
        b_obj_recall = []

        print(f"Starting evaluation on {len(dataset)} samples for {country}...")

        # Tqdm over dataloader
        pbar = tqdm(dataloader, desc=f"Country {country}")
        
        for batch in pbar:
            if batch is None:
                continue
                
            b_images_t = batch["images_t"].to(device) # (B, 2, C, H, W)
            b_img_a_uint8 = batch["img_a_uint8"].numpy() # (B, H, W, 3)
            b_mask_gt = batch["mask_gt"].numpy() # (B, H, W)
            b_orig_size = batch["orig_size"].numpy() # (B, 2)
            
            curr_batch_size = b_images_t.shape[0]
            
            for i in range(curr_batch_size):
                images_t = b_images_t[i] # (2, C, H, W)
                img_a_uint8 = b_img_a_uint8[i] # (H, W, 3)
                mask_gt = b_mask_gt[i] # (H, W)
                orig_h_raw, orig_w_raw = b_orig_size[i]
                
                h, w = img_a_uint8.shape[:2]
                
                # A. Auto-mask Window A
                auto_masks = mask_generator.generate(img_a_uint8)
                
                if len(auto_masks) == 0:
                    # No predictions -> 0 IOU? Or ignore? 
                    # Usually means 0 IOU if GT has objects.
                    # FTW GT always has objects?
                    # Let's assume 0 metrics if failed to predict.
                    b_iou.append(0.0)
                    b_pxl_prec.append(0.0)
                    b_pxl_recall.append(0.0)
                    b_obj_prec.append(0.0)
                    b_obj_recall.append(0.0)
                    continue
                    
                # B. Setup Video Tracking
                inference_state = init_custom_state(
                    predictor, 
                    images_t, 
                    height=h, 
                    width=w, 
                    device=device
                )
                
                # Add each auto-mask as an object
                for mask_idx, mask_result in enumerate(auto_masks):
                    seg = mask_result["segmentation"] # (h, w) bool
                    mask_tensor = torch.tensor(seg, dtype=torch.float32, device=device)
                    
                    predictor.add_new_mask(
                         inference_state=inference_state,
                         frame_idx=0,
                         obj_id=mask_idx,
                         mask=mask_tensor
                    )
                    
                # C. Propagate to Frame 1
                video_segments = {} 
                for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                    if out_frame_idx == 1:
                        for k, out_obj_id in enumerate(out_obj_ids):
                            pred_mask = (out_mask_logits[k] > 0.0).cpu().numpy().squeeze()
                            video_segments[out_obj_id] = pred_mask

                # D. Combine predictions
                if len(video_segments) > 0:
                    combined_pred = np.zeros_like(list(video_segments.values())[0], dtype=np.uint8)
                    for m in video_segments.values():
                        combined_pred = np.logical_or(combined_pred, m)
                    combined_pred = combined_pred.astype(np.uint8)
                else:
                    combined_pred = np.zeros((h, w), dtype=np.uint8)
                    
                # GT Binary
                gt_binary = (mask_gt > 0).astype(np.uint8)
                
                # Metrics
                
                # Pixel IOU
                inter = np.logical_and(gt_binary, combined_pred).sum()
                union = np.sum(gt_binary) + np.sum(combined_pred) - inter
                iou = inter / (union + 1e-6)
                
                # Pixel Precision / Recall
                # TP: Pred=1, GT=1
                # FP: Pred=1, GT=0
                # FN: Pred=0, GT=1
                tps = np.logical_and(combined_pred, gt_binary).sum()
                fps = np.logical_and(combined_pred, np.logical_not(gt_binary)).sum()
                fns = np.logical_and(np.logical_not(combined_pred), gt_binary).sum()
                
                pxl_prec = tps / (tps + fps + 1e-6)
                pxl_recall = tps / (tps + fns + 1e-6)
                
                # Object metrics
                # get_object_level_metrics returns (TP, FP, FN) for objects
                obj_tps_val, obj_fps_val, obj_fns_val = get_object_level_metrics(gt_binary, combined_pred)
                obj_prec = obj_tps_val / (obj_tps_val + obj_fps_val + 1e-6)
                obj_recall = obj_tps_val / (obj_tps_val + obj_fns_val + 1e-6)
                
                # Append
                b_iou.append(iou)
                b_pxl_prec.append(pxl_prec)
                b_pxl_recall.append(pxl_recall)
                b_obj_prec.append(obj_prec)
                b_obj_recall.append(obj_recall)
        
        # Aggregate per country
        if len(b_iou) > 0:
            mean_iou = np.mean(b_iou)
            mean_pxl_prec = np.mean(b_pxl_prec)
            mean_pxl_recall = np.mean(b_pxl_recall)
            mean_obj_prec = np.mean(b_obj_prec)
            mean_obj_recall = np.mean(b_obj_recall)
            
            print(f"Country {country} Results: IOU={mean_iou:.4f}, PxlPrec={mean_pxl_prec:.4f}, PxlRec={mean_pxl_recall:.4f}, ObjPrec={mean_obj_prec:.4f}, ObjRec={mean_obj_recall:.4f}")
            
            with open(CSV_NAME, "a") as f:
                f.write(f"{country},{mean_iou:.4f},{mean_pxl_prec:.4f},{mean_pxl_recall:.4f},{mean_obj_prec:.4f},{mean_obj_recall:.4f}\n")
        else:
            print(f"No valid results for {country}")

    print(f"Evaluation complete. Results saved to {CSV_NAME}")

if __name__ == "__main__":
    main()
