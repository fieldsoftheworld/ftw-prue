#!/usr/bin/env python3
"""
SAM-2 Temporal Training on FTW (v2)
- Uses window_a -> window_b as temporal sequence
- Uses properly loaded instance masks
- Supports nsel (multiple prompts per image)
- DataLoader + Worker support
- Mixed Prompting (Points/Masks/Both)
- Focal + Dice + IoU Score Loss
"""

import os
import sys
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import rasterio

# Add project root to path if needed, but we are running from sam2_ftw usually
# We need to add sam2 repo to path
sam2_repo_path = Path("/projects/benq/atwollam/FTW-Bakeoff/specialized_field_models/sam2/sam2")
sys.path.insert(0, str(sam2_repo_path))

from build_sam_v2 import build_sam2_video_predictor
from sam2.modeling.backbones.utils import PatchEmbed

# Local imports
from dataset_v2 import FTW

DATA_ROOT = "/projects/benq/ftw-data/data/ftw"
CHECKPOINT_PATH = "/projects/benq/atwollam/FTW-Bakeoff/specialized_field_models/sam2/sam2/checkpoints/sam2.1_hiera_small.pt"
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_s.yaml"

OUTPUT_DIR = "sam2_ftw_v2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class Sam2Transform:
    def __init__(self, nsel=1, max_image_size=1024):
        self.nsel = nsel
        self.max_image_size = max_image_size
        self.img_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.img_std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    def get_masks(self, fullmask):
        """
        Sample nsel instance masks from the full mask.
        fullmask: (H, W) int or float instance mask
        """
        mids = np.unique(fullmask)
        mids = mids[mids > 0] # Exclude background
        
        if len(mids) == 0:
            return None, None, None

        # Sample with replacement
        mask_inds = np.random.choice(mids, self.nsel, replace=True)
        
        masks, points, labels = [], [], []
        for i in range(self.nsel):
            mid = mask_inds[i]
            mask = (fullmask == mid).astype(np.float32)
            
            ys, xs = np.where(mask > 0)
            if len(xs) == 0:
                pt = np.array([0, 0], dtype=np.float32)
                lbl = np.array([0], dtype=np.int32)
            else:
                idx = np.random.randint(len(xs))
                pt = np.array([xs[idx], ys[idx]], dtype=np.float32)
                lbl = np.array([1], dtype=np.int32)
                
            masks.append(mask)
            points.append(pt)
            labels.append(lbl)
            
        return np.stack(masks), np.stack(points), np.stack(labels)

    def __call__(self, sample):
        img_a = sample["window_a"].numpy().transpose(1, 2, 0) # C,H,W -> H,W,C
        img_b = sample["window_b"].numpy().transpose(1, 2, 0)
        mask = sample["mask"].numpy() # H,W int

        # Resize
        r = min(self.max_image_size / img_a.shape[0], self.max_image_size / img_a.shape[1])
        h, w = int(img_a.shape[0] * r), int(img_a.shape[1] * r)

        img_a = cv2.resize(img_a, (w, h))
        img_b = cv2.resize(img_b, (w, h))
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # Normalize and Convert to Torch
        img_a = img_a / 255.0
        img_b = img_b / 255.0
        
        img_a = (img_a - self.img_mean) / self.img_std
        img_b = (img_b - self.img_mean) / self.img_std

        img_a_t = torch.from_numpy(img_a).float().permute(2, 0, 1) # C,H,W
        img_b_t = torch.from_numpy(img_b).float().permute(2, 0, 1)

        # Sample nsel masks
        masks_np, points_np, labels_np = self.get_masks(mask)
        
        if masks_np is None:
            return None
            
        gt_masks = torch.from_numpy(masks_np) # nsel, H, W
        points = torch.from_numpy(points_np).unsqueeze(1) # nsel, 1, 2
        labels = torch.from_numpy(labels_np) # nsel, 1

        return {
            "img_a": img_a_t,
            "img_b": img_b_t,
            "gt_masks": gt_masks,
            "points": points,
            "labels": labels,
            "orig_size": torch.tensor([h, w])
        }

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

def main():
    # Hyperparameters
    BATCH_SIZE = 4
    NUM_WORKERS = 4
    NSEL = 3
    EPOCHS = 10
    LR = 1e-4
    WEIGHT_DECAY = 1e-4
    ACCUMULATION_STEPS = 4
    MAX_IMAGE_SIZE = 1024
    CHANNELS = 3
    COUNTRIES = ["france"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = Sam2Transform(nsel=NSEL, max_image_size=MAX_IMAGE_SIZE)
    dataset = FTW(
        root=DATA_ROOT, 
        countries=COUNTRIES, 
        split="train", 
        load_boundaries="instance", 
        temporal_options="sam2",
        transforms=transform
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        drop_last=True
    )

    print(f"Dataset size: {len(dataset)}")

    predictor = build_sam2_video_predictor(
        MODEL_CFG, checkpoint=None, device=device
    )

    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
    state = ckpt["model"] if "model" in ckpt else ckpt
    predictor.load_state_dict(state, strict=False)

    model = predictor

    for p in model.image_encoder.parameters():
        p.requires_grad = False
    model.image_encoder.eval()

    for p in model.sam_prompt_encoder.parameters():
        p.requires_grad = False
    model.sam_prompt_encoder.eval()

    for p in model.sam_mask_decoder.parameters():
        p.requires_grad = True
    model.sam_mask_decoder.train()

    optimizer = torch.optim.AdamW(
        model.sam_mask_decoder.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    mean_iou = 0.0
    global_step = 0

    model.train() 
    
    for epoch in range(EPOCHS):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch in pbar:
            if batch is None:
                continue

            img_a = batch["img_a"].to(device)
            img_b = batch["img_b"].to(device)
            gt_masks = batch["gt_masks"].to(device)
            pts = batch["points"].to(device)
            lbls = batch["labels"].to(device)

            B, _, H, W = img_b.shape
            
            with torch.no_grad():
                 _ = model.forward_image(img_a) 

            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                feats = model.forward_image(img_b)
                
                vis_feats = feats["vision_features"]
                vis_feats = vis_feats.unsqueeze(1).expand(-1, NSEL, -1, -1, -1).flatten(0, 1)
                
                high_res_features = None
                if model.use_high_res_features_in_sam:
                    high_res_features = []
                    for feat in feats["backbone_fpn"][:2]:
                         high_res_features.append(feat.unsqueeze(1).expand(-1, NSEL, -1, -1, -1).flatten(0, 1))

                pts_flat = pts.flatten(0, 1) 
                lbls_flat = lbls.flatten(0, 1) 
                gt_masks_flat = gt_masks.flatten(0, 1) 

                pts_norm = pts_flat.clone()
                pts_norm[:, :, 0] /= W
                pts_norm[:, :, 1] /= H
                pts_norm *= model.image_size
                
                mask_prompt = F.interpolate(
                    gt_masks_flat.unsqueeze(1).float(),
                    size=(256, 256),
                    mode="nearest"
                )

                rand_val = np.random.rand()
                if rand_val < 0.33:
                    sparse, dense = model.sam_prompt_encoder(
                        points=(pts_norm, lbls_flat),
                        boxes=None,
                        masks=None,
                    )
                elif rand_val < 0.66:
                    sparse, dense = model.sam_prompt_encoder(
                        points=None,
                        boxes=None,
                        masks=mask_prompt,
                    )
                else:
                    sparse, dense = model.sam_prompt_encoder(
                        points=(pts_norm, lbls_flat),
                        boxes=None,
                        masks=mask_prompt,
                    )
                
                low_res_masks, prd_scores, _, _ = model.sam_mask_decoder(
                    image_embeddings=vis_feats,
                    image_pe=model.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse,
                    dense_prompt_embeddings=dense,
                    multimask_output=True, 
                    repeat_image=False,
                    high_res_features=high_res_features,
                )
                
                pred = F.interpolate(low_res_masks, size=(H, W), mode="bilinear", align_corners=False)
                
                gt_masks_exp = gt_masks_flat.unsqueeze(1).expand(-1, 3, -1, -1)
                pred_prob = torch.sigmoid(pred)
                
                gamma = 2
                alpha = 0.5
                focal_loss = (-gt_masks_exp * alpha * (1 - pred_prob)**gamma * torch.log(pred_prob + 1e-5) - 
                             (1 - gt_masks_exp) * (1 - alpha) * pred_prob**gamma * torch.log((1 - pred_prob) + 1e-5))
                focal_loss = focal_loss.mean(dim=(-2, -1))
                
                inter = (gt_masks_exp * pred_prob).sum(dim=(-2, -1))
                union = gt_masks_exp.sum(dim=(-2, -1)) + pred_prob.sum(dim=(-2, -1))
                dice_loss = 1 - (2 * inter / (union + 1e-5))
                
                seg_loss_all = focal_loss + dice_loss
                
                seg_loss_min, min_idx = seg_loss_all.min(dim=1)
                seg_loss = seg_loss_min.mean()
                
                inter_all = (gt_masks_exp * (pred_prob > 0.5).float()).sum(dim=(-2, -1))
                union_all = gt_masks_exp.sum(dim=(-2, -1)) + (pred_prob > 0.5).float().sum(dim=(-2, -1)) - inter_all
                iou_all = inter_all / (union_all + 1e-6)
                
                best_iou = iou_all.gather(1, min_idx.unsqueeze(1)).squeeze(1)
                score_loss = torch.abs(prd_scores - iou_all.detach()).mean()
                
                loss = seg_loss + score_loss * 0.05
                
            scaler.scale(loss / ACCUMULATION_STEPS).backward()
            
            global_step += 1
            if global_step % ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            with torch.no_grad():
                avg_iou = best_iou.mean().item()
                mean_iou = 0.99 * mean_iou + 0.01 * avg_iou
            
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "mIoU": f"{mean_iou:.4f}"})

    torch.save(model.sam_mask_decoder.state_dict(),
               os.path.join(OUTPUT_DIR, "mask_decoder_final.pt"))
    print("Training complete")

if __name__ == "__main__":
    main()
