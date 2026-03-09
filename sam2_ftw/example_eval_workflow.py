## Example video workflow
import os
from PIL import Image
import numpy as np
import torch
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# Setup
video_path = "notebooks/videos/bedroom"
sam2_checkpoint = "checkpoints/sam2_hiera_tiny.pt"
model_cfg = "sam2_hiera_t.yaml"
device = "cuda"

# Auto-masking of first frame (from automatic mask generation notebook)
sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
first_frame_path = os.path.join(video_path, os.listdir(video_path)[0])
first_frame = Image.open(first_frame_path)
first_frame = np.array(first_frame.convert("RGB"))
mask_generator = SAM2AutomaticMaskGenerator(sam2)
auto_masks = mask_generator.generate(first_frame)
print("Number of auto-masks:", len(auto_masks))

# Add every 'auto-mask' as it's own prompt for video tracking
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
inference_state = predictor.init_state(video_path=video_path)
dtype = next(predictor.parameters()).dtype
lowres_side_length = predictor.image_size // 4
for mask_idx, mask_result in enumerate(auto_masks):
    # Get mask into form expected by the model
    mask_tensor = torch.tensor(mask_result["segmentation"], dtype=dtype, device=device)
    lowres_mask = torch.nn.functional.interpolate(
        mask_tensor.unsqueeze(0).unsqueeze(0),
        size=(lowres_side_length, lowres_side_length),
        mode="bilinear",
        align_corners=False,
    ).squeeze()

    # Add each mask as it's own 'object' to segment
    _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=mask_idx,
        mask=lowres_mask,
    )

# Do video segmentation (same as video segmentation notebook)
video_segments = {}
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy() for i, out_obj_id in enumerate(out_obj_ids)
    }

# ftw has 2 windows (frames) --> need to merge mask predictions w/ non maximal suppression like sam/sam2 automatic_mask_generator.py does

# perform ftw-prue-ref -like eval (or set generate saved code for eval to be performed by a script like ftw-prue-ref/run_full_eval.sh)
# ...
