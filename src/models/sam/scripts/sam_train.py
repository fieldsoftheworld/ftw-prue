## based on https://github.com/sagieppel/fine-tune-train_segment_anything_2_in_60_lines_of_code/blob/main/TRAIN_multi_image_batch.py
import sys
from pathlib import Path

# Add parent directory to path to import sam modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import FTW
from datamodules import preprocess
from torch.utils.data import DataLoader
from segment_anything import SamPredictor
from build_sam import sam_model_registry
from segment_anything.modeling.mask_decoder import MaskDecoder
from sam_mask_decoder_change import new_predict_masks
from tqdm import tqdm

import torch
import numpy as np
import matplotlib.pyplot as plt
import gc

## mask visualization utility fns
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    del mask
    gc.collect()

def show_masks_on_image(raw_image, masks):
    plt.imshow(np.array(raw_image))
    ax = plt.gca()
    ax.set_autoscale_on(False)
    for mask in masks:
        show_mask(mask, ax=ax, random_color=True)
    plt.axis("off")
    #plt.show()
    del mask
    gc.collect()

def show_masks_on_canvas(raw_image, masks, random_color=True):
    plt.imshow(np.zeros_like(raw_image))
    ax = plt.gca()
    ax.set_autoscale_on(False)
    for mask in masks:
        show_mask(mask, ax=ax, random_color=random_color)
    plt.axis("off")
    #plt.show()
    del mask
    gc.collect()

def get_masks(fullmask,nsel=1):
    mids = fullmask.flatten().unique() # first in is 0 (maskless)
    mask_inds = mids[torch.randint(mids.shape[0],(nsel,))] # (nsel,); sample with replacement
    masks, points, labels = [], [], []
    for i in range(nsel):
        mid = mask_inds[i]
        mask = (fullmask==mid)
        pts = torch.argwhere(mask)
        sel_pt = pts[torch.randint(pts.shape[0],())].flip(0) # (2,); [y,x] -> [x,y]
        mask = mask * (mid>0) # set all 0s if background
        masks.append(mask)
        points.append(sel_pt)
        labels.append((mid>0).to(torch.uint8).squeeze(-1))
    return torch.stack(masks), torch.stack(points), torch.stack(labels) # (nsel,256,256), (nsel,2), (nsel,)

# returns: images (b,c,h,w), masks (b,nsel,h,w), points (b,n,2), labels, (b,nsel)
def read_batch(batch,nsel=1,device='cpu'):
    bimage = batch['image'][:,:3].to(device) # sel RGB channels
    bfullmask = batch['mask'].to(device)
    bmasks, bpoints, blabels = [], [], []
    batch_size = bimage.shape[0]
    for i in range(batch_size):
        masks, points, labels = get_masks(bfullmask[i],nsel=nsel)
        bmasks.append(masks)
        bpoints.append(points)
        blabels.append(labels)
    bmasks = torch.stack(bmasks).to(device)
    bpoints = torch.stack(bpoints).to(device)
    blabels = torch.stack(blabels).to(device)
    return bimage, bmasks, bpoints, blabels # labels: 1=foreground, 0=background

valid_countries = [
    "austria",  
    "belgium",  
    "brazil",  
    "cambodia",  
    "corsica",  
    "croatia",  
    "denmark",  
    "estonia",  
    "finland",  
    "france",  
    "germany",  
    "india",  
    "kenya",  
    "latvia",  
    "lithuania",  
    "luxembourg",  
    "netherlands",  
    "portugal",  
    "rwanda",  
    "slovakia",  
    "slovenia",  
    "south_africa",  
    "spain",  
    "sweden",  
    "vietnam"
]

existing_countries = ["austria", "corsica", "finland", "kenya", "portugal", "south_africa", "belgium", "croatia", "france", "lithuania", "rwanda", "spain", "brazil", "denmark", "germany", "luxembourg", "slovakia", "sweden", "cambodia", "estonia", "india", "netherlands", "slovenia", "vietnam"]

## main
if __name__ == "__main__":
    # device
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    # elif torch.mps.is_available():
    #     device = 'mps'

    ## params
    exp_name = "winA_decoder_2full"
    epochs = 4

    ## setup dataloader
    data_dir = '/projects/benq/ftw-data/data/ftw' #'Directory of dataset'
    countries = existing_countries #['belgium'] #'Countries to evaluate on' # = valid_countries
    split = 'train' # (e.g. "train", "val", "test")
    load_boundaries='instance' #'mask with 3-class, 2-class, instance'
    temporal_options = "windowA" #'Temporal option (stacked, windowA, windowB, etc.)'
    batch_size = 8 #64
    accumulation_steps = 4
    nsel = 1 #needs to be 1, since is number of points PER object mask

    ds = FTW(
        root=data_dir,
        countries=countries,
        split=split,
        transforms=preprocess,
        load_boundaries=load_boundaries,
        temporal_options=temporal_options
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=8)

    ## setup SAM model
    ckpt_path = "/projects/benq/atwollam/FTW-Bakeoff/specialized_field_models/sam/checkpoints/sam_final_winA_decoder_only_3" #sam_vit_h_4b8939.pth"

    sam = sam_model_registry["vit_h"](checkpoint=ckpt_path)
    sam = sam.to(device)
    predictor = SamPredictor(sam)
    predictor.model.mask_decoder.predict_masks = new_predict_masks.__get__(predictor.model.mask_decoder, MaskDecoder)

    # Set training parameters
    predictor.model.mask_decoder.train(True) # enable training of mask decoder
    predictor.model.prompt_encoder.train(True) # enable training of prompt encoder
    predictor.model.image_encoder.train(True) # enable training of image encoder: For this to work you need to scan the code for "no_grad" and remove them all

    optimizer = torch.optim.AdamW(params=predictor.model.parameters(),lr=1e-5,weight_decay=4e-5)
    scaler = torch.cuda.amp.GradScaler() # mixed precision

    dl_len = len(dl)
    for epoch in range(epochs):
        with tqdm(dl, unit='batch') as tdl:
            for itr, batch in enumerate(tdl):
                with torch.cuda.amp.autocast():
                    tdl.set_description(f"Epoch {epoch}")

                    images, masks, points, labels = read_batch(batch,nsel=nsel,device=device) #images (b,c,h,w), masks (b,nsel,h,w), points (b,n,2), labels, (b,1)

                    ## format inputs
                    original_size = tuple(images.shape[-2:])
                    transformed_images = predictor.transform.apply_image_torch(images)
                    transformed_images = (transformed_images*256).to(torch.uint8)
                    input_size = tuple(transformed_images.shape[-2:])

                    points = predictor.transform.apply_coords_torch(points, original_size)

                    ## predict
                    input_images = predictor.model.preprocess(transformed_images)
                    features = predictor.model.image_encoder(input_images)
                    points_inp = (points, labels)
                    # Embed prompts ## sparse_embeddings: (b,1+nsel(bad/num_pts_per_single_obj_seg),256(d))
                    sparse_embeddings, dense_embeddings = predictor.model.prompt_encoder(points=points_inp,boxes=None,masks=None)
                    # Predict masks
                    low_res_masks, prd_scores = predictor.model.mask_decoder(
                        image_embeddings=features,
                        image_pe=predictor.model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=True,
                    )
                    # Upscale/downscale the masks to the original image resolution
                    prd_masks = predictor.model.postprocess_masks(low_res_masks, input_size, original_size)
                    # if True: #if not return_logits:
                    #     out_masks = out_masks > predictor.model.mask_threshold
                    # ^-> out_masks, iou_predictions, low_res_masks
                    #print(prd_scores) #iou_predictions

                    ## loss calculation
                    gt_mask = masks.expand(-1,3,-1,-1).to(torch.float32) # (b,1,h,w) -> (b,3,h,w); bool -> float
                    prd_mask = torch.sigmoid(prd_masks) # Turn logit map to probability map
                    
                    # Segmentation Loss calculation
                    seg_loss = (-gt_mask * torch.log(prd_mask+1e-5) - (1-gt_mask) * torch.log((1-prd_mask)+1e-5)).mean() # cross entropy loss

                    # Score loss calculation (intersection over union) IOU
                    inter = (gt_mask * (prd_mask>0.5)).sum((2,3))
                    iou = inter / (gt_mask.sum((2,3)) + (prd_mask>0.5).sum((2,3)) - inter)
                    iou[labels.squeeze(-1)==0] = torch.zeros_like(iou[labels.squeeze(-1)==0])
                    iou = torch.where(torch.logical_or(iou.isnan(),iou.isinf()),torch.zeros_like(iou),iou)
                    score_loss = torch.abs(prd_scores - iou.detach()).mean()

                    loss = seg_loss + score_loss*0.05  # mix losses
                    if loss.isnan().any(): print(seg_loss, score_loss, loss)

                    # apply back propogation

                    scaler.scale(loss).backward()  # Backpropogate

                    if (itr+1) % accumulation_steps == 0 or (itr+1) == dl_len:
                        scaler.step(optimizer)
                        scaler.update() # Mix precision
                        predictor.model.zero_grad() # empty gradient

                    if itr%3000==0: torch.save(predictor.model.state_dict(), "checkpoints/sam_{}_{}_{}".format(exp_name,epoch,itr)) # save model

                    # Display results

                    if itr==0: mean_iou=0
                    mean_iou = mean_iou * 0.99 + 0.01 * np.mean(iou.cpu().detach().numpy())
                    #print("step)",itr, "Accuracy(IOU)=",mean_iou)
                    tdl.set_postfix(
                        loss=loss.cpu().item(),
                        iou=mean_iou,
                    )
    torch.save(predictor.model.state_dict(), "checkpoints/sam_final_{}_{}".format(exp_name,epochs)) # save model


