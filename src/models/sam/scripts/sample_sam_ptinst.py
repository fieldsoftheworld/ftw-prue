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

## main
if __name__ == "__main__":
    ## setup dataloader
    data_dir = '../../../data/ftw' #'Directory of dataset'
    countries = ['belgium'] #'Countries to evaluate on'
    split = 'test'
    load_boundaries='instance' #'mask with 3-class, 2-class, instance'
    temporal_options = "windowA" #'Temporal option (stacked, windowA, windowB, etc.)'
    batch_size = 2 #64
    nsel = 1

    ds = FTW(
        root=data_dir,
        countries=countries,
        split=split,
        transforms=preprocess,
        load_boundaries=load_boundaries,
        temporal_options=temporal_options
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # test sample
    batch = next(iter(dl)) #data[0]

    # device
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    # elif torch.mps.is_available():
    #     device = 'mps'

    images, masks, points, labels = read_batch(batch,nsel=nsel,device=device) #images (b,c,h,w), masks (b,nsel,h,w), points (b,n,2), labels, (b,1)

    print(images.shape)#, (image*256).to(torch.uint8))

    ## SAM
    sam = sam_model_registry["vit_h"](checkpoint="/Users/alexanderwollam/work/FTW/FTW-Bakeoff/specialized_field_models/sam/checkpoints/sam_vit_h_4b8939.pth")
    sam = sam.to(device)
    #sam.image_encoder.img_size = 256
    #mask_generator = SamAutomaticMaskGenerator(sam)
    #with torch.no_grad():
    #    outputs = mask_generator.generate(image)
    #print(outputs)
    predictor = SamPredictor(sam)
    predictor.model.mask_decoder.predict_masks = new_predict_masks.__get__(predictor.model.mask_decoder, MaskDecoder)

    original_size = tuple(images.shape[-2:])
    transformed_images = predictor.transform.apply_image_torch(images)
    transformed_images = (transformed_images*256).to(torch.uint8)
    print(images.shape, transformed_images.shape)

    points = predictor.transform.apply_coords_torch(points, original_size)

    ## predict
    #predictor.set_torch_image(images, images.shape[-2:])
    input_size = tuple(transformed_images.shape[-2:])
    input_images = predictor.model.preprocess(transformed_images)
    features = predictor.model.image_encoder(input_images)
    # masks: (b,nsel,h,w)
    ### need to flatten batch/nsel together, since ind=1 is per-instance pts (pts: (b*nsel,1,2))
    #out_masks, iou_predictions, low_res_masks = predictor.predict_torch(point_coords=points, point_labels=labels) #return_logits=True
    points_inp = (points, labels)
    # Embed prompts ## sparse_embeddings: (b,1+nsel(bad/num_pts_per_single_obj_seg),256(d))
    sparse_embeddings, dense_embeddings = predictor.model.prompt_encoder(points=points_inp,boxes=None,masks=None)
    # Predict masks
    print(features.shape, sparse_embeddings.shape, dense_embeddings.shape)
    low_res_masks, iou_predictions = predictor.model.mask_decoder(
        image_embeddings=features,
        image_pe=predictor.model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=True,
    )
    # Upscale the masks to the original image resolution
    out_masks = predictor.model.postprocess_masks(low_res_masks, input_size, original_size)
    if True: #if not return_logits:
        out_masks = out_masks > predictor.model.mask_threshold
    # ^-> out_masks, iou_predictions, low_res_masks
    print(iou_predictions)

    ## Visualize

    ind = 0
    #msel = 0

    image = images[ind,:3].moveaxis(0,-1).cpu()
    #gt_mask = masks[ind,msel]

    #image = (image*256).to(torch.uint8).numpy()

    plt.imshow(image); plt.axis('off')
    plt.savefig('images/sample_image.png')
    #fig, axs = plt.subplots(1, 2, figsize=(2 * 5, 8))
    #axs[0].imshow(image); axs[0].axis('off')
    #axs[1].imshow(gt_mask); axs[1].axis('off')
    # fig, axs = plt.subplots(1, nsel, figsize=(nsel * 5, 8))
    # for i in range(nsel):
    show_masks_on_canvas(image, masks[ind], random_color=False)
    plt.savefig('images/sample_mask.png')

    #masks = outputs["masks"]
    #masks = [out['segmentation'] for out in outputs]
    print(out_masks.shape)
    #print(out_masks[ind])
    mask = out_masks[ind]#,msel]
    #show_masks_on_image(image, masks)
    show_masks_on_canvas(image, mask, random_color=False)
    plt.savefig('images/sample_pred.png')


