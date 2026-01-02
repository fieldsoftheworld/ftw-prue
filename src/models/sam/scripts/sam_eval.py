## based on https://github.com/sagieppel/fine-tune-train_segment_anything_2_in_60_lines_of_code/blob/main/TRAIN_multi_image_batch.py
import sys
from pathlib import Path

# Add parent directory to path to import sam modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import FTW
from datamodules import preprocess
from image_mlp import PixelMLP
from metrics import get_object_level_metrics, get_object_level_metrics_mpred
from torch.utils.data import DataLoader
from segment_anything import SamAutomaticMaskGenerator, SamPredictor #, sam_model_registry
from build_sam import sam_model_registry
from segment_anything.modeling.mask_decoder import MaskDecoder
from sam_mask_decoder_change import new_predict_masks
from sam_predictor_set_image_change import new_set_image
from tqdm import tqdm
from torch import nn

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
def read_batch(batch,nsel=1,use_rgb=True,device='cpu'):
    if use_rgb:
        bimage = batch['image'][:,:3].to(device) # sel RGB channels
    else:
        bimage = batch['image'].to(device)
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
    "austria","belgium","brazil","cambodia","corsica","croatia","denmark","estonia","finland","france","germany","india","kenya","latvia","lithuania","luxembourg","netherlands","portugal","rwanda","slovakia","slovenia","south_africa","spain","sweden","vietnam"
]

existing_countries = [
    "austria", "corsica", "finland", "kenya", "portugal", "south_africa", "belgium", "croatia", "france", "lithuania", "rwanda", "spain", "brazil", "denmark", "germany", "luxembourg", "slovakia", "sweden", "cambodia", "estonia", "india", "netherlands", "slovenia", "vietnam"
]

## main
if __name__ == "__main__":
    # device
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    # elif torch.mps.is_available():
    #     device = 'mps'

    ## params
    exp_name = "contrast_imgp"#"allinp_re_ed_e2e_30pv2_7"#"embcond_only" #"allinp_re_ed_e2e_22"
    csv_name = "results/" + exp_name + "_results.csv"

    ## create csv
    with open(csv_name, "w") as f:
        f.write("Country,IOU,Pixel Precision,Pixel Recall,Obj Precision,Obj Recall\n")

    ## setup false color image model
    use_img_prep = True

    if use_img_prep:
        img_ckpt = "/projects/benq/atwollam/FTW-Bakeoff/specialized_field_models/sam/checkpoints/img_model_contr_pxl_all_12" #None
        img_model = PixelMLP(8, 3, 64, 4) # in_chn, out_chn, hidden_dim, num_layers
        if img_ckpt is not None:
            ckpt = torch.load(img_ckpt)
            img_model.load_state_dict(ckpt['state_dict'])
            del ckpt
        img_model.to(device)

    ## setup SAM model
    ckpt_path = "/projects/benq/atwollam/FTW-Bakeoff/specialized_field_models/sam/checkpoints/sam_vit_h_4b8939.pth"#sam_final_e2e_adpt_embcond_20"#sam_vit_h_4b8939.pth"#sam_final_allinp_re_ed_e2e_30"#sam_vit_h_4b8939.pth"#sam_final_winA_decoder_only_3"
    iou_thresh = 0.88 # default: 0.88
    stability_thresh = 0.95 # default: 0.95

    in_chans = 3 #8
    sam = sam_model_registry["vit_h"](in_chans=in_chans, checkpoint=ckpt_path)

    if in_chans==8:
        pixel_mean = 2*[123.675, 116.28, 103.53, 123.675]
        pixel_std = 2*[58.395, 57.12, 57.375, 58.395]
        sam.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        sam.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

    sam = sam.to(device)
    #predictor = SamPredictor(sam)
    #predictor.model.mask_decoder.predict_masks = new_predict_masks.__get__(predictor.model.mask_decoder, MaskDecoder)
    mask_generator = SamAutomaticMaskGenerator(sam,pred_iou_thresh=iou_thresh,stability_score_thresh=stability_thresh)
    mask_generator.predictor.model.mask_decoder.predict_masks = new_predict_masks.__get__(mask_generator.predictor.model.mask_decoder, MaskDecoder)
    mask_generator.predictor.set_image = new_set_image.__get__(mask_generator.predictor, SamPredictor)

    # embed cond
    use_emb_cond = False
    if use_emb_cond:
        emb_ckpt = '/projects/benq/atwollam/FTW-Bakeoff/specialized_field_models/sam/checkpoints/emb_cond_e2e_adpt_embcond_20' #emb_cond_embcond_only_10'
        emb_tokens = 1
        embed_dim = sam.prompt_encoder.embed_dim
        #print(embed_dim) # 256
        emb_cond = nn.Embedding(emb_tokens, embed_dim)
        if emb_ckpt is not None:
            ckpt = torch.load(emb_ckpt)
            emb_cond.load_state_dict(ckpt)
            del ckpt
        emb_cond.to(device)
        mask_generator.predictor.emb_cond = emb_cond

    ## setup dataloader
    data_dir = '/projects/benq/ftw-data/data/ftw' #'Directory of dataset'
    countries = ['india', 'kenya', 'portugal', 'rwanda']# ['slovenia', 'france', 'south_africa'] #existing_countries #['belgium'] #'Countries to evaluate on' # = valid_countries
    split = 'test' # (e.g. "train", "val", "test")
    load_boundaries='instance' #'mask with 3-class, 2-class, instance'
    temporal_options = "stacked" #"stacked" #"windowA" #'Temporal option (stacked, windowA, windowB, etc.)'
    batch_size = 4 #64
    nsel = 1 #needs to be 1, since is number of points PER object mask

    for country in countries:
        ds = FTW(
            root=data_dir,
            countries=[country],
            split=split,
            transforms=preprocess,
            load_boundaries=load_boundaries,
            temporal_options=temporal_options
        )
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

        b_iou, b_pxl_prec, b_pxl_recall, b_obj_prec, b_obj_recall = [], [], [], [], []
        with torch.no_grad():
            with tqdm(dl, unit='batch') as tdl:
                tdl.set_description(f"Country {country}")
                for itr, batch in enumerate(tdl):
                    for ind in range(batch['image'].shape[0]):
                        #image = batch['image'][ind,:in_chans].moveaxis(0,-1) # sel RGB channels
                        #image = (image*256).to(torch.uint8).numpy()
                        if use_img_prep:
                            image = batch['image'][ind,:].to(device).unsqueeze(0)
                            image = img_model(image)
                        else:
                            image = batch['image'][ind,:in_chans].to(device).unsqueeze(0)
                        gt_mask = batch['mask'][ind].to(device) > 0

                        outputs = mask_generator.generate(image)
                        #print(outputs)

                        masks = [torch.tensor(out['segmentation'],device=device) for out in outputs]
                        #print(len(masks))
                        if len(masks)==0: masks = [torch.zeros((gt_mask.shape[-2],gt_mask.shape[-1]),device=device),]

                        pred_mask = torch.stack(masks).sum(0) > 0 # 2-class

                        # pixel metrics
                        inter = torch.logical_and(gt_mask, pred_mask).sum()
                        union = gt_mask.sum() + pred_mask.sum() - inter
                        iou = (inter / union).cpu().item() if union>0 else 1.
                        #print(inter, union, iou)

                        pxl_tps = inter.cpu().item()
                        pxl_fps = torch.logical_and(gt_mask==False, pred_mask).sum().cpu().item()
                        pxl_fns = torch.logical_and(gt_mask, pred_mask==False).sum().cpu().item()

                        pxl_prec = pxl_tps / (pxl_tps + pxl_fps) if pxl_tps>0 or pxl_fps>0 else 0
                        pxl_recall = pxl_tps / (pxl_tps + pxl_fns) if pxl_tps>0 or pxl_fns>0 else 0

                        # object metrics
                        obj_tps, obj_fps, obj_fns = get_object_level_metrics(gt_mask.cpu().to(torch.float32).numpy(), pred_mask.cpu().to(torch.float32).numpy(), iou_threshold=0.5)
                        #o_masks = [mask.cpu().to(torch.float32).numpy() for mask in masks]
                        #obj_tps, obj_fps, obj_fns = get_object_level_metrics_mpred(gt_mask.cpu().to(torch.float32).numpy(), o_masks, iou_threshold=0.5)

                        obj_prec = obj_tps / (obj_tps + obj_fps) if obj_tps>0 or obj_fps>0 else 0
                        obj_recall = obj_tps / (obj_tps + obj_fns) if obj_tps>0 or obj_fns>0 else 0

                        # append metrics
                        b_iou.append(iou)
                        b_pxl_prec.append(pxl_prec); b_pxl_recall.append(pxl_recall)
                        b_obj_prec.append(obj_prec); b_obj_recall.append(obj_recall)

                        # print(
                        #     "IOU {}, pxl precision {}, pxl recall {}, pxl f1 {}, obj precision {}, obj recall {}, obj f1 {}".format(
                        #         iou, pxl_prec, pxl_recall, 2*pxl_prec*pxl_recall/(pxl_prec+pxl_recall),
                        #         obj_prec, obj_recall, 2*obj_prec*obj_recall/(obj_prec+obj_recall)
                        #     )
                        # )

                        # #show_masks_on_image(image, masks)
                        # show_masks_on_canvas(image, masks, random_color=False)
                        # plt.savefig('images/sample_pred.png')

                        # fig, axs = plt.subplots(1, 2, figsize=(2 * 5, 8))
                        # axs[0].imshow(image); axs[0].axis('off')
                        # axs[1].imshow(gt_mask); axs[1].axis('off')
                        # plt.savefig('images/sample_data.png')

                        # fig, axs = plt.subplots(1, 2, figsize=(2 * 5, 8))
                        # axs[0].imshow(image); axs[0].axis('off')
                        # axs[1].imshow(pred_mask); axs[1].axis('off')
                        # plt.savefig('images/sample_mask.png')
                        #break
                    #break

        ## aggregate metrics
        iou = np.array(b_iou).mean()
        pxl_prec = np.array(b_pxl_prec).mean(); pxl_recall = np.array(b_pxl_recall).mean()
        obj_prec = np.array(b_obj_prec).mean(); obj_recall = np.array(b_obj_recall).mean()

        print(
            "Country {}: IOU {}, pxl precision {}, pxl recall {}, pxl f1 {}, obj precision {}, obj recall {}, obj f1 {}".format(
                country,
                iou, pxl_prec, pxl_recall, 2*pxl_prec*pxl_recall/(pxl_prec+pxl_recall),
                obj_prec, obj_recall, 2*obj_prec*obj_recall/(obj_prec+obj_recall)
            )
        )

        ## save metrics
        with open(csv_name, "a") as f:
            f.write("{},{},{},{},{},{}\n".format(country,iou,pxl_prec,pxl_recall,obj_prec,obj_recall))

