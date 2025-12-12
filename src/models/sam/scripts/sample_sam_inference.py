import sys
from pathlib import Path

# Add parent directory to path to import sam modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import FTW
from datamodules import preprocess
from torch.utils.data import DataLoader
from segment_anything import SamAutomaticMaskGenerator
from build_sam import sam_model_registry

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

## main
if __name__ == "__main__":
    # setup dataloader
    data_dir = '../../../data/ftw' #'Directory of dataset'
    countries = ['belgium'] #'Countries to evaluate on'
    split = 'test'
    load_boundaries='instance' #'mask with 3-class, 2-class, instance'
    temporal_options = "windowA" #'Temporal option (stacked, windowA, windowB, etc.)'

    ds = FTW(
        root=data_dir,
        countries=countries,
        split=split,
        transforms=preprocess,
        load_boundaries=load_boundaries,
        temporal_options=temporal_options
    )
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)

    temp = next(iter(dl)) #data[0]

    ind = 0
    image = temp['image'][ind,:3].moveaxis(0,-1)
    gt_mask = temp['mask'][ind]

    # device
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    # elif torch.mps.is_available():
    #     device = 'mps'

    image = (image*256).to(torch.uint8).numpy()
    print(image.shape)#, (image*256).to(torch.uint8))

    fig, axs = plt.subplots(1, 2, figsize=(2 * 5, 8))
    axs[0].imshow(image); axs[0].axis('off')
    axs[1].imshow(gt_mask); axs[1].axis('off')
    plt.savefig('images/sample_data.png')

    # SAM
    sam = sam_model_registry["vit_h"](checkpoint="/Users/alexanderwollam/work/FTW/FTW-Bakeoff/specialized_field_models/sam/checkpoints/sam_vit_h_4b8939.pth")
    sam = sam.to(device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    with torch.no_grad():
        outputs = mask_generator.generate(image)
    #print(outputs)

    #masks = outputs["masks"]
    masks = [out['segmentation'] for out in outputs]
    #show_masks_on_image(image, masks)
    show_masks_on_canvas(image, masks, random_color=False)
    plt.savefig('images/sample_pred.png')


