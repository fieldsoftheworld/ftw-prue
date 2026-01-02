import sys
from pathlib import Path

# Add parent directory to path to import sam modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import FTW
from datamodules import preprocess
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

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
    fig, axs = plt.subplots(1, 2, figsize=(2 * 5, 8))
    axs[0].imshow(temp['image'][ind,:3].moveaxis(0,-1)); axs[0].axis('off')
    axs[1].imshow(temp['mask'][ind]); axs[1].axis('off')
    plt.savefig('images/sample_data.png')
