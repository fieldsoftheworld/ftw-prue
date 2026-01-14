# Mask2Former for PRUE
by: Zhanpei Fang
last updated: 13 Jan 2026

## Installation
All setup can be completed through running the `create_env.sh` script. It works for my system but I haven't tested it for other systems so far. This installs detectron2 and compiles the CUDA kernel for MSDeformAttn. For further guidance see the [official documentation for installation](https://github.com/facebookresearch/Mask2Former/blob/main/INSTALL.md). 

## Dataset conversion
Run the `tools/ftw_to_coco.py` script to convert the FTW dataset to the COCO instance and panoptic formats required for using with detectron2. 

## Training

Training can be run with a command like the following:
```
python scripts/train_panoptic.py --num-gpus 1 --num-machines 1 --machine-rank 0 --dist-url tcp://127.0.0.1:50152 --config-file configs/local/ftw_panoptic_local.yaml --coco-root /path_to_data_folder/ftw/coco/ --weights path_to_weights_folder/weights/model_final_a407fd.pkl
```

I recommend creating a `local/` folder under your configs that contain hardcoded paths for the following:
```
_BASE_: ../ftw/panoptic-segmentation/swin/maskformer2_swin_small_bs16_50ep.yaml

MODEL:
  # Point to your fine-tuning checkpoint or pre-trained Swin weights
  WEIGHTS: "/path_to_weights_folder/model_final_a407fd.pkl"

OUTPUT_DIR: "your_desired_output_path"
```