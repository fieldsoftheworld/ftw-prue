"""
Enhanced visualization utilities for Mask2Former predictions on satellite imagery.

This script runs inference on GeoTIFF images and creates visualization overlays.
Does NOT output GeoJSON - use scripts/predict_geotiff_to_geojson.py for that.

Usage:
    python scripts/predict_and_visualize.py \
        --config-file path/to/config.yaml \
        --input path/to/image.tif \
        --output path/to/output/dir \
        --weights path/to/model.pth

To use programmatically:
    from scripts.predict_and_visualize import predict_viz
    from trainer.prediction import SatellitePredictor
    from trainer.pred_visualization import SatelliteVisualizer

    # Setup
    predictor = SatellitePredictor(cfg)
    visualizer = SatelliteVisualizer(metadata)

    # Run inference and visualization
    image = read_geotiff("image.tif")
    predictions = predictor(image)
    vis_path = visualizer.visualize_predictions(
        image, 
        predictions,
        output_dir="output",
        image_id="image_001"
    )
"""

import os, sys
from pathlib import Path
import logging
from typing import Optional, Tuple, Dict, Any
import torch
import cv2

from detectron2.config import CfgNode
from detectron2.data import MetadataCatalog
from detectron2.data.detection_utils import read_geotiff

# Add project root to Python path BEFORE importing custom modules
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from trainer.prediction import SatellitePredictor
from trainer.pred_visualization import SatelliteVisualizer

logger = logging.getLogger(__name__)

def predict_viz(cfg: CfgNode, 
                     image_path: str, 
                     output_dir: str,
                     metadata=None,
                     ground_truth_dir: Optional[str] = None) -> None:
    """
    Run inference and visualization on a single image or directory of images.
    
    Args:
        cfg (CfgNode): Detectron2 config
        image_path (str): Path to image or directory of images
        output_dir (str): Output directory for visualizations
        metadata: Optional metadata override
    """
    # Setup
    predictor = SatellitePredictor(cfg)
    if metadata is None:
        metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
    visualizer = SatelliteVisualizer(metadata)
    
    # Handle directory or single file
    image_paths = []
    if os.path.isdir(image_path): # get npz files as well
        image_paths.extend(Path(image_path).glob("*.tif"))
        image_paths.extend(Path(image_path).glob("*.npz"))
    else:
        image_paths.append(Path(image_path))
    
    # Process images
    # import pdb; pdb.set_trace()
    for img_path in image_paths:
        try:
            # Read image
            image = read_geotiff(str(img_path), format=cfg.INPUT.FORMAT)
            
            # Run inference
            predictions = predictor(image)
            # print("predictions: ", predictions)
            
            # Load ground truth if available
            ground_truth = None
            if ground_truth_dir:
                try:
                    # Find corresponding ground truth panoptic annotation
                    img_id = img_path.stem
                    gt_path = Path(ground_truth_dir) / "panoptic_test" / f"{img_id}.png"
                    json_path = Path(ground_truth_dir) / "annotations" / "panoptic_test.json"
                    
                    if gt_path.exists() and json_path.exists():
                        # Load panoptic segmentation ground truth
                        import json
                        from panopticapi.utils import rgb2id
                        
                        # Read PNG file
                        pan_gt = cv2.imread(str(gt_path))
                        if pan_gt is not None:
                            pan_gt = cv2.cvtColor(pan_gt, cv2.COLOR_BGR2RGB)
                            pan_seg_gt = rgb2id(pan_gt)
                            
                            # Load and find corresponding segments info from JSON
                            with open(json_path) as f:
                                pan_gt_json = json.load(f)
                            
                            # Find segments info for this image
                            # TODO: find segments info
                            segments_info = None
                            for ann in pan_gt_json['annotations']:
                                if ann['image_id'] == img_id or str(ann['image_id']) == img_id:
                                    segments_info = ann['segments_info']
                                    break
                            
                            if segments_info is not None:
                                ground_truth = {
                                    "panoptic_seg": (torch.as_tensor(pan_seg_gt), segments_info)
                                }
                                logger.info(f"Loaded ground truth from {gt_path}")
                            else:
                                logger.warning(f"No segments info found for image {img_id}")
                        else:
                            logger.warning(f"Failed to read panoptic PNG from {gt_path}")
                    else:
                        logger.warning(f"Ground truth files not found for {img_id}")
                except Exception as e:
                    logger.warning(f"Failed to load ground truth: {str(e)}")
            
            # Visualize
            print("Trying to visualize predictions...")
            vis_path = visualizer.visualize_predictions(
                image, 
                predictions,
                ground_truth=ground_truth,
                output_dir=output_dir,
                image_id=img_path.stem
            )
            
            logger.info(f"Processed {img_path.name}")
            logger.info(f"Saved visualization to {vis_path}")
            
        except Exception as e:
            logger.error(f"Error processing {img_path}: {str(e)}")

if __name__ == "__main__":
    import sys

    dir_path = os.path.dirname(os.path.realpath(__file__))
    print("dir_path: ", dir_path)
    sys.path.append(os.path.realpath(f"{dir_path}/.."))

    import argparse
    from detectron2.config import get_cfg
    from detectron2.projects.deeplab import add_deeplab_config
    from detectron2.modeling import build_model, META_ARCH_REGISTRY
    from mask2former.config import add_maskformer2_config
    from mask2former.maskformer_model import MaskFormer
    from detectron2.data.datasets import register_coco_panoptic

    from trainer.metadata import get_metadata
    
    parser = argparse.ArgumentParser(description="Visualize predictions on satellite imagery")
    parser.add_argument("--config-file", required=True, help="Path to config file")
    parser.add_argument("--input", required=True, help="Path to input image or directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--weights", required=True, help="Path to model weights")
    parser.add_argument("--coco-root", help="Path to COCO format dataset root containing panoptic_test/ and annotations/")
    parser.add_argument("--opts", default=[], nargs=argparse.REMAINDER)
    args = parser.parse_args()
               
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Setup config
    cfg = get_cfg()
    cfg.set_new_allowed(True)
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(["MODEL.WEIGHTS", args.weights])
    
    # Register both TRAIN and TEST datasets with metadata
    # The model may internally reference the training dataset metadata
    custom_metadata = get_metadata()
    
    # Register training dataset (model may reference this)
    # Always register it, even if we don't have COCO paths, so metadata is available
    if args.coco_root:
        register_coco_panoptic(
            name=cfg.DATASETS.TRAIN[0],
            metadata=custom_metadata,
            image_root=Path(args.coco_root)/'train/',
            panoptic_root=Path(args.coco_root)/'panoptic_train/',
            panoptic_json=Path(args.coco_root)/'annotations/panoptic_train.json',
            instances_json=Path(args.coco_root)/'annotations/instances_train.json'
        )
    else:
        # Register with empty paths just to set metadata
        register_coco_panoptic(
            name=cfg.DATASETS.TRAIN[0],
            metadata=custom_metadata,
            image_root="",
            panoptic_root="",
            panoptic_json="",
            instances_json=""
        )
    
    # Register test dataset
    if args.coco_root:
        register_coco_panoptic(
            name=cfg.DATASETS.TEST[0],
            metadata=custom_metadata,
            image_root=args.input,
            panoptic_root=Path(args.coco_root)/'panoptic_test/',
            panoptic_json=Path(args.coco_root)/'annotations/panoptic_test.json',
            instances_json=Path(args.coco_root)/'annotations/instances_test.json'
        )
    else:
        # Register test dataset without COCO paths (for standalone inference)
        register_coco_panoptic(
            name=cfg.DATASETS.TEST[0],
            metadata=custom_metadata,
            image_root=args.input,
            panoptic_root="",
            panoptic_json="",
            instances_json=""
        )
    
    cfg.freeze()
    
    # Run visualization
    predict_viz(cfg, args.input, args.output, ground_truth_dir=None)