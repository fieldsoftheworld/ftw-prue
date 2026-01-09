"""
Shared setup for all scripts:
sets up the config, logger, registers datasets
"""

import sys
import os
import logging
from pathlib import Path
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_panoptic
from detectron2.data.datasets.coco import register_coco_instances, load_coco_json
from detectron2.engine import default_setup
import detectron2.utils.comm as comm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _test_npz_file(file_path):
    """
    Test if an NPZ file can be loaded without encoding errors.
    Returns True if the file is valid, False if it has encoding issues.
    """
    import numpy as np
    try:
        # Try to load the file
        data = np.load(file_path)
        # If successful, close it and return True
        data.close()
        return True
    except LookupError as e:
        # Python 3.11 encoding issue (cp437)
        if "encoding" in str(e).lower() or "cp437" in str(e).lower():
            return False
        # Other LookupError - re-raise
        raise
    except Exception:
        # Other errors - assume file is problematic
        return False


def _filter_problematic_npz_files(dataset_dicts, logger=None):
    """
    Filter out dataset dicts that point to NPZ files with encoding issues.
    
    Args:
        dataset_dicts: List of dataset dictionaries from load_coco_panoptic_json
        logger: Optional logger for reporting filtered files
        
    Returns:
        Filtered list of dataset dictionaries
    """
    import os
    filtered_dicts = []
    skipped_count = 0
    
    for dataset_dict in dataset_dicts:
        file_name = dataset_dict.get("file_name", "")
        
        # Only check NPZ files
        if file_name.endswith(".npz"):
            if not _test_npz_file(file_name):
                skipped_count += 1
                if logger:
                    logger.warning(
                        f"Skipping NPZ file with encoding issues: {file_name}. "
                        f"This file is incompatible with Python 3.11."
                    )
                continue
        
        filtered_dicts.append(dataset_dict)
    
    if skipped_count > 0:
        msg = f"Filtered out {skipped_count} NPZ file(s) with encoding issues (Python 3.11 compatibility)."
        if logger:
            logger.warning(msg)
        print(f"Warning: {msg}")
    
    return filtered_dicts


def _get_dataset_paths(dataset_name, coco_root):
    """
    Get standardized paths for different datasets.
    Returns a dictionary with image_root, panoptic_root, panoptic_json, instances_json.
    """
    dataset_type = dataset_name.split('_')[0]
    split = dataset_name.split('_')[-1]
    coco_path = Path(coco_root)
    
    # Define path templates for each dataset type
    # Only FTW dataset is supported in this branch
    dataset_configs = {
        'ftw': {
            'train': {
                'image_root': coco_path / 'train',
                'panoptic_root': coco_path / 'panoptic_train',
                'panoptic_json': coco_path / 'annotations' / 'panoptic_train.json',
                'instances_json': coco_path / 'annotations' / 'instances_train.json'
            },
            'test': {
                'image_root': coco_path / 'test',
                'panoptic_root': coco_path / 'panoptic_test',
                'panoptic_json': coco_path / 'annotations' / 'panoptic_test.json',
                'instances_json': coco_path / 'annotations' / 'instances_test.json'
            }
        }
    }
    
    # Get configuration for this dataset and split
    if dataset_type not in dataset_configs:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    if split not in dataset_configs[dataset_type]:
        raise ValueError(f"Unknown split: {split} for dataset: {dataset_type}")
    
    return dataset_configs[dataset_type][split]


def register_all_datasets(cfg, meta, coco_root, logger):
    """
    Register both training and test datasets with proper metadata using unified path resolution.
    Filters out NPZ files with encoding issues (Python 3.11 compatibility).
    Supports both panoptic segmentation (Mask2Former) and instance segmentation (Mask-RCNN).
    """
    from detectron2.data.datasets.coco_panoptic import load_coco_panoptic_json
    from detectron2.data import DatasetCatalog, MetadataCatalog
    
    # Check if this is an instance segmentation model (Mask-RCNN)
    is_instance_seg = cfg.MODEL.META_ARCHITECTURE == "GeneralizedRCNN"
    
    dataset_names = [cfg.DATASETS.TRAIN[0], cfg.DATASETS.TEST[0]]
    print("dataset names: ", dataset_names)
    print(f"Model architecture: {cfg.MODEL.META_ARCHITECTURE} (instance_seg={is_instance_seg})")
    
    for dataset_name in dataset_names:
        try:
            # Get standardized paths for this dataset
            paths = _get_dataset_paths(dataset_name, coco_root)
            
            if is_instance_seg:
                # Instance segmentation: use load_coco_json and register_coco_instances
                print(f"Registering instance segmentation dataset: {dataset_name}")
                
                # Load dataset dicts using instance segmentation loader
                dataset_dicts = load_coco_json(
                    str(paths['instances_json']),
                    str(paths['image_root']),
                    dataset_name=dataset_name,
                    extra_annotation_keys=None
                )
                
                # Filter out problematic NPZ files (Python 3.11 encoding issues)
                print(f"Filtering problematic NPZ files for {dataset_name}...")
                filtered_dicts = _filter_problematic_npz_files(dataset_dicts, logger)
                print(f"After filtering: {len(filtered_dicts)} samples remaining for {dataset_name}")
                
                # Register using register_coco_instances
                register_coco_instances(
                    name=dataset_name,
                    metadata=meta,
                    json_file=str(paths['instances_json']),
                    image_root=str(paths['image_root'])
                )
                
                # Override the dataset dicts with filtered version
                def make_get_dataset(dicts):
                    def get_dataset():
                        return dicts
                    return get_dataset
                
                DatasetCatalog.register(dataset_name, make_get_dataset(filtered_dicts))
                
                # Set additional metadata
                MetadataCatalog.get(dataset_name).set(
                    image_root=paths['image_root'],
                    json_file=paths['instances_json'],
                    evaluator_type="coco",
                    **meta,
                )
                
            else:
                # Panoptic segmentation: use existing panoptic loader
                print(f"Registering panoptic segmentation dataset: {dataset_name}")
                
                # Load dataset dicts
                dataset_dicts = load_coco_panoptic_json(
                    paths['panoptic_json'],
                    paths['image_root'],
                    paths['panoptic_root'],
                    meta
                )
                
                # Filter out problematic NPZ files (Python 3.11 encoding issues)
                print(f"Filtering problematic NPZ files for {dataset_name}...")
                filtered_dicts = _filter_problematic_npz_files(dataset_dicts, logger)
                print(f"After filtering: {len(filtered_dicts)} samples remaining for {dataset_name}")
                
                # Register the filtered dataset
                # Create a closure to properly capture the filtered dataset_dicts for this dataset
                def make_get_dataset(dicts):
                    def get_dataset():
                        return dicts
                    return get_dataset
                
                DatasetCatalog.register(dataset_name, make_get_dataset(filtered_dicts))
                MetadataCatalog.get(dataset_name).set(
                    panoptic_root=paths['panoptic_root'],
                    image_root=paths['image_root'],
                    panoptic_json=paths['panoptic_json'],
                    json_file=paths['instances_json'],
                    evaluator_type="coco_panoptic_seg",
                    ignore_label=255,
                    label_divisor=1000,
                    **meta,
                )
            
            print(f"Successfully registered dataset: {dataset_name}")
            
        except Exception as e:
            logger.warning(f"Could not register COCO dataset {dataset_name}: {e}")
            print(f"Failed to register dataset {dataset_name}: {e}")


def shared_setup(args):
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Get custom metadata
    from trainer.metadata import get_metadata
    custom_metadata = get_metadata()

    # Setup config
    print(f"Setting up config.")
    cfg = get_cfg()
    cfg.set_new_allowed(True)  # Allow new keys to be added
    add_deeplab_config(cfg)
    print(f"Added deeplab config.")

    # Add Mask2Former config (only model type supported in main branch)
    print(f"Adding mask2former config.")
    from mask2former.config import add_maskformer2_config
    add_maskformer2_config(cfg)
    print(f"Added mask2former config.")

    print(f"Merging config file.")
    cfg.merge_from_file(args.config_file)
    print(f"Merged config file.")
    
    # only if doing pred/eval, load weights from args.weights, which may not exist for training
    print(f"Merging weights.")
    if hasattr(args, 'weights') and args.weights:
        cfg.merge_from_list(["MODEL.WEIGHTS", args.weights])
    print(f"Merged weights.")
    
    # Merge command-line options with better error handling
    if getattr(args, 'opts', None) and len(args.opts) > 0:
        try:
            cfg.merge_from_list(args.opts)
            print(f"Merged command-line options: {args.opts}")
        except (AssertionError, KeyError) as e:
            error_msg = str(e)
            if "Non-existent key" in error_msg:
                logger.error(f"Failed to merge command-line options: {error_msg}")
                logger.error(f"Options provided: {args.opts}")
                logger.error("Tip: Make sure all config keys exist. Use cfg.set_new_allowed(True) if you need to add new keys.")
                # Try to continue anyway - some keys might be optional
                logger.warning("Attempting to continue with partial config merge...")
            else:
                raise

    # Normalize COCO root to point at directory containing 'annotations/'
    coco_root_path = Path(args.coco_root)
    if not (coco_root_path / 'annotations').exists() and (coco_root_path / 'coco' / 'annotations').exists():
        coco_root_path = coco_root_path / 'coco'
    # Register datasets
    print(f"Registering datasets.")
    register_all_datasets(cfg, custom_metadata, str(coco_root_path), logger)
    print(f"Registered datasets.")

    # Freeze config
    print(f"Freezing config.")
    cfg.freeze()
    print(f"Frozen config.")
    default_setup(cfg, args)
    print(f"Default setup.")

    return cfg


