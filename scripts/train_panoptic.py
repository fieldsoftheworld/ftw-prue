#!/usr/bin/env python3
"""
Unified training script for panoptic segmentation models (Mask2Former),
for use with custom multispectral satellite imagery datasets.
"""

import os
import sys
import pickle
import numpy as np
from pathlib import Path

# Add project root to Python path BEFORE importing detectron2
# This ensures detectron2 (which is a subdirectory) can be found
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from shapely.errors import ShapelyDeprecationWarning
    import warnings

    warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
except ImportError:
    pass

import torch
import torch.nn as nn

# Import detectron2 - must be installed in the environment
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import default_argument_parser, launch
from detectron2.evaluation import verify_results
from detectron2.modeling import build_model

from scripts.setup import shared_setup
from trainer.custom_trainer import CustomTrainer
from trainer.log_system import log_system_info, print_resource_limits


def load_model_with_expanded_queries(cfg):
    """
    Builds a model and loads weights from a checkpoint, handling mismatches
    in query and input channel parameters by resizing and initializing them.
    This is particularly useful for fine-tuning on a different number of classes
    or with different input modalities than the original pre-trained model.
    """
    model = build_model(cfg)
    pretrained_weights_path = cfg.MODEL.WEIGHTS

    if not pretrained_weights_path:
        print("No pre-trained weights specified. Returning randomly initialized model.")
        return model

    print(f"Loading weights from {pretrained_weights_path}")

    with open(pretrained_weights_path, "rb") as f:
        checkpoint = pickle.load(f)

    # Extract model state dict
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    model_state_dict = model.state_dict()
    modified_state_dict = {}

    # Get query expansion setting from config
    query_expansion_enabled = getattr(cfg.MODEL.MASK_FORMER.QUERY_EXPANSION, "ENABLED", False)
    print(f"\nQuery expansion is {'enabled' if query_expansion_enabled else 'disabled'}")

    for name, param in state_dict.items():
        if name not in model_state_dict:
            print(f"Skipping {name}: not in current model.")
            continue

        if isinstance(param, np.ndarray):
            param = torch.from_numpy(param).float()

        target_shape = model_state_dict[name].shape
        if param.shape == target_shape:
            modified_state_dict[name] = param
            continue

        print(f"Mismatch for {name}: source {param.shape}, target {target_shape}. Handling...")

        # Handle backbone input layer for different channel counts
        if name == "backbone.patch_embed.proj.weight":
            # Adapts a 3-channel (RGB) pretrained weight to an 8-channel input
            if param.shape[1] == 3 and target_shape[1] == 8:
                new_param = torch.zeros(target_shape, dtype=param.dtype, device=param.device)
                new_param[:, 0:3, :, :] = param  # First date RGB
                new_param[:, 4:7, :, :] = param  # Second date RGB
                new_param[:, 3, :, :] = param[:, 0, :, :]  # First NIR from R
                new_param[:, 7, :, :] = param[:, 0, :, :]  # Second NIR from R
                modified_state_dict[name] = new_param
                print(f"  - Adapted input layer to 8 channels.")
                continue

            # Adapts a 3-channel (RGB) pretrained weight to a 4-channel input
            if param.shape[1] == 3 and target_shape[1] == 4:
                new_param = torch.zeros(target_shape, dtype=param.dtype, device=param.device)
                new_param[:, 0:3, :, :] = param  # First date RGB
                new_param[:, 3, :, :] = param[:, 0, :, :]  # First NIR from R
                modified_state_dict[name] = new_param
                print(f"  - Adapted input layer to 4 channels.")
                continue

        # Handle query expansion/contraction
        if query_expansion_enabled and ("query_feat" in name or "query_embed" in name):
            old_size, dim = param.shape
            new_size = target_shape[0]
            # Safeguard: Skip expansion if query sizes already match
            if old_size == new_size:
                print(f"  - Query sizes match ({old_size}), skipping expansion for {name}.")
                modified_state_dict[name] = param
                continue
            new_param = torch.zeros(target_shape, dtype=param.dtype, device=param.device)
            new_param[: min(old_size, new_size)] = param[: min(old_size, new_size)]
            if new_size > old_size:
                nn.init.normal_(new_param[old_size:], mean=0, std=0.01)
            modified_state_dict[name] = new_param
            print(f"  - Resized query from {old_size} to {new_size}.")
            continue

        if query_expansion_enabled and name == "sem_seg_head.predictor.query_feat.bias":
            old_size = param.shape[0]
            new_size = target_shape[0]
            # Safeguard: Skip expansion if query sizes already match
            if old_size == new_size:
                print(f"  - Query bias sizes match ({old_size}), skipping expansion for {name}.")
                modified_state_dict[name] = param
                continue
            new_param = torch.zeros(new_size, dtype=param.dtype, device=param.device)
            new_param[:old_size] = param
            modified_state_dict[name] = new_param
            print(f"  - Resized query bias from {old_size} to {new_size}.")
            continue

        if "class_embed" in name or "criterion.empty_weight" in name:
            print(f"  - Skipping {name} due to class count mismatch.")
            continue

        print(f"  - Skipping {name} due to unhandled shape mismatch.")

    incompatible = model.load_state_dict(modified_state_dict, strict=False)
    if incompatible.missing_keys:
        print(f"Missing keys: {len(incompatible.missing_keys)} keys")
        if len(incompatible.missing_keys) > 0:
            print(f"First few missing keys: {incompatible.missing_keys[:10]}")
    if incompatible.unexpected_keys:
        print(f"Unexpected keys: {len(incompatible.unexpected_keys)} keys")
        if len(incompatible.unexpected_keys) > 0:
            print(f"First few unexpected keys: {incompatible.unexpected_keys[:10]}")

    return model


def main(args):
    """Main training and evaluation logic."""
    print(f"Starting shared setup.")
    cfg = shared_setup(args)

    if args.eval_only:
        model = CustomTrainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
        res = CustomTrainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(CustomTrainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    # Check if query expansion is enabled in config
    query_expansion_enabled = getattr(cfg.MODEL.MASK_FORMER.QUERY_EXPANSION, "ENABLED", False)
    if query_expansion_enabled:
        model = load_model_with_expanded_queries(cfg)
        print(f"Model built with expanded queries. Resume: {args.resume}")
        trainer = CustomTrainer(cfg)
        # Replace the trainer's model with our custom-loaded model
        trainer.model = model
        trainer._checkpoint_dir = cfg.OUTPUT_DIR
        trainer.start_iter = 0
        trainer.optimizer = trainer.build_optimizer(cfg, trainer.model)
        trainer.scheduler = trainer.build_lr_scheduler(cfg, trainer.optimizer)
        # Initialize scheduler to avoid the warning
        trainer.scheduler.last_epoch = -1
        res = trainer.train()
        return res
    else:
        print(f"Building trainer.")
        trainer = CustomTrainer(cfg)
        print(f"Trainer built. Start iter: {trainer.start_iter}")
        trainer.resume_or_load(resume=args.resume)

    log_system_info()
    print_resource_limits()

    return trainer.train()


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--coco-root", type=str, required=True, help="Root directory of the COCO dataset.")
    parser.add_argument("--weights", type=str, help="Path to weights to load for prediction/evaluation.")
    args = parser.parse_args()

    print("Command Line Args:", args)

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
