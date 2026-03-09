import os, sys
import itertools
from typing import Any, Dict, List, Set
import copy
import logging
import random
import torch
import types
import time

import detectron2.utils.comm as comm
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader, build_detection_train_loader
from detectron2.engine import DefaultTrainer, hooks
from detectron2.evaluation import (
    COCOEvaluator,
    COCOPanopticEvaluator,
    FilteredCOCOPanopticEvaluator,
    SemSegEvaluator,
    DatasetEvaluators,
)
from detectron2.utils.events import EventWriter, get_event_storage
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.modeling import build_model, BACKBONE_REGISTRY, META_ARCH_REGISTRY
from fvcore.nn.precise_bn import get_bn_modules
from detectron2.checkpoint import DetectionCheckpointer

sys.path.insert(0, "../")
from mask2former.data.dataset_mappers import COCOPanopticNewBaselineDatasetMapperCustom, COCOInstanceCustomDatasetMapper
from .custom_hooks import CheckpointCleanupHook, ValidationHook, PredictionVisualizationHook
from .evaluation import FTWEvaluator
from mask2former.modeling.backbone.swin import D2SwinTransformer


class FilteredTensorboardXWriter(EventWriter):
    """
    TensorBoard writer that filters out individual loss components (e.g., loss_ce_0, loss_ce_1)
    and only logs aggregate losses (e.g., loss_ce, loss_mask, loss_dice).
    """

    def __init__(self, log_dir: str, window_size: int = 20, **kwargs):
        from torch.utils.tensorboard import SummaryWriter

        self._window_size = window_size
        self._writer = SummaryWriter(log_dir=log_dir, **kwargs)
        self._last_write = -1

        # Patterns to exclude (individual loss components with layer indices)
        self._exclude_patterns = [
            r"loss_ce_\d+$",  # loss_ce_0, loss_ce_1, etc.
            r"loss_mask_\d+$",  # loss_mask_0, loss_mask_1, etc.
            r"loss_dice_\d+$",  # loss_dice_0, loss_dice_1, etc.
        ]
        import re

        self._exclude_regex = [re.compile(pattern) for pattern in self._exclude_patterns]

    def _should_exclude(self, key: str) -> bool:
        """Check if a metric key should be excluded from logging."""
        for pattern in self._exclude_regex:
            if pattern.match(key):
                return True
        return False

    def write(self):
        storage = get_event_storage()
        new_last_write = self._last_write
        for k, (v, iter) in storage.latest_with_smoothing_hint(self._window_size).items():
            if iter > self._last_write:
                # Only log if not excluded
                if not self._should_exclude(k):
                    self._writer.add_scalar(k, v, iter)
                new_last_write = max(new_last_write, iter)
        self._last_write = new_last_write

        # Handle images and histograms (same as TensorboardXWriter)
        if len(storage._vis_data) >= 1:
            for img_name, img, step_num in storage._vis_data:
                self._writer.add_image(img_name, img, step_num)
            storage.clear_images()

        if len(storage._histograms) >= 1:
            for params in storage._histograms:
                self._writer.add_histogram_raw(**params)
            storage.clear_histograms()

    def close(self):
        if hasattr(self, "_writer"):
            self._writer.close()


class CustomTrainer(DefaultTrainer):
    """
    Custom Trainer class that extends the DefaultTrainer.
    Designed with COCO separated panoptic annotations of a multi-band satellite-image dataset in mind.
    """

    def build_writers(self):
        """
        Build a list of writers, using filtered TensorBoard writer to exclude individual loss components.
        """
        from detectron2.utils.events import CommonMetricPrinter, JSONWriter
        from detectron2.utils.file_io import PathManager
        import os

        output_dir = self.cfg.OUTPUT_DIR
        PathManager.mkdirs(output_dir)
        tb_writer = FilteredTensorboardXWriter(output_dir)  # Use filtered writer instead of default
        # Store reference to TensorBoard writer for hooks to access
        self._tb_writer = tb_writer._writer
        return [
            CommonMetricPrinter(self.max_iter),
            JSONWriter(os.path.join(output_dir, "metrics.json")),
            tb_writer,
        ]

    def build_hooks(self):
        """
        Build a list of default hooks, including mid-training validation and visualization hooks.
        """
        hooks = super().build_hooks()

        # Add checkpoint cleanup hook to keep only the latest 5 checkpoints
        hooks.append(CheckpointCleanupHook(checkpoint_dir=self.cfg.OUTPUT_DIR, keep_last=5))

        # Add mid-training validation hooks if eval period is set
        if self.cfg.TEST.EVAL_PERIOD > 0 and self.cfg.TEST.EVAL_PERIOD < self.cfg.SOLVER.MAX_ITER:
            # Build test data loader for mid-training validation
            test_loader = self.build_test_loader(self.cfg, self.cfg.DATASETS.TEST[0])

            # Add validation hook for loss logging during training
            hooks.append(
                ValidationHook(
                    self.cfg,
                    self.model,
                    test_loader,
                    tb_writer=None,  # Will be set up in before_train
                    period=self.cfg.TEST.EVAL_PERIOD,
                )
            )

            # Add prediction visualization hook for mid-training monitoring
            hooks.append(
                PredictionVisualizationHook(
                    self.cfg,
                    self.model,
                    test_loader,
                    metadata=None,  # Will be resolved from cfg in before_train
                    tb_writer=None,  # Will be set up in before_train
                    period=self.cfg.TEST.EVAL_PERIOD,
                    max_images=4,
                )
            )

        return hooks

    def after_train(self):
        """Clean up shared TensorBoard writer if it exists"""
        if hasattr(self, "_shared_tb_writer") and self._shared_tb_writer is not None:
            self._shared_tb_writer.close()
            self._shared_tb_writer = None
        super().after_train()

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for the dataset.
        """
        if output_folder is None:
            if not os.path.exists(cfg.OUTPUT_DIR):
                os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")

        # Get country names if available
        country_names = None
        if hasattr(cfg.DATASETS, "COUNTRIES_EVAL") and cfg.DATASETS.COUNTRIES_EVAL:
            country_names = cfg.DATASETS.COUNTRIES_EVAL.split(",")

        # # Create the unified satellite segmentation evaluator
        # evaluator = SatelliteSegEvaluator(
        #     dataset_name=dataset_name,
        #     tasks=["sem_seg", "instance", "panoptic"],
        #     distributed=True,
        #     output_dir=output_folder,
        #     country_names=country_names,
        #     use_fast_impl=True,
        #     log_visualization=True,
        #     max_dets_per_image=100  # Adjust based on your needs
        # )

        evaluator = FTWEvaluator(
            dataset_name=dataset_name,
            distributed=True,
            output_dir=output_folder,
            country_names=country_names,
            iou_threshold=0.5,
        )

        return evaluator

    @classmethod
    def build_train_loader(cls, cfg):
        if cfg.INPUT.DATASET_MAPPER_NAME == "coco_panoptic_custom":
            # specify augs here or elsewhere, passed into the mapper
            mapper = COCOPanopticNewBaselineDatasetMapperCustom(cfg, True)
            loader = build_detection_train_loader(cfg, mapper=mapper)
            return loader
        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_instance_custom":
            # Custom instance segmentation mapper for 8-channel satellite imagery
            mapper = COCOInstanceCustomDatasetMapper(cfg, True)
            loader = build_detection_train_loader(cfg, mapper=mapper)
            return loader
        else:
            mapper = None
            return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        if cfg.INPUT.DATASET_MAPPER_NAME == "coco_panoptic_custom":
            # # specify augs here or elsewhere, passed into the mapper
            # # NB: test time augs may need to be reworked to make sure annotations are transformed along with images
            mapper = COCOPanopticNewBaselineDatasetMapperCustom(cfg, False)
            dataset = DatasetCatalog.get(dataset_name)

            if cfg.TEST.EVAL_SUBSET_SIZE > 0:
                # Use subset if specified
                num_test_samples = min(cfg.TEST.EVAL_SUBSET_SIZE, len(dataset))
                test_indices = random.sample(range(len(dataset)), num_test_samples)
                dataset_subset = [dataset[i] for i in test_indices]

                logger = logging.getLogger(__name__)
                logger.setLevel(logging.INFO)
                logger.info(f"Using {num_test_samples} images for evaluation (full test set: {len(dataset)})")

                loader = build_detection_test_loader(
                    dataset_subset,
                    mapper=mapper,
                    num_workers=cfg.DATALOADER.NUM_WORKERS,
                    batch_size=cfg.TEST.BATCH_SIZE,
                )
            else:
                # Use full dataset
                loader = build_detection_test_loader(
                    cfg,
                    dataset_name,
                    mapper=mapper,
                    num_workers=cfg.DATALOADER.NUM_WORKERS,
                    batch_size=cfg.TEST.BATCH_SIZE,
                )

            return loader
        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_instance_custom":
            # Custom instance segmentation mapper for 8-channel satellite imagery
            mapper = COCOInstanceCustomDatasetMapper(cfg, False)
            dataset = DatasetCatalog.get(dataset_name)

            if cfg.TEST.EVAL_SUBSET_SIZE > 0:
                # Use subset if specified
                num_test_samples = min(cfg.TEST.EVAL_SUBSET_SIZE, len(dataset))
                test_indices = random.sample(range(len(dataset)), num_test_samples)
                dataset_subset = [dataset[i] for i in test_indices]

                logger = logging.getLogger(__name__)
                logger.setLevel(logging.INFO)
                logger.info(f"Using {num_test_samples} images for evaluation (full test set: {len(dataset)})")

                loader = build_detection_test_loader(
                    dataset_subset,
                    mapper=mapper,
                    num_workers=cfg.DATALOADER.NUM_WORKERS,
                    batch_size=cfg.TEST.BATCH_SIZE,
                )
            else:
                # Use full dataset
                loader = build_detection_test_loader(
                    cfg,
                    dataset_name,
                    mapper=mapper,
                    num_workers=cfg.DATALOADER.NUM_WORKERS,
                    batch_size=cfg.TEST.BATCH_SIZE,
                )

            return loader
        else:
            mapper = None
            return build_detection_test_loader(
                cfg, dataset_name, mapper=mapper, num_workers=cfg.DATALOADER.NUM_WORKERS, batch_size=cfg.TEST.BATCH_SIZE
            )

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        Build learning rate scheduler.

        Uses custom schedulers from trainer.lr_schedulers if LR_SCHEDULER_NAME is set,
        otherwise falls back to Detectron2's default (WarmupMultiStepLR for COCO).
        """
        from trainer.lr_schedulers import build_lr_scheduler

        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if "relative_position_bias_table" in module_param_name or "absolute_pos_embed" in module_param_name:
                    print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(params, cfg.SOLVER.BASE_LR)
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def build_full_test_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator for full test evaluation on all countries.
        This is similar to build_evaluator but includes all countries.
        """
        if output_folder is None:
            if not os.path.exists(cfg.OUTPUT_DIR):
                os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
            output_folder = os.path.join(cfg.OUTPUT_DIR, "full_test_inference")

        evaluator_list = []

        # Get the evaluator type from metadata
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        if evaluator_type == "coco_panoptic_seg":
            country_names = None
            if hasattr(cfg.DATASETS, "COUNTRIES_EVAL") and cfg.DATASETS.COUNTRIES_EVAL:
                country_names = cfg.DATASETS.COUNTRIES_EVAL.split(",")

            print(f"country names: {country_names}")

            # Create the unified satellite segmentation evaluator
            # evaluator_list.append(SatelliteSegEvaluator(
            #     dataset_name=dataset_name,
            #     tasks=["sem_seg", "instance", "panoptic"],
            #     distributed=True,
            #     output_dir=output_folder,
            #     country_names=country_names,
            #     use_fast_impl=True,
            #     log_visualization=True,
            #     max_dets_per_image=100  # Adjust based on your needs
            # ))

            evaluator_list.append(
                FTWEvaluator(
                    dataset_name=dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                    country_names=country_names,
                    iou_threshold=0.5,
                    metrics=["pixel", "object", "coco"],
                )
            )
            print("Built evaluator")

        if len(evaluator_list) == 0:
            raise NotImplementedError(
                f"No evaluator implementation found for dataset {dataset_name} with type {evaluator_type}"
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]

        return DatasetEvaluators(evaluator_list)

    @classmethod
    def full_test(cls, cfg, model=None, eval_output_dir=None):
        """
        Run evaluation on the full test set and all countries.

        Args:
            cfg: Configuration
            model: Model to evaluate (if None, will load from checkpoint)
            eval_output_dir: Directory to save evaluation results

        Returns:
            Dictionary with evaluation results
        """
        # Create output directory for evaluation
        if eval_output_dir is None:
            eval_output_dir = os.path.join(cfg.OUTPUT_DIR, "full_test_results")
        os.makedirs(eval_output_dir, exist_ok=True)

        # Clone and modify config to ensure we use full test set
        cfg = cfg.clone()
        cfg.defrost()
        cfg.TEST.EVAL_SUBSET_SIZE = 0  # Use full test set
        cfg.freeze()

        # Get model
        if model is None:
            model = cls.build_model(cfg)

            # Load checkpoint using Detectron2's Checkpointer
            from detectron2.checkpoint import DetectionCheckpointer

            checkpointer = DetectionCheckpointer(model, save_dir=eval_output_dir)
            checkpointer.load(cfg.MODEL.WEIGHTS)

        # Get data loader
        test_loader = cls.build_test_loader(cfg, cfg.DATASETS.TEST[0])

        # Get evaluator
        evaluator = cls.build_full_test_evaluator(cfg, cfg.DATASETS.TEST[0], output_folder=eval_output_dir)

        # Run evaluation
        results = cls.test_with_evaluator(model, test_loader, evaluator)

        # Save results
        if comm.is_main_process():
            with open(os.path.join(eval_output_dir, "full_test_results.json"), "w") as f:
                import json

                json.dump(results, f, indent=4)

            # Also create a pretty-printed version
            formatted_results = cls._format_results(results)
            with open(os.path.join(eval_output_dir, "full_test_summary.txt"), "w") as f:
                f.write(formatted_results)

        return results

    @staticmethod
    def _format_results(results):
        """Format results into a readable text summary"""
        lines = ["# Fields of The World - Full Test Evaluation Results\n"]

        # Overall metrics
        if "sem_seg" in results:
            lines.append("## Overall Semantic Segmentation Metrics\n")
            for key, value in results["sem_seg"].items():
                if not key.startswith("IoU-") and not key.startswith("ACC-") and not "/" in key:
                    lines.append(f"* {key}: {value:.2f}")
            lines.append("")

        # Country metrics
        country_metrics = {}
        for key, value in results.items():
            if "/" in key:
                country, metric = key.split("/", 1)
                if country not in country_metrics:
                    country_metrics[country] = {}
                country_metrics[country][metric] = value

        if country_metrics:
            lines.append("## Per-Country Metrics\n")
            for country, metrics in sorted(country_metrics.items()):
                lines.append(f"### {country.title()}\n")
                lines.append("| Metric | Value |")
                lines.append("| ------ | ----- |")
                for metric, value in sorted(metrics.items()):
                    lines.append(f"| {metric} | {value:.2f} |")
                lines.append("")

        return "\n".join(lines)

    @classmethod
    def test_with_evaluator(cls, model, data_loader, evaluator):
        """
        Run model on the data_loader and evaluate the metrics with evaluator.
        Also benchmark the inference speed of `model.forward` accurately.
        """
        import time
        import logging
        import sys

        # Configure root logger if not already configured
        if not logging.getLogger().handlers:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                handlers=[
                    logging.StreamHandler(sys.stdout),
                ],
            )

        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        # Force logger to use stdout
        if not logger.handlers:
            logger.addHandler(logging.StreamHandler(sys.stdout))

        model.eval()
        results = {}

        # Reset evaluator
        evaluator.reset()

        # Get target countries from evaluator if available
        target_countries = getattr(evaluator, "_country_names", None)

        with torch.no_grad():
            # Start timer
            total_compute_time = 0
            total_samples = 0

            logger.info("Starting full test evaluation. This may take a while...")

            # Add country filtering logic
            if target_countries:
                logger.info(f"Filtering evaluation to only include countries: {target_countries}")

            # Process in batches
            num_processed = 0
            num_skipped = 0

            for idx, inputs in enumerate(data_loader):
                # Filter inputs by country if target_countries is specified
                if target_countries:
                    filtered_inputs = []
                    for input_dict in inputs:
                        # Extract country from filename
                        file_name = input_dict.get("file_name", "")
                        basename = os.path.basename(file_name)
                        country = basename.split("_")[0].lower() if "_" in basename else None
                        if country == "south":  # quick hack to fix south_africa
                            country = "south_africa"

                        if country in target_countries:
                            filtered_inputs.append(input_dict)
                        else:
                            num_skipped += 1
                            if num_skipped <= 10:  # Log first few skips
                                logger.info(f"Skipping sample from {country} (not in target countries)")

                    # Skip this batch if no samples match target countries
                    if not filtered_inputs:
                        continue

                    inputs = filtered_inputs

                # Forward pass
                start_compute_time = time.perf_counter()
                # import pdb; pdb.set_trace()
                outputs = model(inputs)
                torch.cuda.synchronize()
                total_compute_time += time.perf_counter() - start_compute_time
                total_samples += len(inputs)

                # Process predictions
                evaluator.process(inputs, outputs)

                # Logging
                num_processed += len(inputs)
                if idx % 50 == 0:
                    if target_countries:
                        logger.info(
                            f"Processed {num_processed} samples, skipped {num_skipped} samples from non-target countries"
                        )
                    else:
                        logger.info(f"Processed {num_processed}/{len(data_loader.dataset)} samples")

        # Final logging
        if target_countries:
            logger.info(f"Evaluation complete: processed {num_processed} samples, skipped {num_skipped} samples")

        # Evaluate
        eval_results = evaluator.evaluate()

        # Add inference speed
        if total_samples > 0:
            results["inference_time_per_image"] = total_compute_time / total_samples

        # Combine results
        if isinstance(eval_results, dict):
            results.update(eval_results)

        return results
