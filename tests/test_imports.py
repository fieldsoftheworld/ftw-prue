"""Smoke tests: verify all packages import and key classes are accessible."""

import importlib
from dataclasses import fields as dataclass_fields

import pytest


# ── Core packages ──────────────────────────────────────────────────────────


class TestFtwTools:
    def test_import_settings(self):
        from ftw_tools.settings import ALL_COUNTRIES, TEMPORAL_OPTIONS

        assert len(ALL_COUNTRIES) > 0
        assert len(TEMPORAL_OPTIONS) > 0

    def test_import_cli(self):
        from ftw_tools.cli import ftw

        assert callable(ftw)

    def test_import_segmentor(self):
        from ftw_tools.models.segmentor import SegmentationHead

        assert issubclass(SegmentationHead, __import__("torch").nn.Module)

    def test_import_losses(self):
        from ftw_tools.torchgeo.losses import logCoshDice, logCoshDiceCE

        assert callable(logCoshDice)
        assert callable(logCoshDiceCE)

    def test_import_metrics(self):
        from ftw_tools.postprocess.metrics import get_object_level_metrics

        assert callable(get_object_level_metrics)

    def test_import_baseline_inference(self):
        from ftw_tools.models.baseline_inference import predict

        assert callable(predict)

    def test_import_baseline_eval(self):
        from ftw_tools.models.baseline_eval import evaluate

        assert callable(evaluate)

    def test_import_trainers(self):
        from ftw_tools.torchgeo.trainers import CustomSemanticSegmentationTask

        assert hasattr(CustomSemanticSegmentationTask, "training_step")

    def test_import_datamodules(self):
        from ftw_tools.torchgeo.datamodules import preprocess

        assert callable(preprocess)

    def test_import_datasets(self):
        from ftw_tools.torchgeo.datasets import FTW

        assert callable(FTW)

    def test_import_utils(self):
        from ftw_tools.utils import validate_checksums, compute_md5

        assert callable(validate_checksums)
        assert callable(compute_md5)

    def test_import_model_utils(self):
        from pretrained.models.model_utils import get_preprocessor, prepare_clay_sample, prepare_general_sample

        assert callable(get_preprocessor)
        assert callable(prepare_clay_sample)
        assert callable(prepare_general_sample)

    def test_import_download_modules(self):
        import ftw_tools.download.download_img
        import ftw_tools.download.download_ftw
        import ftw_tools.download.unpack

        assert hasattr(ftw_tools.download.download_img, "__file__")

    def test_import_postprocess_polygonize(self):
        import ftw_tools.postprocess.polygonize

        assert hasattr(ftw_tools.postprocess.polygonize, "__file__")

    def test_import_postprocess_lulc_filtering(self):
        import ftw_tools.postprocess.lulc_filtering

        assert hasattr(ftw_tools.postprocess.lulc_filtering, "__file__")


# ── Pretrained encoders ───────────────────────────────────────────────────


class TestPretrainedImports:
    def test_import_path_config(self):
        from pretrained.path_config import get_model_path

        assert callable(get_model_path)

    def test_import_terrafm_encoder(self):
        from pretrained.models.TerraFM.terrafm_segment import TerraFMEncoderWrapper

        assert hasattr(TerraFMEncoderWrapper, "forward")

    def test_import_dinov3_encoder(self):
        from pretrained.models.dinov3.dinov3_segmentor import SegmentEncoder

        assert hasattr(SegmentEncoder, "forward")

    def test_import_terramind_encoder(self):
        pytest.importorskip("terratorch")
        from pretrained.models.terramind.terramind import SegmentEncoder

        assert hasattr(SegmentEncoder, "forward")

    def test_import_clay_encoder(self):
        # Clay's vendored src/ uses a relative import that requires the subpackage
        # to be discoverable — skip if the vendored Encoder isn't importable
        try:
            from pretrained.models.clay.finetune.segment.factory import SegmentEncoder
        except (ImportError, ModuleNotFoundError) as e:
            pytest.skip(f"Clay encoder not importable: {e}")
        assert hasattr(SegmentEncoder, "forward")

    def test_import_factory(self):
        from pretrained.pretrained_factory import get_encoder

        assert callable(get_encoder)


# ── DECODE ─────────────────────────────────────────────────────────────────


class TestDecodeImports:
    def test_import_fractal_resunet(self):
        from decode.fractal_resunet.models.semanticsegmentation.FracTAL_ResUNet import (
            FracTAL_ResUNet_cmtsk,
        )

        assert hasattr(FracTAL_ResUNet_cmtsk, "forward")

    def test_instantiate_default(self):
        import torch
        from decode.fractal_resunet.models.semanticsegmentation.FracTAL_ResUNet import (
            FracTAL_ResUNet_cmtsk,
        )

        model = FracTAL_ResUNet_cmtsk(
            nfilters_init=32, NClasses=2, depth=6, ftdepth=5, psp_depth=4,
            norm_type="GroupNorm", norm_groups=4, nheads_start=4, in_channels=8,
        )
        assert isinstance(model, torch.nn.Module)


# ── PRUE evaluation framework ─────────────────────────────────────────────


class TestPrueEvalImports:
    def test_import_top_level(self):
        from prue_eval import Detections, Evaluator, SemanticOutput, InstanceOutput, PanopticOutput

        assert hasattr(Detections, "from_semantic_logits")
        assert hasattr(Evaluator, "evaluate")
        # dataclass fields are not class attributes — check field names
        sem_fields = {f.name for f in dataclass_fields(SemanticOutput)}
        assert "logits" in sem_fields
        inst_fields = {f.name for f in dataclass_fields(InstanceOutput)}
        assert "masks" in inst_fields
        pan_fields = {f.name for f in dataclass_fields(PanopticOutput)}
        assert "seg_map" in pan_fields

    def test_import_converters(self):
        from prue_eval.converters import (
            convert_baseline_output,
            convert_decode_output,
            convert_sam_output,
            convert_delineate_anything_output,
            convert_d2_panoptic_output,
        )

        for fn in [
            convert_baseline_output,
            convert_decode_output,
            convert_sam_output,
            convert_delineate_anything_output,
            convert_d2_panoptic_output,
        ]:
            assert callable(fn)

    def test_import_registry(self):
        from prue_eval.models.registry import (
            Segmenter,
            register_model,
            create_segmenter,
            available_models,
        )

        assert callable(register_model)
        assert callable(create_segmenter)
        assert isinstance(available_models(), dict)

    def test_intermediate_formats_roundtrip(self):
        import numpy as np
        from prue_eval.intermediate_formats import SemanticOutput, InstanceOutput, PanopticOutput

        sem = SemanticOutput(logits=np.zeros((2, 8, 8)))
        assert sem.logits.shape == (2, 8, 8)

        inst = InstanceOutput(masks=np.zeros((3, 8, 8), dtype=np.uint8), scores=np.array([0.9, 0.8, 0.7]))
        assert inst.num_instances == 3

        pan = PanopticOutput(
            seg_map=np.zeros((8, 8), dtype=np.int32),
            segments_info=[{"id": 1, "category_id": 0, "isthing": True}],
        )
        assert pan.seg_map.shape == (8, 8)

    def test_detections_empty(self):
        import numpy as np
        from prue_eval.detections import Detections

        d = Detections(xyxy=np.empty((0, 4)))
        assert len(d) == 0

    def test_detections_from_semantic(self):
        import numpy as np
        from prue_eval.intermediate_formats import SemanticOutput
        from prue_eval.detections import Detections

        # 2-class logits: background vs field
        logits = np.zeros((2, 16, 16), dtype=np.float32)
        logits[1, 4:12, 4:12] = 1.0  # field region
        logits[0] = 1.0 - logits[1]

        sem = SemanticOutput(logits=logits)
        dets = Detections.from_semantic_logits(sem, field_class_id=1)
        assert len(dets) > 0

    def test_convert_baseline_output(self):
        import numpy as np
        from prue_eval.converters import convert_baseline_output

        logits = np.random.randn(2, 32, 32).astype(np.float32)
        out = convert_baseline_output(logits)
        assert out.logits.shape == (2, 32, 32)
        # Should be softmaxed (sums to ~1)
        sums = out.logits.sum(axis=0)
        np.testing.assert_allclose(sums, 1.0, atol=0.01)

    def test_convert_decode_output(self):
        import numpy as np
        from prue_eval.converters import convert_decode_output

        seg = np.random.randn(1, 2, 32, 32).astype(np.float32)
        bnd = np.random.randn(1, 2, 32, 32).astype(np.float32)
        dist = np.random.randn(1, 1, 32, 32).astype(np.float32)
        out = convert_decode_output((seg, bnd, dist))
        assert out.logits.shape == (2, 32, 32)
        assert "boundary_logits" in out.auxiliary

    def test_evaluator_pixel_metrics(self):
        import numpy as np
        from prue_eval.evaluator import get_pixel_level_metrics

        gt = np.zeros((32, 32), dtype=np.uint8)
        gt[8:24, 8:24] = 1
        pred = np.zeros((32, 32), dtype=np.uint8)
        pred[10:22, 10:22] = 1

        metrics = get_pixel_level_metrics(gt, pred)
        assert "pixel_accuracy" in metrics
        assert "mean_iou" in metrics
        assert "pixel_f1_field" in metrics
        assert 0 < metrics["pixel_iou_field"] < 100

    def test_evaluator_object_metrics(self):
        import numpy as np
        from prue_eval.evaluator import get_object_level_metrics_from_semantic_masks

        gt = np.zeros((32, 32), dtype=np.uint8)
        gt[4:12, 4:12] = 1
        gt[18:28, 18:28] = 1

        pred = np.zeros((32, 32), dtype=np.uint8)
        pred[5:11, 5:11] = 1  # overlaps first gt
        pred[2:6, 20:28] = 1  # false positive

        tps, fps, fns = get_object_level_metrics_from_semantic_masks(gt, pred)
        assert tps >= 0
        assert fps >= 0
        assert fns >= 0
        assert tps + fns == 2  # 2 GT objects

    def test_import_decode_adapter(self):
        from prue_eval.models.decode.segmenter import create_decode_segmenter

        assert callable(create_decode_segmenter)

    def test_import_da_adapter(self):
        pytest.importorskip("ftw.models.delineate_anything")
        from prue_eval.models.delineate_anything.segmenter import create_delineate_anything_segmenter

        assert callable(create_delineate_anything_segmenter)

    def test_import_sam_adapter(self):
        pytest.importorskip("segment_anything")
        from prue_eval.models.sam.segmenter import create_sam_segmenter

        assert callable(create_sam_segmenter)

    def test_import_d2_adapter(self):
        pytest.importorskip("detectron2.config")
        from prue_eval.models.d2 import create_mask2former_segmenter

        assert callable(create_mask2former_segmenter)


# ── Trainer (lazy imports) ─────────────────────────────────────────────────


class TestTrainerImports:
    def test_import_lr_schedulers(self):
        pytest.importorskip("detectron2.solver")
        from trainer import StepDecayLRScheduler, CosineWarmupLRScheduler

        assert callable(StepDecayLRScheduler)
        assert callable(CosineWarmupLRScheduler)

    def test_import_io_utils(self):
        from trainer.io_utils import read_geotiff

        assert callable(read_geotiff)

    def test_import_postprocessing(self):
        import trainer.postprocessing
        import trainer.postprocessing_pixel

        assert hasattr(trainer.postprocessing, "__file__")
        assert hasattr(trainer.postprocessing_pixel, "__file__")

    def test_import_trainer_detectron2_modules(self):
        pytest.importorskip("detectron2.engine")
        import trainer.prediction
        import trainer.custom_trainer
        import trainer.evaluation

        assert hasattr(trainer.prediction, "__file__")


# ── Mask2Former (requires compiled CUDA ops) ──────────────────────────────


class TestMask2FormerImports:
    @pytest.mark.skipif(
        not importlib.util.find_spec("detectron2"),
        reason="detectron2 not installed",
    )
    def test_import_mask2former_config(self):
        try:
            from mask2former.config import add_maskformer2_config
        except ModuleNotFoundError:
            pytest.skip("mask2former CUDA ops not compiled")
        assert callable(add_maskformer2_config)


# ── SAM2 (optional) ───────────────────────────────────────────────────────


class TestSAM2Imports:
    @pytest.mark.skipif(
        not importlib.util.find_spec("sam2"),
        reason="sam2 not installed",
    )
    def test_import_sam2(self):
        import sam2

        assert hasattr(sam2, "__file__")
