__all__ = ["FTWEvaluator", "CustomTrainer", "StepDecayLRScheduler", "CosineWarmupLRScheduler"]


def __getattr__(name):
    """Lazy imports — mask2former requires compiled CUDA ops."""
    if name == "FTWEvaluator":
        from .evaluation import FTWEvaluator

        return FTWEvaluator
    if name == "CustomTrainer":
        from .custom_trainer import CustomTrainer

        return CustomTrainer
    if name in ("StepDecayLRScheduler", "CosineWarmupLRScheduler"):
        from .lr_schedulers import StepDecayLRScheduler, CosineWarmupLRScheduler

        return StepDecayLRScheduler if name == "StepDecayLRScheduler" else CosineWarmupLRScheduler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
