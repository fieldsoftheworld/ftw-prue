"""
Custom learning rate schedulers for Mask2Former training.

Mask2Former Defaults:
- COCO: WarmupMultiStepLR (default Detectron2 scheduler, uses SOLVER.STEPS)
- Cityscapes/ADE20K/Mapillary: WarmupPolyLR

This module provides:
1. CosineWarmupLRScheduler - Cosine annealing with warmup
2. StepDecayLRScheduler - Regular step decay with warmup
3. WarmupMultiStepLRScheduler - Explicit wrapper for Mask2Former default (COCO)
4. WarmupPolyLRScheduler - Polynomial decay with warmup (Cityscapes/ADE20K)
5. ExponentialLRScheduler - Exponential decay with warmup
6. LinearLRScheduler - Linear decay with warmup

All schedulers fall back to Detectron2's build_lr_scheduler if not explicitly named.
"""

from typing import Optional
from detectron2.solver import LRMultiplier, WarmupParamScheduler, build_lr_scheduler as detectron2_build_lr_scheduler
from fvcore.common.param_scheduler import (
    MultiStepParamScheduler,
    LinearParamScheduler,
    CompositeParamScheduler,
    CosineParamScheduler,
    ParamScheduler,
)


class CosineWarmupLRScheduler:
    """
    Cosine annealing learning rate scheduler with warmup.

    This scheduler:
    1. Starts at warmup_factor * base_lr
    2. Linearly increases to base_lr over warmup_iters
    3. Decays following cosine curve to min_factor * base_lr

    Config requirements:
    - SOLVER.COSINE.MIN_FACTOR: Final LR multiplier (default: 0.01)
    - SOLVER.WARMUP_FACTOR: Initial warmup multiplier
    - SOLVER.WARMUP_ITERS: Number of warmup iterations
    - SOLVER.MAX_ITER: Total training iterations
    """

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        warmup_length = cfg.SOLVER.WARMUP_ITERS / cfg.SOLVER.MAX_ITER
        min_factor = cfg.SOLVER.COSINE.MIN_FACTOR

        cosine_scheduler = CosineParamScheduler(1.0, min_factor)
        lr_scheduler = WarmupParamScheduler(
            cosine_scheduler,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_length=warmup_length,
            warmup_method="linear",
            rescale_interval=True,
        )

        return LRMultiplier(
            optimizer,
            lr_scheduler,
            max_iter=cfg.SOLVER.MAX_ITER,
        )


class StepDecayLRScheduler:
    """
    Step decay learning rate scheduler with warmup.

    This scheduler:
    1. Starts at warmup_factor * base_lr
    2. Linearly increases to base_lr over warmup_iters
    3. Decays by gamma every step_size iterations

    Config requirements:
    - SOLVER.STEP_DECAY.GAMMA: Decay factor (e.g., 0.5 = halve LR each step)
    - SOLVER.STEP_DECAY.STEP_SIZE: Iterations between decays
    - SOLVER.WARMUP_FACTOR: Initial warmup multiplier
    - SOLVER.WARMUP_ITERS: Number of warmup iterations
    - SOLVER.MAX_ITER: Total training iterations
    """

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        # Get parameters from config
        warmup_factor = cfg.SOLVER.WARMUP_FACTOR
        warmup_iters = cfg.SOLVER.WARMUP_ITERS
        max_iters = cfg.SOLVER.MAX_ITER
        gamma = cfg.SOLVER.STEP_DECAY.GAMMA
        step_size = cfg.SOLVER.STEP_DECAY.STEP_SIZE  # New config parameter for step size

        # Create warmup scheduler
        warmup = LinearParamScheduler(
            start_value=warmup_factor,
            end_value=1.0,
        )

        # Generate regular step intervals starting after warmup period
        steps = []
        current_step = warmup_iters + step_size
        while current_step < max_iters:
            steps.append(current_step)
            current_step += step_size

        # Create decay values - one more than the number of steps
        num_decays = len(steps)
        decay_values = [1.0]  # Initial LR multiplier
        for i in range(num_decays):
            decay_values.append(gamma ** (i + 1))

        # Create step decay scheduler
        decay = MultiStepParamScheduler(values=decay_values, milestones=steps, num_updates=max_iters)

        # Combine warmup and decay schedulers
        warmup_length = warmup_iters / max_iters
        decay_length = 1.0 - warmup_length

        scheduler = CompositeParamScheduler(
            schedulers=[warmup, decay],
            lengths=[warmup_length, decay_length],
            interval_scaling=["rescaled", "fixed"],
        )

        return LRMultiplier(
            optimizer=optimizer,
            multiplier=scheduler,
            max_iter=max_iters,
        )


class WarmupMultiStepLRScheduler:
    """
    WarmupMultiStepLR - The default Mask2Former scheduler for COCO.

    This is the standard Detectron2 scheduler that Mask2Former uses by default.
    It uses SOLVER.STEPS to define milestone iterations for LR decay.

    This wrapper makes it explicit when you want to use the default Mask2Former scheduler.

    Config requirements:
    - SOLVER.STEPS: Tuple of iteration milestones for LR decay (e.g., (327778, 355092))
    - SOLVER.WARMUP_FACTOR: Initial warmup multiplier (default: 1.0)
    - SOLVER.WARMUP_ITERS: Number of warmup iterations (default: 10)
    - SOLVER.MAX_ITER: Total training iterations
    - SOLVER.GAMMA: LR decay factor at each step (default: 0.1)

    Note: If LR_SCHEDULER_NAME is not set, Detectron2 automatically uses this
    when SOLVER.STEPS is provided.
    """

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        # Delegate to Detectron2's default implementation
        return detectron2_build_lr_scheduler(cfg, optimizer)


class WarmupPolyLRScheduler:
    """
    Polynomial decay learning rate scheduler with warmup.

    Used by Mask2Former for Cityscapes, ADE20K, and Mapillary Vistas datasets.
    The LR follows a polynomial decay: (1 - iter/max_iter)^power

    This scheduler:
    1. Starts at warmup_factor * base_lr
    2. Linearly increases to base_lr over warmup_iters
    3. Decays polynomially to end_factor * base_lr

    Config requirements:
    - SOLVER.POLY_LR_POWER: Polynomial power (default: 0.9)
    - SOLVER.POLY_LR_CONSTANT_ENDING: End factor (default: 0.0)
    - SOLVER.WARMUP_FACTOR: Initial warmup multiplier
    - SOLVER.WARMUP_ITERS: Number of warmup iterations
    - SOLVER.MAX_ITER: Total training iterations

    Note: Uses Detectron2's DeepLab WarmupPolyLR implementation.
    """

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        # Try to use Detectron2's DeepLab WarmupPolyLR implementation
        # Try both import paths (case sensitivity issues)
        WarmupPolyLR = None
        try:
            # First try the standard import path (lowercase)
            from detectron2.projects.deeplab.lr_scheduler import WarmupPolyLR
        except ImportError:
            try:
                # Fallback to direct path (capitalized)
                from detectron2.projects.DeepLab.deeplab.lr_scheduler import WarmupPolyLR
            except ImportError:
                pass  # Will use fallback implementation

        if WarmupPolyLR is not None:
            # Use Detectron2's implementation
            power = getattr(cfg.SOLVER, "POLY_LR_POWER", 0.9)
            constant_ending = getattr(cfg.SOLVER, "POLY_LR_CONSTANT_ENDING", 0.0)
            warmup_factor = cfg.SOLVER.WARMUP_FACTOR
            warmup_iters = cfg.SOLVER.WARMUP_ITERS
            warmup_method = getattr(cfg.SOLVER, "WARMUP_METHOD", "linear")

            return WarmupPolyLR(
                optimizer,
                max_iters=cfg.SOLVER.MAX_ITER,
                warmup_factor=warmup_factor,
                warmup_iters=warmup_iters,
                warmup_method=warmup_method,
                power=power,
                constant_ending=constant_ending,
            )
        else:
            # Fallback: implement polynomial decay using fvcore schedulers
            from detectron2.solver.lr_scheduler import WarmupParamScheduler
            from fvcore.common.param_scheduler import ParamScheduler

            class PolynomialParamScheduler(ParamScheduler):
                """Simple polynomial scheduler: (1 - t)^power"""

                def __init__(self, power=0.9, constant_ending=0.0):
                    self.power = power
                    self.constant_ending = constant_ending
                    # ParamScheduler is a callable interface, no super().__init__() needed

                def __call__(self, t):
                    """Return multiplier at normalized time t (0.0 to 1.0)"""
                    if self.constant_ending > 0 and t >= 1.0:
                        return self.constant_ending
                    return (1.0 - t) ** self.power

            power = getattr(cfg.SOLVER, "POLY_LR_POWER", 0.9)
            constant_ending = getattr(cfg.SOLVER, "POLY_LR_CONSTANT_ENDING", 0.0)

            poly_scheduler = PolynomialParamScheduler(power=power, constant_ending=constant_ending)

            warmup_length = cfg.SOLVER.WARMUP_ITERS / cfg.SOLVER.MAX_ITER
            lr_scheduler = WarmupParamScheduler(
                poly_scheduler,
                warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
                warmup_length=warmup_length,
                warmup_method="linear",
                rescale_interval=True,
            )

            return LRMultiplier(
                optimizer,
                lr_scheduler,
                max_iter=cfg.SOLVER.MAX_ITER,
            )


class ExponentialLRScheduler:
    """
    Exponential decay learning rate scheduler with warmup.

    This scheduler:
    1. Starts at warmup_factor * base_lr
    2. Linearly increases to base_lr over warmup_iters
    3. Exponentially decays: lr = base_lr * gamma ^ (iter / step_size)

    Config requirements:
    - SOLVER.EXPONENTIAL.GAMMA: Decay factor per step (e.g., 0.95)
    - SOLVER.EXPONENTIAL.STEP_SIZE: Iterations per decay step
    - SOLVER.WARMUP_FACTOR: Initial warmup multiplier
    - SOLVER.WARMUP_ITERS: Number of warmup iterations
    - SOLVER.MAX_ITER: Total training iterations
    """

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        from detectron2.solver.lr_scheduler import WarmupParamScheduler

        gamma = cfg.SOLVER.EXPONENTIAL.GAMMA
        step_size = cfg.SOLVER.EXPONENTIAL.STEP_SIZE
        max_iters = cfg.SOLVER.MAX_ITER
        warmup_iters = cfg.SOLVER.WARMUP_ITERS

        class ExponentialParamScheduler(ParamScheduler):
            """Exponential decay scheduler: gamma ^ (iter / step_size)"""

            def __init__(self, gamma, step_size, max_iters):
                self.gamma = gamma
                self.step_size = step_size
                self.max_iters = max_iters

            def __call__(self, t):
                """Return multiplier at normalized time t (0.0 to 1.0)"""
                # t is normalized [0, 1], convert to iteration number
                iter_num = t * self.max_iters
                # Continuous exponential decay: gamma ^ (iter / step_size)
                num_steps = iter_num / self.step_size
                return self.gamma**num_steps

        exp_scheduler = ExponentialParamScheduler(gamma=gamma, step_size=step_size, max_iters=max_iters)

        warmup_length = warmup_iters / max_iters
        lr_scheduler = WarmupParamScheduler(
            exp_scheduler,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_length=warmup_length,
            warmup_method="linear",
            rescale_interval=True,
        )

        return LRMultiplier(
            optimizer,
            lr_scheduler,
            max_iter=max_iters,
        )


class LinearLRScheduler:
    """
    Linear decay learning rate scheduler with warmup.

    This scheduler:
    1. Starts at warmup_factor * base_lr
    2. Linearly increases to base_lr over warmup_iters
    3. Linearly decays from base_lr to min_factor * base_lr

    Config requirements:
    - SOLVER.LINEAR.MIN_FACTOR: Final LR multiplier (default: 0.0)
    - SOLVER.WARMUP_FACTOR: Initial warmup multiplier
    - SOLVER.WARMUP_ITERS: Number of warmup iterations
    - SOLVER.MAX_ITER: Total training iterations
    """

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        min_factor = getattr(cfg.SOLVER.LINEAR, "MIN_FACTOR", 0.0)
        max_iters = cfg.SOLVER.MAX_ITER
        warmup_iters = cfg.SOLVER.WARMUP_ITERS

        # Create linear decay scheduler (from 1.0 to min_factor)
        linear_scheduler = LinearParamScheduler(
            start_value=1.0,
            end_value=min_factor,
        )

        warmup_length = warmup_iters / max_iters
        lr_scheduler = WarmupParamScheduler(
            linear_scheduler,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_length=warmup_length,
            warmup_method="linear",
            rescale_interval=True,
        )

        return LRMultiplier(
            optimizer,
            lr_scheduler,
            max_iter=max_iters,
        )


# Registry mapping scheduler names to their builders
SCHEDULER_REGISTRY = {
    "CosineWarmup": CosineWarmupLRScheduler,
    "StepDecay": StepDecayLRScheduler,
    "WarmupMultiStepLR": WarmupMultiStepLRScheduler,
    "WarmupPolyLR": WarmupPolyLRScheduler,
    "Exponential": ExponentialLRScheduler,
    "Linear": LinearLRScheduler,
}


def build_lr_scheduler(cfg, optimizer):
    """
    Build learning rate scheduler from config.

    If SOLVER.LR_SCHEDULER_NAME is set, uses the corresponding custom scheduler.
    Otherwise, falls back to Detectron2's default build_lr_scheduler.

    Args:
        cfg: Detectron2 config object
        optimizer: PyTorch optimizer

    Returns:
        Learning rate scheduler
    """
    scheduler_name = getattr(cfg.SOLVER, "LR_SCHEDULER_NAME", None)

    if scheduler_name and scheduler_name in SCHEDULER_REGISTRY:
        return SCHEDULER_REGISTRY[scheduler_name].build_lr_scheduler(cfg, optimizer)
    else:
        # Fall back to Detectron2's default (WarmupMultiStepLR when STEPS is provided)
        return detectron2_build_lr_scheduler(cfg, optimizer)
