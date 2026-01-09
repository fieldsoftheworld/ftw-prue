"""
Lightweight registry and interface for segmentation backends used in PRUE.

This sits at the *front* of the existing pipeline:

    Segmenter.predict(...) → IntermediateOutput (Semantic/Instance/Panoptic)
        → Detections → Evaluator

Backends (FTW baselines, SAM, DECODE, DA, Mask2Former, etc.) should expose a
small adapter that implements the Segmenter protocol and registers itself here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Protocol, Union, runtime_checkable

from intermediate_formats import SemanticOutput, InstanceOutput, PanopticOutput


# Type alias for any of the existing intermediate output types.
IntermediateOutput = Union[SemanticOutput, InstanceOutput, PanopticOutput]


@runtime_checkable
class Segmenter(Protocol):
    """
    Minimal interface all segmentation backends should implement.

    The intent is *not* to prescribe training APIs, only the runtime prediction
    behavior that the evaluation pipeline relies on.
    """

    model_name: str

    def predict(self, batch: Any) -> Iterable[IntermediateOutput]:
        """
        Run inference on a batch of inputs and yield intermediate outputs.

        The exact batch type is backend-specific (e.g., a dict from a Dataset,
        a torch.Tensor of images, etc.), but the result must always be an
        iterable of SemanticOutput, InstanceOutput, or PanopticOutput objects.
        """
        ...


@dataclass
class ModelEntry:
    """
    Metadata and factory for a registered model backend.

    The factory is typically a callable like:

        def create_segmenter(**kwargs) -> Segmenter: ...
    """

    name: str
    family: str  # e.g. "ftw", "sam", "decode", "delineate_anything", "d2"
    create: Callable[..., Segmenter]


# Global registry: maps model key (e.g. "ftw", "sam") to a ModelEntry.
_REGISTRY: Dict[str, ModelEntry] = {}


def register_model(name: str, family: str) -> Callable[[Callable[..., Segmenter]], Callable[..., Segmenter]]:
    """
    Decorator to register a Segmenter factory under a given key.

    Example:

        @register_model("ftw", family="ftw")
        def create_ftw_segmenter(**kwargs) -> Segmenter:
            ...
            return FtWSegmenter(...)
    """

    def decorator(factory: Callable[..., Segmenter]) -> Callable[..., Segmenter]:
        # Allow re-registration if same function name and family (handles module re-imports)
        # Normalize module names to handle different import paths (e.g., "models.ftw" vs "src.models.ftw")
        if name in _REGISTRY:
            existing = _REGISTRY[name]
            existing_name = getattr(existing.create, "__name__", None)
            new_name = getattr(factory, "__name__", None)
            
            # If same function name and family, allow re-registration (idempotent)
            if existing_name == new_name and existing.family == family:
                # Same factory being re-registered - update and return
                _REGISTRY[name] = ModelEntry(name=name, family=family, create=factory)
                return factory
            else:
                existing_module = getattr(existing.create, "__module__", None)
                new_module = getattr(factory, "__module__", None)
                raise ValueError(
                    f"Model '{name}' is already registered. "
                    f"Existing: {existing_module}.{existing_name} (family={existing.family}), "
                    f"New: {new_module}.{new_name} (family={family})"
                )
        _REGISTRY[name] = ModelEntry(name=name, family=family, create=factory)
        return factory

    return decorator


def available_models() -> Dict[str, ModelEntry]:
    """
    Return a shallow copy of the registry for introspection or CLI help.
    """
    return dict(_REGISTRY)


def create_segmenter(name: str, **kwargs: Any) -> Segmenter:
    """
    Instantiate a registered Segmenter by name.

    This is the primary entry point scripts should eventually use instead of
    ad-hoc model selection logic.
    """
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys())) or "<none registered yet>"
        raise ValueError(f"Unknown model '{name}'. Available models: {available}")
    entry = _REGISTRY[name]
    return entry.create(**kwargs)


