# Adapter registration — import errors are deferred for optional backends.
# Each adapter requires its own set of dependencies (segment_anything, detectron2, etc.)

from importlib import import_module
import logging

_log = logging.getLogger(__name__)

_ADAPTERS = [
    ("sam", ".sam.segmenter"),
    ("decode", ".decode.segmenter"),
    ("delineate_anything", ".delineate_anything.segmenter"),
    ("d2", ".d2"),
]

for _name, _module in _ADAPTERS:
    try:
        import_module(_module, package=__name__)
    except ImportError as e:
        _log.info("Skipping %s adapter (missing dependency): %s", _name, e)
    except Exception as e:
        _log.warning("Failed to load %s adapter: %s", _name, e)
