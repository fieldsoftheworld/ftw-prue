# Adapter registration — import errors are deferred for optional backends.
# Each adapter requires its own set of dependencies (segment_anything, detectron2, etc.)

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
        __import__(_module, globals(), locals(), ["*"], level=1)
    except ImportError as e:
        _log.debug("Skipping %s adapter: %s", _name, e)
