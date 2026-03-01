# Trigger adapter registration on import
from .ftw.segmenter import create_ftw_segmenter  # noqa: F401
from .sam.segmenter import create_sam_segmenter  # noqa: F401
from .decode.segmenter import create_decode_segmenter  # noqa: F401
from .delineate_anything.segmenter import (  # noqa: F401
    create_delineate_anything_segmenter,
    create_da_segmenter,
)
from .d2 import create_mask2former_segmenter  # noqa: F401
