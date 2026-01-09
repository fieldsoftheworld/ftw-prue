# Trigger adapter registration on import
from .ftw.segmenter import create_ftw_segmenter  # noqa: F401
from .sam.segmenter import create_sam_segmenter  # noqa: F401
from .decode.segmenter import create_decode_segmenter  # noqa: F401