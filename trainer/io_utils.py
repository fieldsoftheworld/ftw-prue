"""I/O utilities for reading satellite imagery."""

import numpy as np


def read_geotiff(file_name, format=None):
    """Read a GeoTIFF file and return as HWC numpy array.

    Replaces the non-existent detectron2.data.detection_utils.read_geotiff.
    """
    import rasterio

    with rasterio.open(file_name) as src:
        image = src.read()  # (C, H, W)
        image = np.transpose(image, (1, 2, 0))  # (H, W, C)
    return image
