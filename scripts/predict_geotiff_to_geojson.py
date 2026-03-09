#!/usr/bin/env python3
"""
Run inference on GeoTIFF images and convert results to GeoJSON.

This script processes multiple GeoTIFF files directly (without COCO format conversion)
and converts the panoptic segmentation predictions to GeoJSON format.

**Full pipeline:** Inference → Pixel-space filtering → Conversion → Geographic postprocessing → GeoJSON

Uses the modular postprocessing pipeline:
- Pixel-space filtering (postprocessing_pixel.py)
- Conversion to GeoDataFrame (postprocessing.py)
- Geographic postprocessing (postprocessing_geo.py)

**Recommended script** for inference + postprocessing with chipping support.

Usage:
    python predict_geotiff_to_geojson.py \
        --config-file path/to/config.yaml \
        --input path/to/geotiff/folder \
        --output path/to/output/dir \
        --weights path/to/model.pth \
        --chip-size 512 \
        --overlap 256 \
        --confidence-threshold 0.5 \
        --min-area 10.0 \
        --edge-buffer 50.0 \
        --min-area-pixels 100 \
        --cropland-mask path/to/cropland_mask.tif \
        --osm-waterways path/to/waterways.geojson \
        --osm-roads path/to/roads.geojson
"""

import os
import sys
import json
from pathlib import Path
import logging
import argparse
import numpy as np
import torch
import cv2
from tqdm import tqdm

from detectron2.config import CfgNode, get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data.detection_utils import read_geotiff
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.visualizer import Visualizer, ColorMode

# Add project root to Python path BEFORE importing custom modules
# This ensures trainer and mask2former modules can be found
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import your custom modules
from trainer.prediction import SatellitePredictor
from trainer.pred_visualization import SatelliteVisualizer, NoTextVisualizer
from mask2former.config import add_maskformer2_config
from trainer.metadata import get_metadata
from trainer.postprocessing import panoptic_to_geojson
from trainer.postprocessing_pixel import (
    filter_panoptic_segments_pixel_space,
    detect_edge_polygons,
)
from trainer.postprocessing_geo import (
    filter_edge_polygons_geo,
    resolve_overlaps,
    merge_overlapping_fields,
    filter_by_cropland_mask,
    add_osm_attributes,
)

logger = logging.getLogger(__name__)

def chip_image(image: np.ndarray, chip_size: int, overlap: int = 0) -> tuple:
    """
    Split a large image into smaller chips with optional overlap.
    
    Args:
        image: Input image of shape [C, H, W] or [H, W, C]
        chip_size: Size of each chip (square)
        overlap: Overlap between adjacent chips in pixels
        
    Returns:
        chips: List of image chips
        coords: List of (x, y) coordinates for the top-left corner of each chip
    """
    # Ensure image is in [H, W, C] format
    if image.shape[0] < image.shape[-1]:
        # Image is likely in [C, H, W] format, transpose to [H, W, C]
        image = np.transpose(image, (1, 2, 0))
    
    height, width = image.shape[:2]
    chips = []
    coords = []
    
    effective_size = chip_size - overlap
    
    for y in range(0, height, effective_size):
        for x in range(0, width, effective_size):
            # Make sure we don't go out of bounds
            y_end = min(y + chip_size, height)
            x_end = min(x + chip_size, width)
            
            # Adjust start position to maintain chip size when possible
            y_start = max(0, y_end - chip_size)
            x_start = max(0, x_end - chip_size)
            
            # Extract the chip
            chip = image[y_start:y_end, x_start:x_end]
            
            # Pad if chip is smaller than chip_size
            if chip.shape[0] < chip_size or chip.shape[1] < chip_size:
                padded_chip = np.zeros((chip_size, chip_size, image.shape[2]), dtype=image.dtype)
                padded_chip[:chip.shape[0], :chip.shape[1]] = chip
                chip = padded_chip
            
            chips.append(chip)
            coords.append((x_start, y_start))
    
    return chips, coords

# DEPRECATED: Old custom stitching functions removed.
# Now using modular postprocessing pipeline:
# 1. Pixel-space filtering (postprocessing_pixel.py)
# 2. Conversion to GeoDataFrame (postprocessing.py)
# 3. Geographic postprocessing (postprocessing_geo.py)

def process_single_geotiff(
    geotiff_path: str,
    predictor: SatellitePredictor,
    chip_size: int,
    overlap: int,
    confidence_threshold: float = 0.8,  # Default matches MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD
    min_area: float = 100.0,  # 100 m² = 0.01ha (from methods section)
    max_area: float = 90000.0,  # 90,000 m² = 9ha (from methods section)
    iou_threshold: float = 0.7,
    edge_buffer: float = 5.0,  # Reduced default - 50m was too large for small chips
    min_area_pixels: int = 10,  # Small default - just removes tiny fragments
    save_visualization: bool = False,
    output_dir: str = None,
    cropland_mask_path: str = None,
    osm_waterways_path: str = None,
    osm_roads_path: str = None,
) -> dict:
    """
    Process a single GeoTIFF file and return GeoJSON-compatible results.
    
    Uses the new modular postprocessing pipeline:
    1. Inference on chips
    2. Pixel-space filtering (confidence, category, area, edge detection)
    3. Conversion to GeoDataFrame (with georeferencing)
    4. Geographic postprocessing (edge filtering, overlap resolution)
    5. Optional: Cropland mask filtering, OSM attributes
    
    Args:
        geotiff_path: Path to GeoTIFF file
        predictor: Initialized predictor
        chip_size: Size of image chips
        overlap: Overlap between chips in pixels
        confidence_threshold: Minimum confidence for valid predictions (default: from config)
        min_area: Minimum area for valid fields (square meters, default: 100 m² = 0.01ha)
        max_area: Maximum area for valid fields (square meters, default: 90,000 m² = 9ha)
        iou_threshold: IoU threshold for merging overlapping predictions
        edge_buffer: Edge buffer in meters for filtering edge polygons
        min_area_pixels: Minimum area in pixels for pixel-space filtering (default: 10, just removes tiny fragments)
        save_visualization: Whether to save visualization images
        output_dir: Output directory for visualizations
        cropland_mask_path: Optional path to cropland mask GeoTIFF
        osm_waterways_path: Optional path to OSM waterways GeoJSON/Shapefile
        osm_roads_path: Optional path to OSM roads GeoJSON/Shapefile
        
    Returns:
        Dictionary with GeoJSON features and metadata
    """
    import rasterio
    
    logger.info(f"Processing {geotiff_path}")
    
    # Read GeoTIFF with rasterio to get transform and CRS
    with rasterio.open(geotiff_path) as src:
        transform = src.transform
        crs = src.crs
        bounds = src.bounds  # (minx, miny, maxx, maxy)
        logger.info(f"Image CRS: {crs}")
        logger.info(f"Image transform: {transform}")
    
    # Read image data
    image = read_geotiff(geotiff_path)
    logger.info(f"Image shape: {image.shape}")
    
    # Create output directory for this file (only if saving visualizations)
    geotiff_name = Path(geotiff_path).stem
    file_output_dir = None
    if output_dir and save_visualization:
        file_output_dir = Path(output_dir) / geotiff_name
        file_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Chip the image
    chips, coords = chip_image(image, chip_size, overlap)
    logger.info(f"Created {len(chips)} chips with {overlap}px overlap")
    
    # Process each chip: inference → pixel-space filtering → conversion → geographic filtering
    chip_geodataframes = []
    import rasterio
    from shapely.geometry import box
    import geopandas as gpd
    import pandas as pd
    
    for i, (chip, (x, y)) in enumerate(tqdm(zip(chips, coords), total=len(chips), desc=f"Processing {geotiff_name}")):
        try:
            # Step 1: Run inference
            predictions = predictor(chip)
            
            if "panoptic_seg" not in predictions:
                logger.debug(f"Chip {i} at ({x}, {y}): No panoptic_seg in predictions")
                continue
            
            panoptic_seg_tensor, segments_info = predictions["panoptic_seg"]
            panoptic_seg = panoptic_seg_tensor.cpu().numpy()
            
            logger.debug(f"Chip {i} at ({x}, {y}): Model produced {len(segments_info)} segments")
            
            # Step 2: Apply pixel-space filtering
            filtered_panoptic, filtered_segments = filter_panoptic_segments_pixel_space(
                panoptic_seg,
                segments_info,
                confidence_threshold=confidence_threshold,
                min_area_pixels=min_area_pixels,
                chip_size=(chip_size, chip_size),
                edge_threshold=5,
            )
            
            if len(filtered_segments) == 0:
                logger.debug(f"Chip {i} at ({x}, {y}): No segments after pixel-space filtering")
                continue
            
            logger.debug(f"Chip {i} at ({x}, {y}): {len(filtered_segments)} segments after pixel-space filtering")
            
            # Step 3: Calculate chip transform (georeferencing for this chip)
            # Get pixel coordinates in original image
            y_end = min(y + chip_size, image.shape[0])
            x_end = min(x + chip_size, image.shape[1])
            
            # Calculate geographic transform for this chip
            # Transform formula: [a, b, c, d, e, f] where:
            # x_geo = a * x_pixel + b * y_pixel + c
            # y_geo = d * x_pixel + e * y_pixel + f
            chip_transform = rasterio.Affine(
                transform[0],  # a: pixel width
                transform[1],  # b: rotation (usually 0)
                transform[2] + x * transform[0] + y * transform[1],  # c: x origin
                transform[3],  # d: rotation (usually 0)
                transform[4],  # e: pixel height (usually negative)
                transform[5] + x * transform[3] + y * transform[4]   # f: y origin
            )
            
            # Step 4: Convert to GeoDataFrame
            chip_gdf = panoptic_to_geojson(
                filtered_panoptic,
                filtered_segments,
                chip_transform,
                crs,
                min_area=min_area,
                max_area=max_area,
                confidence_threshold=None,  # Already filtered
                apply_pixel_filters=False,  # Already filtered
            )
            
            if chip_gdf.empty:
                logger.debug(f"Chip {i} at ({x}, {y}): No polygons after conversion (had {len(filtered_segments)} segments)")
                continue
            
            logger.debug(f"Chip {i} at ({x}, {y}): {len(chip_gdf)} polygons after conversion")
            
            logger.debug(f"Chip {i} at ({x}, {y}): {len(chip_gdf)} polygons before edge filtering")
            
            # Step 5: Apply geographic postprocessing (edge filtering)
            # Calculate chip bounds in geographic coordinates
            chip_bounds_geo = box(
                transform[2] + x * transform[0],
                transform[5] + y * transform[4],
                transform[2] + x_end * transform[0],
                transform[5] + y_end * transform[4]
            )
            
            polygons_before_edge = len(chip_gdf)
            chip_gdf = filter_edge_polygons_geo(
                chip_gdf,
                chip_bounds_geo,
                edge_buffer=edge_buffer,
            )
            polygons_after_edge = len(chip_gdf)
            
            if polygons_before_edge > 0 and polygons_after_edge == 0:
                logger.debug(f"Chip {i} at ({x}, {y}): Lost all {polygons_before_edge} polygons to edge filtering (buffer={edge_buffer}m)")
            elif polygons_after_edge < polygons_before_edge:
                logger.debug(f"Chip {i} at ({x}, {y}): Edge filtering removed {polygons_before_edge - polygons_after_edge}/{polygons_before_edge} polygons")
            
            if not chip_gdf.empty:
                chip_geodataframes.append(chip_gdf)
            
        except AttributeError as e:
            if "thing_dataset_id_to_contiguous_id" in str(e):
                logger.error(f"Metadata error in chip {i} at ({x}, {y}): {str(e)}")
                logger.error("This usually means the metadata is not properly set up. Check dataset registration.")
                continue
            else:
                logger.error(f"Attribute error in chip {i} at ({x}, {y}): {str(e)}")
                continue
        except Exception as e:
            logger.error(f"Error processing chip {i} at ({x}, {y}): {str(e)}")
            import traceback
            logger.debug(traceback.format_exc())
            continue
    
    if not chip_geodataframes:
        logger.warning(f"No valid predictions for {geotiff_path}")
        logger.warning(f"Processed {len(chips)} chips but none produced valid polygons after filtering")
        logger.warning(f"Consider: reducing --edge-buffer (current: {edge_buffer}m), reducing --min-area (current: {min_area}m²), or checking model predictions")
        # Still return bounds even if no predictions
        return {"features": [], "crs": crs, "bounds": bounds, "source_file": geotiff_path}
    
    logger.info(f"Successfully processed {len(chip_geodataframes)} chips with valid predictions")
    
    # Step 6: Combine all chip GeoDataFrames
    logger.info("Combining chip predictions")
    try:
        combined_gdf = pd.concat(chip_geodataframes, ignore_index=True)
        logger.info(f"Combined {len(combined_gdf)} polygons from {len(chip_geodataframes)} chips")
        
        # Step 7: Resolve overlaps (geographic operations)
        logger.info("Resolving overlaps")
        resolved_gdf = resolve_overlaps(
            combined_gdf,
            overlap_threshold=iou_threshold,
            keep_higher_confidence=True,
        )
        logger.info(f"After overlap resolution: {len(resolved_gdf)} polygons")
        
        # Step 8: Merge overlapping fields if needed
        if len(resolved_gdf) > 0:
            merged_gdf = merge_overlapping_fields(
                resolved_gdf,
                iou_threshold=iou_threshold,
            )
            logger.info(f"After merging: {len(merged_gdf)} polygons")
        else:
            merged_gdf = resolved_gdf
        
        # Step 9: Apply optional cropland mask filtering
        if cropland_mask_path:
            logger.info("Applying cropland mask filter")
            merged_gdf = filter_by_cropland_mask(
                merged_gdf,
                cropland_mask_path,
                overlap_fraction=0.5,
            )
            logger.info(f"After cropland filtering: {len(merged_gdf)} polygons")
        
        # Step 10: Add OSM attributes if provided
        if osm_waterways_path or osm_roads_path:
            logger.info("Adding OSM attributes")
            merged_gdf = add_osm_attributes(
                merged_gdf,
                osm_waterways_path=osm_waterways_path,
                osm_roads_path=osm_roads_path,
            )
        
        gdf = merged_gdf
        
    except Exception as e:
        logger.error(f"Error combining/processing chip predictions: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return {"features": [], "crs": crs, "bounds": bounds, "source_file": geotiff_path}
    
    # Convert GeoDataFrame to GeoJSON format
    logger.info("Converting to GeoJSON format")
    try:
        
        # Convert GeoDataFrame to GeoJSON features
        from shapely.geometry import mapping
        features = []
        if not gdf.empty:
            for idx, row in gdf.iterrows():
                # Get all properties except geometry
                properties = {k: v for k, v in row.items() if k != "geometry"}
                properties["source_file"] = geotiff_name
                
                feature = {
                    "type": "Feature",
                    "geometry": mapping(row.geometry),
                    "properties": properties
                }
                features.append(feature)
        
        # Save visualization if requested
        if save_visualization and output_dir:
            try:
                # Prepare RGB image for visualization
                if image.shape[2] == 4:  # BGRN
                    rgb_image = np.stack([
                        image[..., 2],  # R
                        image[..., 1],  # G
                        image[..., 0]   # B
                    ], axis=2)
                    
                    # Normalize to 0-255 for visualization
                    rgb_image_norm = np.zeros_like(rgb_image, dtype=np.float32)
                    for i in range(3):
                        band = rgb_image[..., i]
                        if band.max() > band.min():
                            rgb_image_norm[..., i] = 255 * (band - band.min()) / (band.max() - band.min())
                    rgb_image = rgb_image_norm.astype(np.uint8)
                else:
                    rgb_image = image
                
                # Create visualization (only if file_output_dir was created)
                if file_output_dir is None:
                    # Create output directory only when needed
                    file_output_dir = Path(output_dir) / geotiff_name
                    file_output_dir.mkdir(parents=True, exist_ok=True)
                vis_path = file_output_dir / f"{geotiff_name}_prediction.png"
                
                # Resize if too large
                height, width = rgb_image.shape[:2]
                max_size = 2000
                scale = min(max_size / height, max_size / width)
                
                if scale < 1:
                    new_height = int(height * scale)
                    new_width = int(width * scale)
                    small_img = cv2.resize(rgb_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
                    small_seg = cv2.resize(
                        full_panoptic_seg.cpu().numpy().astype(np.int32),
                        (new_width, new_height),
                        interpolation=cv2.INTER_NEAREST
                    )
                    full_panoptic_seg_small = torch.as_tensor(small_seg, dtype=torch.int64)
                else:
                    small_img = rgb_image
                    full_panoptic_seg_small = full_panoptic_seg
                
                # Create simple visualization
                import matplotlib.pyplot as plt
                plt.figure(figsize=(20, 20))
                
                # Overlay segmentation on image (simplified - use gdf for visualization)
                # Note: For full visualization, would need to rasterize gdf back to image
                # For now, create a simple colored overlay
                colored_seg = np.zeros((*small_img.shape[:2], 3), dtype=np.uint8)
                # Simple visualization - could be enhanced by rasterizing gdf
                
                # Blend with original image
                alpha = 0.7
                blended = cv2.addWeighted(small_img, 1-alpha, colored_seg, alpha, 0)
                
                plt.imshow(blended)
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(vis_path, bbox_inches='tight', dpi=150)
                plt.close()
                
                logger.info(f"Saved visualization to {vis_path}")
                
            except Exception as e:
                logger.error(f"Error saving visualization: {e}")
        
        return {
            "features": features,
            "crs": crs,
            "bounds": bounds,
            "source_file": str(Path(geotiff_path).name),
            "transform": transform,
            "image_shape": image.shape,
            "num_segments": len(gdf) if not gdf.empty else 0
        }
        
    except Exception as e:
        logger.error(f"Error converting to GeoJSON: {e}")
        return {"features": [], "crs": crs, "bounds": bounds, "source_file": str(Path(geotiff_path).name)}

def main(args):
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Setup config
    cfg = get_cfg()
    cfg.set_new_allowed(True)
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(["MODEL.WEIGHTS", args.weights])
    
    # Always register the dataset with proper metadata
    from detectron2.data.datasets import register_coco_panoptic
    from detectron2.data import MetadataCatalog
    
    # Get custom metadata
    custom_metadata = get_metadata()
    
    # Register both training and test datasets with proper metadata
    # The model internally references the training dataset metadata
    dataset_names = [cfg.DATASETS.TRAIN[0], cfg.DATASETS.TEST[0]]
    
    for dataset_name in dataset_names:
        try:
            register_coco_panoptic(
                name=dataset_name,
                metadata=custom_metadata,
                image_root=Path(args.input),  # Use input directory as image root
                panoptic_root=Path(args.input),  # Dummy path since we're not using COCO format
                panoptic_json="",  # Empty since we're not using COCO format
                instances_json=""   # Empty since we're not using COCO format
            )
            logger.info(f"Registered dataset {dataset_name} with metadata")
        except Exception as e:
            logger.warning(f"Could not register dataset {dataset_name}: {e}")
    
    # Handle dataset registration if needed
    if args.coco_root:
        for dataset_name in dataset_names:
            try:
                register_coco_panoptic(
                    name=dataset_name,
                    metadata=custom_metadata,
                    image_root=Path(args.coco_root),
                    panoptic_root=Path(args.coco_root)/'panoptic_test/',
                    panoptic_json=Path(args.coco_root)/'annotations/panoptic_test.json',
                    instances_json=Path(args.coco_root)/'annotations/instances_test.json'
                )
            except Exception as e:
                logger.warning(f"Could not register COCO dataset {dataset_name}: {e}")
    
    # Override Mask2Former internal thresholds if provided via CLI (BEFORE freezing)
    if args.object_mask_threshold is not None:
        cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD = args.object_mask_threshold
        logger.info(f"Using custom OBJECT_MASK_THRESHOLD: {args.object_mask_threshold}")
        # Also update confidence_threshold for postprocessing if not explicitly set
        if args.confidence_threshold is None:
            args.confidence_threshold = args.object_mask_threshold
    
    if args.overlap_threshold is not None:
        cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD = args.overlap_threshold
        logger.info(f"Using custom OVERLAP_THRESHOLD: {args.overlap_threshold}")
    
    # Get confidence threshold from config if not provided (for postprocessing)
    if args.confidence_threshold is None:
        # Use OBJECT_MASK_THRESHOLD from config (default Mask2Former: 0.8)
        args.confidence_threshold = cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD
        logger.info(f"Using confidence threshold from config: {args.confidence_threshold}")
    else:
        logger.info(f"Using user-specified confidence threshold: {args.confidence_threshold}")
        # If confidence_threshold was explicitly set, also update model threshold for consistency
        if args.object_mask_threshold is None:
            cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD = args.confidence_threshold
    
    cfg.freeze()
    
    # Get metadata and ensure it's properly set up
    try:
        metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
        logger.info(f"Using dataset: {cfg.DATASETS.TEST[0]}")
    except KeyError:
        # If dataset registration failed, create metadata manually
        logger.warning(f"Dataset {cfg.DATASETS.TEST[0]} not found, creating metadata manually")
        metadata = type('Metadata', (), {})()
        
        # Set up metadata manually using the same structure as get_metadata()
        for key, value in custom_metadata.items():
            setattr(metadata, key, value)
        
        # Register the metadata with the dataset name
        MetadataCatalog.set(cfg.DATASETS.TEST[0], metadata)
        logger.info("Manually registered metadata")
    
    # Also verify training dataset metadata (which the model internally uses)
    try:
        train_metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        logger.info(f"Training dataset metadata verified: {cfg.DATASETS.TRAIN[0]}")
    except KeyError:
        logger.warning(f"Training dataset {cfg.DATASETS.TRAIN[0]} not found, creating metadata manually")
        train_metadata = type('Metadata', (), {})()
        for key, value in custom_metadata.items():
            setattr(train_metadata, key, value)
        MetadataCatalog.set(cfg.DATASETS.TRAIN[0], train_metadata)
        logger.info("Manually registered training dataset metadata")
    
    # Verify metadata has required fields
    required_fields = [
        "thing_classes", "stuff_classes", "thing_colors", "stuff_colors",
        "thing_dataset_id_to_contiguous_id", "stuff_dataset_id_to_contiguous_id"
    ]
    
    missing_fields = [field for field in required_fields if not hasattr(metadata, field)]
    if missing_fields:
        logger.warning(f"Metadata missing fields: {missing_fields}")
        # Set defaults for missing fields
        for field in missing_fields:
            if field in custom_metadata:
                setattr(metadata, field, custom_metadata[field])
                logger.info(f"Set {field} to {custom_metadata[field]}")
    
    logger.info(f"Final metadata - thing_classes: {getattr(metadata, 'thing_classes', 'NOT SET')}")
    logger.info(f"Final metadata - thing_dataset_id_to_contiguous_id: {getattr(metadata, 'thing_dataset_id_to_contiguous_id', 'NOT SET')}")
    
    # Setup predictor
    predictor = SatellitePredictor(cfg)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all GeoTIFF files
    input_path = Path(args.input)
    if input_path.is_file():
        geotiff_files = [input_path]
    else:
        geotiff_files = list(input_path.glob("*.tif")) + list(input_path.glob("*.tiff"))
    
    logger.info(f"Found {len(geotiff_files)} GeoTIFF files to process")
    
    # Process each GeoTIFF
    all_features = []
    all_metadata = []
    image_bounds_features = []  # Track image bounds for density computation
    from shapely.geometry import box, mapping  # Import here for bounds tracking
    
    for geotiff_file in geotiff_files:
        try:
            result = process_single_geotiff(
                str(geotiff_file),
                predictor,
                args.chip_size,
                args.overlap,
                args.confidence_threshold,
                args.min_area,
                args.max_area,
                args.iou_threshold,
                args.edge_buffer,
                args.min_area_pixels,
                args.save_visualization,
                args.output,
                args.cropland_mask,
                args.osm_waterways,
                args.osm_roads,
            )
            
            all_features.extend(result["features"])
            all_metadata.append({
                "file": geotiff_file.name,
                "crs": str(result["crs"]),
                "num_segments": result.get("num_segments", 0),
                "num_features": len(result["features"])
            })
            
            # Track image bounds for density computation
            if "bounds" in result and result["bounds"]:
                bounds = result["bounds"]  # (minx, miny, maxx, maxy)
                bounds_polygon = box(bounds[0], bounds[1], bounds[2], bounds[3])
                # Get CRS for bounds polygon
                bounds_crs = result.get("crs")
                image_bounds_features.append({
                    "type": "Feature",
                    "geometry": mapping(bounds_polygon),
                    "properties": {
                        "source_file": result.get("source_file", geotiff_file.name),
                        "filename": geotiff_file.name,
                        "num_predictions": len(result["features"]),
                        "crs": str(bounds_crs) if bounds_crs else "unknown"
                    }
                })
            
        except Exception as e:
            logger.error(f"Error processing {geotiff_file}: {e}")
            continue
    
    # Create combined GeoJSON
    if all_features:
        geojson = {
            "type": "FeatureCollection",
            "crs": {
                "type": "name",
                "properties": {
                    "name": f"urn:ogc:def:crs:EPSG::{all_metadata[0]['crs'].split(':')[-1] if ':' in all_metadata[0]['crs'] else '4326'}"
                }
            },
            "features": all_features
        }
        
        # Save combined GeoJSON
        combined_geojson_path = output_dir / "combined_predictions.geojson"
        with open(combined_geojson_path, 'w') as f:
            json.dump(geojson, f, indent=2)
        
        logger.info(f"Saved combined GeoJSON with {len(all_features)} features to {combined_geojson_path}")
        
        # Save metadata summary
        metadata_path = output_dir / "processing_summary.json"
        summary = {
            "total_files_processed": len(all_metadata),
            "total_features": len(all_features),
            "files": all_metadata
        }
        with open(metadata_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Saved processing summary to {metadata_path}")
        
        # Save image bounds GeoJSON for density computation
        if image_bounds_features:
            bounds_geojson = {
                "type": "FeatureCollection",
                "crs": {
                    "type": "name",
                    "properties": {
                        "name": f"urn:ogc:def:crs:EPSG::{all_metadata[0]['crs'].split(':')[-1] if ':' in all_metadata[0]['crs'] else '4326'}"
                    }
                },
                "features": image_bounds_features
            }
            bounds_path = output_dir / "image_bounds.geojson"
            with open(bounds_path, 'w') as f:
                json.dump(bounds_geojson, f, indent=2)
            logger.info(f"Saved image bounds GeoJSON with {len(image_bounds_features)} features to {bounds_path}")
        
    else:
        logger.warning("No features were extracted from any files")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process GeoTIFFs and convert to GeoJSON")
    parser.add_argument("--config-file", required=True, help="Path to config file")
    parser.add_argument("--input", required=True, help="Path to GeoTIFF file or directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--weights", required=True, help="Path to model weights")
    parser.add_argument("--chip-size", type=int, default=512, help="Size of image chips")
    parser.add_argument("--overlap", type=int, default=0, help="Overlap between adjacent chips")
    parser.add_argument("--confidence-threshold", type=float, default=None, 
                       help="Minimum confidence score (default: from config MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD)")
    parser.add_argument("--min-area", type=float, default=100.0, 
                       help="Minimum area threshold in square meters (default: 100 m² = 0.01ha)")
    parser.add_argument("--max-area", type=float, default=90000.0,
                       help="Maximum area threshold in square meters (default: 90,000 m² = 9ha)")
    parser.add_argument("--iou-threshold", type=float, default=0.7, help="IoU threshold for merging overlapping predictions")
    parser.add_argument("--edge-buffer", type=float, default=5.0, help="Edge buffer in meters for filtering edge polygons (default: 5.0m, reduce for small chips)")
    parser.add_argument("--min-area-pixels", type=int, default=10, 
                       help="Minimum area in pixels for pixel-space filtering (default: 10, just removes tiny fragments)")
    parser.add_argument("--object-mask-threshold", type=float, default=None,
                       help="Mask2Former internal OBJECT_MASK_THRESHOLD (default: from config, typically 0.8). Lower values = more predictions.")
    parser.add_argument("--overlap-threshold", type=float, default=None,
                       help="Mask2Former internal OVERLAP_THRESHOLD (default: from config, typically 0.8). Lower values = more overlapping masks kept.")
    parser.add_argument("--save-visualization", action="store_true", help="Save prediction visualizations")
    parser.add_argument("--cropland-mask", help="Path to cropland mask GeoTIFF (optional)")
    parser.add_argument("--osm-waterways", help="Path to OSM waterways GeoJSON/Shapefile (optional)")
    parser.add_argument("--osm-roads", help="Path to OSM roads GeoJSON/Shapefile (optional)")
    parser.add_argument("--coco-root", help="Path to COCO format dataset root (if needed)")
    parser.add_argument("--opts", default=[], nargs=argparse.REMAINDER)
    
    args = parser.parse_args()
    main(args) 