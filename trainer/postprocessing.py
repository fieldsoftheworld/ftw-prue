"""
Postprocessing utilities for agricultural field boundary delineation.

This module provides conversion functions to bridge pixel-space and geographic postprocessing:
- mask_to_polygon(): Convert binary masks to shapely polygons (pixel coordinates)
- panoptic_to_geojson(): Convert panoptic segmentation to GeoDataFrame (geographic coordinates)

For pixel-space filtering, see trainer/postprocessing_pixel.py
For geographic filtering, see trainer/postprocessing_geo.py

These are utility functions used by scripts and trainer modules. For command-line usage,
see scripts/postprocess_saved_predictions.py.
"""

import os
import json
import glob
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import logging

import numpy as np
import cv2
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.transform import xy
from shapely.geometry import Polygon, box
from tqdm import tqdm
import math

# Import pixel-space and geographic postprocessing modules
from trainer.postprocessing_pixel import (
    filter_segments_by_confidence,
    filter_segments_by_category,
    filter_segments_by_isthing
)
from trainer.postprocessing_geo import (
    filter_edge_polygons_geo,
    merge_overlapping_fields,
    resolve_overlaps,
    filter_by_cropland_mask,
    add_osm_attributes
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def mask_to_polygon(mask: np.ndarray, min_area: float = 10.0) -> Optional[Polygon]:
    """
    Convert a binary mask to a polygon.
    
    Args:
        mask: Binary mask array
        min_area: Minimum area threshold for valid polygons
        
    Returns:
        Polygon object or None if mask is too small
    """
    # Find contours in the mask
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    if not contours:
        return None
    
    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    
    if area < min_area:
        return None
    
    # Convert contour to polygon
    contour_points = largest_contour.reshape(-1, 2)
    if len(contour_points) < 3:
        return None
    
    # Ensure polygon is closed
    if not np.array_equal(contour_points[0], contour_points[-1]):
        contour_points = np.vstack([contour_points, contour_points[0]])
    
    try:
        polygon = Polygon(contour_points)
        if polygon.is_valid and polygon.area > min_area:
            return polygon
    except Exception as e:
        logger.warning(f"Failed to create polygon from contour: {e}")
    
    return None


def panoptic_to_geojson(
    panoptic_seg: np.ndarray, 
    segments_info: List[Dict], 
    transform: rasterio.Affine,
    crs: CRS,
    min_area: float = 100.0,  # 100 m² = 0.01ha (from methods section)
    max_area: Optional[float] = 90000.0,  # 90,000 m² = 9ha (from methods section)
    confidence_threshold: Optional[float] = None,
    apply_pixel_filters: bool = True
) -> gpd.GeoDataFrame:
    """
    Convert panoptic segmentation to GeoJSON format (GeoDataFrame).
    
    This function bridges pixel-space and geographic postprocessing:
    - Takes panoptic segmentation (pixel-space)
    - Converts masks to polygons (pixel coordinates)
    - Applies georeferencing (pixel → geographic coordinates)
    - Returns GeoDataFrame ready for geographic postprocessing
    
    Args:
        panoptic_seg: Panoptic segmentation array (H, W)
        segments_info: List of segment information dictionaries (can be pre-filtered)
        transform: Rasterio transform for georeferencing
        crs: Coordinate reference system
        min_area: Minimum area threshold for valid fields (in square meters, default: 100 m² = 0.01ha)
        max_area: Maximum area threshold for valid fields (in square meters, default: 90,000 m² = 9ha)
        confidence_threshold: Minimum confidence score (if None, uses segments_info as-is)
        apply_pixel_filters: If True, apply basic pixel-space filters (confidence, category, isthing)
                            If False, assume segments_info is already filtered
        
    Returns:
        GeoDataFrame with field polygons and metadata
    """
    # Apply basic pixel-space filters if requested
    if apply_pixel_filters:
        if confidence_threshold is not None:
            segments_info = filter_segments_by_confidence(segments_info, confidence_threshold)
        # category_id=0 is ag_field (contiguous ID), category_id=1 is background
        segments_info = filter_segments_by_category(segments_info, category_id=0, exclude_background=True)
        segments_info = filter_segments_by_isthing(segments_info, keep_things=True)
    
    features = []
    
    for segment in segments_info:
        # Extract mask for this segment
        segment_id = segment["id"]
        mask = (panoptic_seg == segment_id).astype(np.uint8)
        
        # Skip if mask is empty
        if np.sum(mask) == 0:
            continue
        
        # Convert mask to polygon (pixel coordinates)
        # Calculate pixel area threshold - need to handle geographic vs projected CRS
        is_geographic = crs is not None and (
            str(crs).startswith('EPSG:4326') or 
            (hasattr(crs, 'is_geographic') and crs.is_geographic)
        )
        
        if is_geographic:
            # For geographic CRS, transform values are in degrees
            # Convert min_area (m²) to approximate pixel area threshold
            # Approximate: 1 degree ≈ 111,320 m at equator
            pixel_width_deg = abs(transform[0])
            pixel_height_deg = abs(transform[4])
            # Get approximate latitude from transform origin (y-origin for EPSG:4326)
            # Use transform[5] which is the y-origin (latitude for EPSG:4326)
            approx_lat = abs(transform[5]) if transform[5] != 0 else 0.0
            meters_per_degree_lat = 111320.0
            meters_per_degree_lon = 111320.0 * math.cos(math.radians(approx_lat))
            # Average for conservative estimate (use smaller value)
            avg_meters_per_degree = (meters_per_degree_lat + meters_per_degree_lon) / 2.0
            pixel_area_m2 = (pixel_width_deg * avg_meters_per_degree) * (pixel_height_deg * avg_meters_per_degree)
            pixel_area_threshold = min_area / pixel_area_m2 if pixel_area_m2 > 0 else 0
        else:
            # For projected CRS, transform values are in meters
            pixel_area_threshold = min_area / (abs(transform[0]) * abs(transform[4])) if min_area > 0 else 0
        
        polygon = mask_to_polygon(mask, min_area=pixel_area_threshold)
        if polygon is None:
            continue
        
        # Transform polygon coordinates to geographic space
        coords = np.array(polygon.exterior.coords)
        if len(coords) < 3:
            continue
            
        # Convert pixel coordinates to geographic coordinates
        x_coords, y_coords = xy(
            transform, 
            coords[:, 1],  # rows
            coords[:, 0]   # cols
        )
        
        # Create new polygon with geographic coordinates
        geo_polygon = Polygon(list(zip(x_coords, y_coords)))
        
        # Validate and check area (in square meters)
        # Apply geographic area filtering (bulk of area filtering happens here)
        polygon_area = geo_polygon.area
        
        if not geo_polygon.is_valid:
            continue
        
        # For geographic CRS, area is in square degrees - need to convert to m²
        if is_geographic:
            # Approximate conversion: 1 degree² ≈ 12,364 km² at equator
            # More accurate: use latitude-dependent conversion
            center_lat = (geo_polygon.bounds[1] + geo_polygon.bounds[3]) / 2.0
            meters_per_degree_lat = 111320.0
            meters_per_degree_lon = 111320.0 * math.cos(math.radians(abs(center_lat)))
            # Convert square degrees to square meters (approximate)
            polygon_area_m2 = polygon_area * (meters_per_degree_lat * meters_per_degree_lon)
        else:
            # For projected CRS, area is already in m² (or CRS units)
            polygon_area_m2 = polygon_area
        
        # Check area thresholds (now in m²)
        if polygon_area_m2 >= min_area:
            # Check max_area if specified
            if max_area is not None and polygon_area_m2 > max_area:
                continue  # Skip polygons larger than max_area
            
            features.append({
                "geometry": geo_polygon,
                "properties": {
                    "segment_id": segment_id,
                    "category_id": segment.get("category_id", 1),
                    "confidence": segment.get("confidence", 0.0),
                    "score": segment.get("score", 0.0),
                    "mask_score": segment.get("mask_score", 0.0),
                    "isthing": segment.get("isthing", True),
                    "area": polygon_area_m2,  # Store area in m² for consistency
                    "touches_edge": segment.get("touches_edge", False)  # From pixel-space filtering
                }
            })
    
    if not features:
        return gpd.GeoDataFrame(columns=["geometry"], crs=crs)
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame.from_features(features, crs=crs)
    return gdf


def extract_tile_info(filename: str) -> Dict[str, Any]:
    """
    Extract geographic information from tile filename.
    
    Args:
        filename: Tile filename (e.g., "tile_123456_789012_512_64_32632.tif")
        
    Returns:
        Dictionary with tile information
    """
    # Remove extension and split
    basename = Path(filename).stem
    parts = basename.split("_")
    
    # Try to extract coordinates and parameters
    # This pattern may need adjustment based on your naming convention
    if len(parts) >= 5:
        try:
            minx = int(parts[-5])
            miny = int(parts[-4])
            width = int(parts[-3])
            buffer_size = int(parts[-2])
            epsg = int(parts[-1])
            
            return {
                "minx": minx,
                "miny": miny,
                "width": width,
                "buffer_size": buffer_size,
                "epsg": epsg,
                "filename": filename
            }
        except ValueError:
            logger.warning(f"Could not parse tile info from {filename}")
    
    return {"filename": filename}


def create_tile_boundary_box(tile_info: Dict[str, Any], buffer_margin: int = 0) -> box:
    """
    Create a bounding box for a tile with optional buffer margin.
    
    Args:
        tile_info: Tile information dictionary
        buffer_margin: Buffer margin in meters to avoid edge effects
        
    Returns:
        Shapely box representing tile boundary
    """
    minx = tile_info["minx"] - tile_info["buffer_size"] + buffer_margin
    miny = tile_info["miny"] - tile_info["buffer_size"] + buffer_margin
    maxx = tile_info["minx"] + tile_info["width"] + tile_info["buffer_size"] - buffer_margin
    maxy = tile_info["miny"] + tile_info["width"] + tile_info["buffer_size"] - buffer_margin
    
    return box(minx, miny, maxx, maxy)


# Geographic filtering functions moved to trainer/postprocessing_geo.py
# Import them here for backward compatibility
# filter_edge_polygons_geo and merge_overlapping_fields are imported at top of file


def stitch_tile_predictions(
    predictions_dir: str,
    tiles_dir: str,
    output_dir: str,
    edge_buffer: int = 50,
    iou_threshold: float = 0.7,
    confidence_threshold: float = 0.8,  # Default matches MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD
    min_area: float = 100.0,  # 100 m² = 0.01ha (from methods section)
    max_area: float = 90000.0  # 90,000 m² = 9ha (from methods section)
) -> gpd.GeoDataFrame:
    """
    Stitch together predictions from multiple tiles and merge overlapping fields.
    
    Args:
        predictions_dir: Directory containing prediction files
        tiles_dir: Directory containing original tile files
        output_dir: Output directory for results
        edge_buffer: Buffer distance from tile edges in meters
        iou_threshold: IoU threshold for merging overlapping fields
        confidence_threshold: Minimum confidence score (default: from config)
        min_area: Minimum area threshold for valid fields (square meters, default: 100 m²)
        max_area: Maximum area threshold for valid fields (square meters, default: 90,000 m²)
        
    Returns:
        Combined GeoDataFrame with all fields
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Find all prediction files
    pred_files = glob.glob(os.path.join(predictions_dir, "*.json"))
    logger.info(f"Found {len(pred_files)} prediction files")
    
    all_fields = []
    
    for pred_file in tqdm(pred_files, desc="Processing predictions"):
        try:
            # Extract tile info from filename
            tile_info = extract_tile_info(Path(pred_file).stem)
            
            # Find corresponding tile file
            tile_pattern = f"*{tile_info['minx']}_{tile_info['miny']}*.tif"
            tile_files = glob.glob(os.path.join(tiles_dir, tile_pattern))
            
            if not tile_files:
                logger.warning(f"No tile file found for {pred_file}")
                continue
                
            tile_file = tile_files[0]
            
            # Read tile metadata
            with rasterio.open(tile_file) as src:
                transform = src.transform
                crs = src.crs
            
            # Load predictions
            with open(pred_file, 'r') as f:
                predictions = json.load(f)
            
            # Convert predictions to GeoJSON
            if "panoptic_seg" in predictions:
                panoptic_seg = np.array(predictions["panoptic_seg"][0])
                segments_info = predictions["panoptic_seg"][1]
                
                tile_fields = panoptic_to_geojson(
                    panoptic_seg, segments_info, transform, crs,
                    min_area=min_area,
                    max_area=max_area,
                    confidence_threshold=confidence_threshold
                )
                
                if not tile_fields.empty:
                    # Create tile boundary
                    tile_boundary = create_tile_boundary_box(tile_info, edge_buffer)
                    
                    # Filter edge fields (using geographic function)
                    filtered_fields = filter_edge_polygons_geo(tile_fields, tile_boundary, edge_buffer)
                    
                    if not filtered_fields.empty:
                        all_fields.append(filtered_fields)
                        
        except Exception as e:
            logger.error(f"Error processing {pred_file}: {e}")
            continue
    
    if not all_fields:
        logger.warning("No valid fields found")
        return gpd.GeoDataFrame()
    
    # Combine all fields
    combined_fields = pd.concat(all_fields, ignore_index=True)
    
    # Merge overlapping fields
    logger.info("Merging overlapping fields...")
    merged_fields = merge_overlapping_fields(combined_fields, iou_threshold)
    
    # Save results
    output_file = os.path.join(output_dir, "merged_fields.geojson")
    merged_fields.to_file(output_file, driver="GeoJSON")
    logger.info(f"Saved {len(merged_fields)} merged fields to {output_file}")
    
    return merged_fields


def process_single_prediction(
    prediction_file: str,
    tile_file: str,
    output_file: str,
    confidence_threshold: float = 0.8,  # Default matches MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD
    min_area: float = 100.0,  # 100 m² = 0.01ha (from methods section)
    max_area: float = 90000.0  # 90,000 m² = 9ha (from methods section)
) -> gpd.GeoDataFrame:
    """
    Process a single prediction file and convert to GeoJSON.
    
    Args:
        prediction_file: Path to prediction JSON file
        tile_file: Path to corresponding tile file
        output_file: Output GeoJSON file path
        confidence_threshold: Minimum confidence score (default: from config)
        min_area: Minimum area threshold (square meters, default: 100 m²)
        max_area: Maximum area threshold (square meters, default: 90,000 m²)
        
    Returns:
        GeoDataFrame with field polygons
    """
    # Read tile metadata
    with rasterio.open(tile_file) as src:
        transform = src.transform
        crs = src.crs
    
    # Load predictions
    with open(prediction_file, 'r') as f:
        predictions = json.load(f)
    
    # Convert to GeoJSON
    if "panoptic_seg" in predictions:
        panoptic_seg = np.array(predictions["panoptic_seg"][0])
        segments_info = predictions["panoptic_seg"][1]
        
        fields_gdf = panoptic_to_geojson(
            panoptic_seg, segments_info, transform, crs,
            min_area=min_area,
            max_area=max_area,
            confidence_threshold=confidence_threshold
        )
        
        # Save to file
        if not fields_gdf.empty:
            fields_gdf.to_file(output_file, driver="GeoJSON")
            logger.info(f"Saved {len(fields_gdf)} fields to {output_file}")
        else:
            logger.warning("No valid fields found")
        
        return fields_gdf
    
    logger.error("No panoptic segmentation found in predictions")
    return gpd.GeoDataFrame()


# CLI moved to scripts/postprocess_saved_predictions.py
# This module contains utility functions only 