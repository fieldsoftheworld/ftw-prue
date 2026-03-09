"""
Geographic postprocessing utilities for agricultural field boundary delineation.

This module provides functions to process vector polygons in geographic space.
These functions operate on GeoDataFrames after conversion from pixel-space masks.

See trainer/postprocessing_pixel.py for pixel-space filtering and
trainer/postprocessing.py for conversion functions.
"""

import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.crs import CRS
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
from typing import Dict, List, Optional, Tuple
import logging
import os
import math

logger = logging.getLogger(__name__)


def filter_by_cropland_mask(
    polygons_gdf: gpd.GeoDataFrame,
    cropland_mask_path: str,
    overlap_fraction: float = 0.5,
    check_bounds: bool = True
) -> gpd.GeoDataFrame:
    """
    Filter polygons by cropland extent mask.
    
    Only keeps polygons that overlap with cropland areas in the mask.
    If polygons are outside the mask's bounding box, they are kept (mask covers subregion only).
    
    Args:
        polygons_gdf: GeoDataFrame with field polygons
        cropland_mask_path: Path to cropland mask GeoTIFF (values: 1=cropland, nodata=elsewhere)
        overlap_fraction: Minimum fraction of polygon that must overlap with cropland
        check_bounds: If True, check if polygons intersect mask bounds before filtering
        
    Returns:
        Filtered GeoDataFrame
    """
    if polygons_gdf.empty:
        return polygons_gdf
    
    try:
        # Open cropland mask
        with rasterio.open(cropland_mask_path) as mask_src:
            mask_crs = mask_src.crs
            mask_bounds = box(*mask_src.bounds)
            
            # Reproject polygons to mask CRS if needed
            if polygons_gdf.crs != mask_crs:
                polygons_gdf = polygons_gdf.to_crs(mask_crs)
            
            # Check which polygons intersect mask bounds
            intersects_mask = polygons_gdf.geometry.intersects(mask_bounds)
            
            # For polygons outside mask bounds, keep them (mask doesn't cover that area)
            polygons_outside = polygons_gdf[~intersects_mask].copy()
            
            # For polygons inside mask bounds, check cropland overlap
            polygons_inside = polygons_gdf[intersects_mask].copy()
            
            if polygons_inside.empty:
                return polygons_outside
            
            # Sample mask values for each polygon
            kept_polygons = []
            
            for idx, row in polygons_inside.iterrows():
                try:
                    # Get polygon bounds
                    geom = row.geometry
                    bounds = geom.bounds
                    
                    # Read mask window
                    window = rasterio.windows.from_bounds(*bounds, mask_src.transform)
                    mask_data = mask_src.read(1, window=window)
                    
                    if mask_data.size == 0:
                        # No mask data, keep polygon (outside mask coverage)
                        kept_polygons.append(row)
                        continue
                    
                    # Get transform for this window
                    window_transform = rasterio.windows.transform(window, mask_src.transform)
                    
                    # Rasterize polygon to same window
                    from rasterio.features import rasterize
                    polygon_raster = rasterize(
                        [geom],
                        out_shape=mask_data.shape,
                        transform=window_transform,
                        fill=0,
                        default_value=1,
                        dtype=np.uint8
                    )
                    
                    # Calculate overlap
                    overlap_mask = (polygon_raster > 0) & (mask_data == 1)
                    polygon_pixels = np.sum(polygon_raster > 0)
                    overlap_pixels = np.sum(overlap_mask)
                    
                    if polygon_pixels > 0:
                        overlap_frac = overlap_pixels / polygon_pixels
                        
                        if overlap_frac >= overlap_fraction:
                            kept_polygons.append(row)
                    
                except Exception as e:
                    logger.warning(f"Error processing polygon {idx} with cropland mask: {e}")
                    # On error, keep polygon (conservative approach)
                    kept_polygons.append(row)
            
            # Combine polygons inside and outside mask bounds
            if kept_polygons:
                kept_inside_gdf = gpd.GeoDataFrame(kept_polygons, crs=mask_crs)
                if not polygons_outside.empty:
                    result = pd.concat([kept_inside_gdf, polygons_outside], ignore_index=True)
                else:
                    result = kept_inside_gdf
            else:
                result = polygons_outside
            
            # Reproject back to original CRS if needed
            if polygons_gdf.crs != mask_crs:
                result = result.to_crs(polygons_gdf.crs)
            
            return result
            
    except Exception as e:
        logger.error(f"Error reading cropland mask {cropland_mask_path}: {e}")
        # On error, return original polygons (conservative approach)
        return polygons_gdf


def resolve_overlaps(
    polygons_gdf: gpd.GeoDataFrame,
    overlap_threshold: float = 0.3,
    keep_higher_confidence: bool = True
) -> gpd.GeoDataFrame:
    """
    Resolve overlapping polygons using IoU-based rules (following Rufin et al. approach).
    
    For overlapping polygons:
    - If IoU > overlap_threshold: keep polygon with higher confidence, remove other
    - If IoU <= overlap_threshold: keep both (they are distinct fields)
    
    Args:
        polygons_gdf: GeoDataFrame with field polygons
        overlap_threshold: IoU threshold for considering polygons as duplicates
        keep_higher_confidence: If True, keep polygon with higher confidence when overlapping
        
    Returns:
        GeoDataFrame with resolved overlaps
    """
    if polygons_gdf.empty or len(polygons_gdf) == 1:
        return polygons_gdf
    
    # Create spatial index for efficient intersection queries
    spatial_index = polygons_gdf.sindex
    
    # Track which polygons have been processed/removed
    to_remove = set()
    
    for idx, row in polygons_gdf.iterrows():
        if idx in to_remove:
            continue
        
        # Find overlapping polygons
        bounds = row.geometry.bounds
        possible_matches = list(spatial_index.intersection(bounds))
        
        current_confidence = row.get("confidence", 0.0)
        if isinstance(current_confidence, dict):
            current_confidence = current_confidence.get("confidence", 0.0)
        
        for match_idx in possible_matches:
            if match_idx == idx or match_idx in to_remove:
                continue
            
            match_row = polygons_gdf.iloc[match_idx]
            
            # Calculate IoU
            intersection = row.geometry.intersection(match_row.geometry)
            union = row.geometry.union(match_row.geometry)
            
            if union.area > 0:
                iou = intersection.area / union.area
                
                if iou > overlap_threshold:
                    # High overlap - keep one based on confidence
                    match_confidence = match_row.get("confidence", 0.0)
                    if isinstance(match_confidence, dict):
                        match_confidence = match_confidence.get("confidence", 0.0)
                    
                    if keep_higher_confidence:
                        if match_confidence > current_confidence:
                            # Remove current polygon, keep match
                            to_remove.add(idx)
                            break  # Current polygon removed, move to next
                        else:
                            # Remove match polygon, keep current
                            to_remove.add(match_idx)
                    else:
                        # Keep first one encountered
                        to_remove.add(match_idx)
    
    # Remove overlapping polygons
    if to_remove:
        result = polygons_gdf.drop(index=list(to_remove)).copy()
        return result
    
    return polygons_gdf


def add_osm_attributes(
    polygons_gdf: gpd.GeoDataFrame,
    osm_waterways_path: Optional[str] = None,
    osm_roads_path: Optional[str] = None,
    buffer_distance: float = 50.0
) -> gpd.GeoDataFrame:
    """
    Add OSM infrastructure intersection attributes to polygons.
    
    Adds columns:
    - `intersects_waterway`: Boolean indicating if polygon intersects waterways
    - `intersects_road`: Boolean indicating if polygon intersects roads
    - `distance_to_waterway`: Minimum distance to nearest waterway (meters)
    - `distance_to_road`: Minimum distance to nearest road (meters)
    
    Args:
        polygons_gdf: GeoDataFrame with field polygons
        osm_waterways_path: Path to OSM waterways GeoJSON/Shapefile (optional)
        osm_roads_path: Path to OSM roads GeoJSON/Shapefile (optional)
        buffer_distance: Buffer distance for intersection check (meters)
        
    Returns:
        GeoDataFrame with added OSM attributes
    """
    result_gdf = polygons_gdf.copy()
    
    # Initialize OSM attribute columns
    result_gdf["intersects_waterway"] = False
    result_gdf["intersects_road"] = False
    result_gdf["distance_to_waterway"] = np.inf
    result_gdf["distance_to_road"] = np.inf
    
    # Process waterways
    if osm_waterways_path:
        try:
            waterways_gdf = gpd.read_file(osm_waterways_path)
            
            # Reproject to match polygons if needed
            if waterways_gdf.crs != result_gdf.crs:
                waterways_gdf = waterways_gdf.to_crs(result_gdf.crs)
            
            # Create buffered polygons for intersection check
            buffered_polygons = result_gdf.geometry.buffer(buffer_distance)
            
            # Check intersections
            for idx, polygon in enumerate(buffered_polygons):
                intersecting = waterways_gdf[waterways_gdf.geometry.intersects(polygon)]
                
                if not intersecting.empty:
                    result_gdf.loc[result_gdf.index[idx], "intersects_waterway"] = True
                    
                    # Calculate minimum distance
                    distances = result_gdf.geometry.iloc[idx].distance(intersecting.geometry)
                    result_gdf.loc[result_gdf.index[idx], "distance_to_waterway"] = distances.min()
            
        except Exception as e:
            logger.warning(f"Error processing OSM waterways {osm_waterways_path}: {e}")
    
    # Process roads
    if osm_roads_path:
        try:
            roads_gdf = gpd.read_file(osm_roads_path)
            
            # Reproject to match polygons if needed
            if roads_gdf.crs != result_gdf.crs:
                roads_gdf = roads_gdf.to_crs(result_gdf.crs)
            
            # Create buffered polygons for intersection check
            buffered_polygons = result_gdf.geometry.buffer(buffer_distance)
            
            # Check intersections
            for idx, polygon in enumerate(buffered_polygons):
                intersecting = roads_gdf[roads_gdf.geometry.intersects(polygon)]
                
                if not intersecting.empty:
                    result_gdf.loc[result_gdf.index[idx], "intersects_road"] = True
                    
                    # Calculate minimum distance
                    distances = result_gdf.geometry.iloc[idx].distance(intersecting.geometry)
                    result_gdf.loc[result_gdf.index[idx], "distance_to_road"] = distances.min()
            
        except Exception as e:
            logger.warning(f"Error processing OSM roads {osm_roads_path}: {e}")
    
    return result_gdf


def filter_edge_polygons_geo(
    polygons_gdf: gpd.GeoDataFrame,
    chip_bounds: box,
    edge_buffer: float = 50.0
) -> gpd.GeoDataFrame:
    """
    Filter out polygons that are too close to chip edges (geographic coordinates).
    
    Args:
        polygons_gdf: GeoDataFrame with field polygons
        chip_bounds: Shapely box representing chip boundaries
        edge_buffer: Buffer distance from edges in meters
        
    Returns:
        Filtered GeoDataFrame
    """
    if polygons_gdf.empty:
        return polygons_gdf
    
    # Check if CRS is geographic (EPSG:4326 or similar)
    # For geographic CRS, buffer() uses degrees, not meters
    # We need to convert meters to degrees or skip edge filtering
    crs = polygons_gdf.crs
    is_geographic = crs is not None and (
        str(crs).startswith('EPSG:4326') or 
        (hasattr(crs, 'is_geographic') and crs.is_geographic)
    )
    
    if is_geographic:
        # For geographic CRS, convert meters to approximate degrees
        # At equator: 1 degree ≈ 111,320 meters
        # This is approximate and varies with latitude, but should be fine for small buffers
        # Get approximate latitude from chip bounds center
        center_lat = (chip_bounds.bounds[1] + chip_bounds.bounds[3]) / 2.0
        # Meters per degree at this latitude (approximate)
        meters_per_degree_lat = 111320.0  # Constant for latitude
        meters_per_degree_lon = 111320.0 * math.cos(math.radians(center_lat))  # Varies with latitude
        
        # Use average for buffer (conservative - uses smaller value)
        avg_meters_per_degree = (meters_per_degree_lat + meters_per_degree_lon) / 2.0
        edge_buffer_degrees = edge_buffer / avg_meters_per_degree
        
        logger.debug(f"Geographic CRS detected. Converting {edge_buffer} m to ~{edge_buffer_degrees:.8f} degrees")
        
        # Create inner boundary (excluding edge buffer in degrees)
        inner_boundary = chip_bounds.buffer(-edge_buffer_degrees)
    else:
        # For projected CRS, buffer is already in meters (or CRS units)
        inner_boundary = chip_bounds.buffer(-edge_buffer)
    
    # Filter fields that are completely within inner boundary
    within_mask = polygons_gdf.geometry.within(inner_boundary)
    
    return polygons_gdf[within_mask].copy()


def merge_overlapping_fields(
    polygons_gdf: gpd.GeoDataFrame,
    iou_threshold: float = 0.7
) -> gpd.GeoDataFrame:
    """
    Merge overlapping fields based on IoU threshold.
    
    This is a more aggressive merging than resolve_overlaps - it actually combines
    geometries rather than just removing duplicates.
    
    Args:
        polygons_gdf: GeoDataFrame with field polygons
        iou_threshold: IoU threshold for merging
        
    Returns:
        Merged GeoDataFrame
    """
    if polygons_gdf.empty or len(polygons_gdf) == 1:
        return polygons_gdf
    
    # Create spatial index for efficient intersection queries
    spatial_index = polygons_gdf.sindex
    
    # Track which fields have been processed
    processed = set()
    merged_features = []
    
    for idx, row in polygons_gdf.iterrows():
        if idx in processed:
            continue
        
        # Find overlapping fields
        bounds = row.geometry.bounds
        possible_matches = list(spatial_index.intersection(bounds))
        
        overlapping_geometries = []
        overlapping_confidences = []
        
        for match_idx in possible_matches:
            if match_idx in processed:
                continue
            
            match_row = polygons_gdf.iloc[match_idx]
            
            # Calculate IoU
            intersection = row.geometry.intersection(match_row.geometry)
            union = row.geometry.union(match_row.geometry)
            
            if union.area > 0:
                iou = intersection.area / union.area
                
                if iou > iou_threshold:
                    overlapping_geometries.append(match_row.geometry)
                    
                    # Get confidence
                    match_conf = match_row.get("confidence", 0.0)
                    if isinstance(match_conf, dict):
                        match_conf = match_conf.get("confidence", 0.0)
                    overlapping_confidences.append(match_conf)
                    
                    processed.add(match_idx)
        
        if overlapping_geometries:
            # Merge overlapping fields
            merged_geometry = unary_union([row.geometry] + overlapping_geometries)
            
            # Calculate weighted average confidence
            current_conf = row.get("confidence", 0.0)
            if isinstance(current_conf, dict):
                current_conf = current_conf.get("confidence", 0.0)
            
            all_confidences = [current_conf] + overlapping_confidences
            avg_confidence = np.mean(all_confidences)
            
            # Create merged feature
            merged_feature = row.copy()
            merged_feature.geometry = merged_geometry
            merged_feature["confidence"] = avg_confidence
            merged_feature["area"] = merged_geometry.area
            merged_feature["merged_count"] = len(overlapping_geometries) + 1
            
            merged_features.append(merged_feature)
        else:
            # No overlaps, keep original field
            merged_features.append(row)
    
    if not merged_features:
        return polygons_gdf
    
    # Create new GeoDataFrame
    merged_gdf = gpd.GeoDataFrame(merged_features, crs=polygons_gdf.crs)
    return merged_gdf
