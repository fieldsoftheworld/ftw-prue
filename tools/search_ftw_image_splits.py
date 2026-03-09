#!/usr/bin/env python3
"""
Get FTW Image Filename by Split Index

This utility script helps you find the filename for a specific image in the FTW dataset
by its index within a split (train/val/test).

Example usage:
    # Get the 104th image in the Austria test set
    python search_ftw_image_splits.py --country austria --split test --index 104 --window a
    
    # Get the first 5 images in the France train set
    python search_ftw_image_splits.py --country france --split train --index 0 --count 5
    
    # Show all info about the 10th test image
    python search_ftw_image_splits.py --country austria --split test --index 10 --show_all
"""

import argparse
from pathlib import Path
import pandas as pd

try:
    import geopandas as gpd
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False


def get_ftw_image_info(
    data_root: str,
    country: str,
    split: str,
    index: int,
    window: str = "a",
    count: int = 1,
    show_all: bool = False
):
    """
    Get filename and info for a specific image in the FTW dataset.
    
    Args:
        data_root: Root directory of FTW dataset
        country: Country name (e.g., 'austria', 'france')
        split: Dataset split ('train', 'val', or 'test')
        index: Zero-based index of the image in the split
        window: Temporal window ('a' or 'b')
        count: Number of images to retrieve (starting from index)
        show_all: Show all available columns from the GeoParquet
    
    Returns:
        List of dictionaries containing image information
    """
    data_root = Path(data_root)
    
    # Load the chips GeoParquet file
    chips_file = data_root / country / f"chips_{country}.parquet"
    
    if not chips_file.exists():
        raise FileNotFoundError(f"Chips file not found: {chips_file}")
    
    print(f"Loading chips from: {chips_file}")
    
    # Try to read with geopandas first, fall back to pandas
    if HAS_GEOPANDAS:
        chips_gdf = gpd.read_parquet(chips_file)
    else:
        chips_gdf = pd.read_parquet(chips_file)
    
    # Filter by split
    split_chips = chips_gdf[chips_gdf["split"] == split].copy()
    
    if split_chips.empty:
        raise ValueError(f"No chips found for split '{split}' in {country}")
    
    # Sort by aoi_id for consistent ordering
    split_chips = split_chips.sort_values("aoi_id").reset_index(drop=True)
    
    print(f"\nFound {len(split_chips)} images in {country} {split} set")
    
    # Check if index is valid
    if index < 0 or index >= len(split_chips):
        raise ValueError(
            f"Index {index} out of range. Valid range: 0-{len(split_chips)-1}"
        )
    
    # Get the requested images
    end_index = min(index + count, len(split_chips))
    results = []
    
    for i in range(index, end_index):
        row = split_chips.iloc[i]
        
        # Construct the filename based on aoi_id and window
        aoi_id = row["aoi_id"]
        filename = f"{aoi_id}.tif"
        
        # Construct the full path
        image_path = data_root / country / "s2_images" / f"window_{window}" / filename
        mask_path = data_root / country / "label_masks" / filename
        
        result = {
            "index": i,
            "aoi_id": aoi_id,
            "filename": filename,
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "exists": image_path.exists(),
            "mask_exists": mask_path.exists(),
        }
        
        # Add all columns if requested
        if show_all:
            for col in split_chips.columns:
                if col not in result:
                    result[col] = row[col]
        
        results.append(result)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Get FTW image filename by split index",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Get the 104th image in the Austria test set
  %(prog)s --country austria --split test --index 104

  # Get the first 5 images in the France train set
  %(prog)s --country france --split train --index 0 --count 5 --window a
  
  # Show all info about the 10th test image
  %(prog)s --country austria --split test --index 10 --show_all
        """
    )
    
    parser.add_argument(
        "--data_root",
        type=str,
        default="./data/ftw",
        help="Root directory of FTW dataset"
    )
    parser.add_argument(
        "--country",
        type=str,
        required=True,
        help="Country name (e.g., austria, france, germany)"
    )
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        choices=["train", "val", "test"],
        help="Dataset split"
    )
    parser.add_argument(
        "--index",
        type=int,
        required=True,
        help="Zero-based index of the image in the split"
    )
    parser.add_argument(
        "--window",
        type=str,
        default="a",
        choices=["a", "b"],
        help="Temporal window (default: a)"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of images to retrieve (default: 1)"
    )
    parser.add_argument(
        "--show_all",
        action="store_true",
        help="Show all available columns from the GeoParquet"
    )
    
    args = parser.parse_args()
    
    try:
        results = get_ftw_image_info(
            data_root=args.data_root,
            country=args.country,
            split=args.split,
            index=args.index,
            window=args.window,
            count=args.count,
            show_all=args.show_all
        )
        
        # Print results
        print("\n" + "="*80)
        for result in results:
            print(f"\nImage {result['index']} in {args.country} {args.split} set:")
            print(f"  AOI ID: {result['aoi_id']}")
            print(f"  Filename: {result['filename']}")
            print(f"  Image path: {result['image_path']}")
            print(f"  Image exists: {result['exists']}")
            print(f"  Mask path: {result['mask_path']}")
            print(f"  Mask exists: {result['mask_exists']}")
            
            if args.show_all:
                print("\n  Additional fields:")
                for key, value in result.items():
                    if key not in ["index", "aoi_id", "filename", "image_path", 
                                   "mask_path", "exists", "mask_exists", "geometry"]:
                        print(f"    {key}: {value}")
        
        print("="*80)
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
