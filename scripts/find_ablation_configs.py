#!/usr/bin/env python3
"""
Find all YAML config files in an ablations directory (including subdirectories).

This script recursively searches for .yaml files in the specified directory
and returns their paths relative to the project root.
"""

import argparse
import sys
from pathlib import Path


def find_configs(ablations_dir: Path, recursive: bool = True) -> list[Path]:
    """
    Find all YAML config files in the ablations directory.

    Args:
        ablations_dir: Path to the ablations directory
        recursive: If True, search subdirectories recursively

    Returns:
        List of config file paths (relative to project root)
    """
    if not ablations_dir.exists():
        raise FileNotFoundError(f"Ablations directory not found: {ablations_dir}")

    if not ablations_dir.is_dir():
        raise ValueError(f"Path is not a directory: {ablations_dir}")

    configs = []

    if recursive:
        # Find all .yaml files recursively
        pattern = "**/*.yaml"
    else:
        # Only find .yaml files in the top level
        pattern = "*.yaml"

    for config_path in ablations_dir.glob(pattern):
        if config_path.is_file():
            configs.append(config_path)

    # Sort for consistent ordering
    configs.sort()

    return configs


def main():
    parser = argparse.ArgumentParser(description="Find all YAML config files in an ablations directory")
    parser.add_argument(
        "ablations_dir", type=str, help="Path to the ablations directory (relative to project root or absolute)"
    )
    parser.add_argument(
        "--recursive", action="store_true", default=True, help="Search subdirectories recursively (default: True)"
    )
    parser.add_argument(
        "--no-recursive", dest="recursive", action="store_false", help="Only search top-level directory"
    )
    parser.add_argument(
        "--format",
        choices=["paths", "names", "basenames"],
        default="paths",
        help="Output format: paths (full paths), names (with dir), basenames (filename only)",
    )

    args = parser.parse_args()

    # Get project root (parent of scripts directory)
    project_root = Path(__file__).parent.parent

    # Resolve ablations directory path
    ablations_path = Path(args.ablations_dir)
    if not ablations_path.is_absolute():
        ablations_path = project_root / ablations_path

    try:
        configs = find_configs(ablations_path, recursive=args.recursive)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if not configs:
        print(f"No YAML config files found in {ablations_path}", file=sys.stderr)
        return 1

    # Output configs in requested format
    for config in configs:
        if args.format == "paths":
            # Output relative to project root
            print(config.relative_to(project_root))
        elif args.format == "names":
            # Output with directory structure
            rel_path = config.relative_to(ablations_path)
            print(rel_path)
        elif args.format == "basenames":
            # Output just the filename
            print(config.name)

    return 0


if __name__ == "__main__":
    sys.exit(main())
