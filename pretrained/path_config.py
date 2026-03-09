"""
Path configuration for pretrained models and data.

Uses environment variables with sensible defaults.
Users can override by setting environment variables:
- FTW_CKPT_BASE_DIR: Base directory for model checkpoints
- FTW_DATA_ROOT: Root directory for FTW dataset
- FTW_METADATA_PATH: Path to metadata YAML file
"""

import os
from pathlib import Path


def _get_ckpt_base_dir():
    """Get base checkpoint directory from environment or default."""
    if "FTW_CKPT_BASE_DIR" in os.environ:
        return Path(os.environ["FTW_CKPT_BASE_DIR"])

    work_dir = os.environ.get("WORK_DIR", ".")
    return Path(work_dir) / "gfm_ckpts" / "encoders"


def _get_data_root():
    """Get FTW data root directory from environment or default."""
    if "FTW_DATA_ROOT" in os.environ:
        return Path(os.environ["FTW_DATA_ROOT"])

    if "FTW_DATA_DIR" in os.environ:
        return Path(os.environ["FTW_DATA_DIR"])

    return Path("./data/ftw")


def _get_metadata_path():
    """Get metadata YAML path from environment or default."""
    if "FTW_METADATA_PATH" in os.environ:
        return Path(os.environ["FTW_METADATA_PATH"])

    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "configs" / "metadata.yaml"


CKPT_BASE_DIR = _get_ckpt_base_dir()
DATA_ROOT = _get_data_root()
METADATA_PATH = _get_metadata_path()


MODEL_PATHS = {
    "clay": CKPT_BASE_DIR / "clay-v1.5.ckpt",
    "terrafm": CKPT_BASE_DIR / "TerraFM-B.pth",
    "dinov3": CKPT_BASE_DIR / "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth",
    "croma": CKPT_BASE_DIR / "GALILEO" / "croma",
    "decur": CKPT_BASE_DIR / "GALILEO" / "decur",
    "dofa": CKPT_BASE_DIR / "GALILEO" / "dofa",
    "prithvi": CKPT_BASE_DIR / "GALILEO" / "prithvi",
    "satlas": CKPT_BASE_DIR / "GALILEO" / "satlas",
    "softcon": CKPT_BASE_DIR / "GALILEO" / "softcon",
    "galileo": CKPT_BASE_DIR / "GALILEO" / "galileo",
}


def get_model_path(model_name: str) -> Path:
    """
    Get checkpoint path for a model.

    Args:
        model_name: Name of the model (e.g., "clay", "croma", etc.)

    Returns:
        Path to the model checkpoint directory or file
    """
    model_name = model_name.lower()
    if model_name not in MODEL_PATHS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_PATHS.keys())}")
    return MODEL_PATHS[model_name]


def get_data_root() -> Path:
    """
    Get FTW data root directory.

    Returns:
        Path to the FTW data root directory
    """
    return DATA_ROOT


def get_metadata_path() -> Path:
    """
    Get metadata YAML file path.

    Returns:
        Path to the metadata YAML file
    """
    return METADATA_PATH
