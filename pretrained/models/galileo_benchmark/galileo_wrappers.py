"""
Unified wrapper classes for Galileo benchmark models.

This module provides encoder wrappers that adapt Galileo benchmark models
to the existing FTW pretrained model interface, enabling consistent usage
for both batch feature extraction and single-sample inference.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys

galileo_path = Path(__file__).parent / "galileo"
if str(galileo_path) not in sys.path:
    sys.path.insert(0, str(galileo_path))

from src.eval.baseline_models import (
    CROMAWrapper,
    DeCurWrapper,
    DOFAWrapper,
    PrithviWrapper,
    SatlasWrapper,
    SoftConWrapper,
)
from src.galileo import GalileoWrapper

S2_BAND_NAMES = [
    "B01", "B02", "B03", "B04", "B05", "B06", "B07",
    "B08", "B08A", "B09", "B10", "B11", "B12"
]

OURS_S2_MEAN = torch.tensor([
    1395.34, 1395.34, 1338.40, 1343.09, 1543.86, 2186.20, 2525.09,
    2410.33, 2750.28, 2750.28, 2234.91, 2234.91, 1474.53,
], dtype=torch.float32)

OURS_S2_STD = torch.tensor([
    917.70, 917.70, 913.29, 1092.68, 1047.22, 1048.01, 1143.69,
    1098.97, 1204.47, 1204.47, 1145.97, 1145.97, 980.24,
], dtype=torch.float32)

DOFA_S2_MEAN = torch.tensor([
    114.1099739, 114.81779093, 126.63977424, 84.33539309,
    97.84789168, 103.94461911, 101.435633, 72.32804172,
    56.66528851
], dtype=torch.float32)

DOFA_S2_STD = torch.tensor([
    77.84352553, 69.96844919, 67.42465279, 64.57022983, 61.72545487,
    61.34187099, 60.29744676, 47.88519516, 42.55886798
], dtype=torch.float32)

while len(DOFA_S2_MEAN) < 13:
    DOFA_S2_MEAN = torch.cat([DOFA_S2_MEAN, torch.tensor([DOFA_S2_MEAN.mean().item()])])
    DOFA_S2_STD = torch.cat([DOFA_S2_STD, torch.tensor([DOFA_S2_STD.mean().item()])])

IMPUTES = [
    ("B04", "B05"), ("B04", "B06"), ("B08", "B07"),
    ("B08", "B08A"), ("B08", "B09"), ("B08", "B10"),
    ("B08", "B11"), ("B08", "B12")
]


def impute_bands_torch(image_tensor, available_bands, imputes, all_bands, device):
    """
    Impute missing Sentinel-2 bands from 4-band input.
    
    Args:
        image_tensor: [B, 4, H, W] tensor with bands [B04, B03, B02, B08] (Red, Green, Blue, NIR)
        available_bands: List of available band names ["B04", "B03", "B02", "B08"]
        imputes: List of (source, target) tuples for imputation
        all_bands: List of all 13 Sentinel-2 band names
        device: torch.device
    
    Returns:
        [B, 13, H, W] tensor with imputed bands
    """
    B, C, H, W = image_tensor.shape
    assert C == 4, f"Expected 4 channels, got {C}"
    
    imputed_list = []
    for band in all_bands:
        if band in available_bands:
            idx = available_bands.index(band)
            imputed_list.append(image_tensor[:, idx:idx+1, :, :])
        else:
            imputed = False
            for src, tgt in imputes:
                if tgt == band and src in available_bands:
                    src_idx = available_bands.index(src)
                    imputed_list.append(image_tensor[:, src_idx:src_idx+1, :, :])
                    imputed = True
                    break
            if not imputed:
                imputed_list.append(torch.zeros(B, 1, H, W, device=device))
    
    return torch.cat(imputed_list, dim=1)


class BaseGalileoEncoder(nn.Module):
    """Base class for Galileo benchmark encoders."""
    
    def __init__(self, wrapper_class, wrapper_kwargs, freeze_encoder="all"):
        super().__init__()
        self.model = wrapper_class(**wrapper_kwargs)
        self.freeze_encoder = freeze_encoder
        
        if freeze_encoder == "all":
            for param in self.model.parameters():
                param.requires_grad = False
        
        self.model.eval()
    
    def preprocess_4band_to_13band(self, x, mean, std, device):
        """
        Convert 4-band input [B, 4, H, W] to 13-band Sentinel-2 format.
        
        Args:
            x: [B, 4, H, W] tensor with bands [Red, Green, Blue, NIR]
            mean: Mean tensor for normalization (13 values)
            std: Std tensor for normalization (13 values)
            device: torch.device
        
        Returns:
            [B, 13, H, W] normalized tensor
        """
        available_bands = ["B04", "B03", "B02", "B08"]
        
        s2_13band = impute_bands_torch(x, available_bands, IMPUTES, S2_BAND_NAMES, device)
        
        mean = mean.to(device)
        std = std.to(device)
        s2_normalized = (s2_13band - mean.view(1, 13, 1, 1)) / std.view(1, 13, 1, 1)
        
        return s2_normalized
    
    def forward(self, datacube):
        """
        Forward pass compatible with existing interface.
        
        Args:
            datacube: torch.Tensor [B, C, H, W] where C=4 or C=8
        
        Returns:
            torch.Tensor [B, L, D] patch embeddings
        """
        x = datacube
        B, C, H, W = x.shape
        
        if C == 4:
            s2_13band = self.preprocess_4band_to_13band(
                x, self.mean, self.std, x.device
            )
            s2_hwc = s2_13band.permute(0, 2, 3, 1)
            
            with torch.no_grad():
                patch_embeddings = self.model(s2=s2_hwc)
            
            return patch_embeddings
        
        elif C == 8:
            x_a = x[:, :4, :, :]
            x_b = x[:, 4:, :, :]
            
            s2_13band_a = self.preprocess_4band_to_13band(
                x_a, self.mean, self.std, x.device
            )
            s2_13band_b = self.preprocess_4band_to_13band(
                x_b, self.mean, self.std, x.device
            )
            
            s2_hwc_a = s2_13band_a.permute(0, 2, 3, 1)
            s2_hwc_b = s2_13band_b.permute(0, 2, 3, 1)
            
            with torch.no_grad():
                patches_a = self.model(s2=s2_hwc_a)
                patches_b = self.model(s2=s2_hwc_b)
            
            return torch.cat([patches_a, patches_b], dim=1)
        
        else:
            raise ValueError(f"Expected 4 or 8 channels, got {C}")


class CROMAEncoder(BaseGalileoEncoder):
    """CROMA encoder wrapper.
    
    Note: CROMA's preprocess method expects 13-band input and removes B10 internally.
    """
    
    def __init__(self, ckpt_path, size="base", freeze_encoder="all", device="cpu"):
        wrapper_kwargs = {
            "weights_path": Path(ckpt_path),
            "size": size,
            "modality": "optical",
            "do_pool": False,
            "load_state": True,
        }
        super().__init__(CROMAWrapper, wrapper_kwargs, freeze_encoder)
        self.mean = OURS_S2_MEAN
        self.std = OURS_S2_STD
        self.model = self.model.to(device)


class DeCurEncoder(BaseGalileoEncoder):
    """DeCur encoder wrapper."""
    
    def __init__(self, ckpt_path, freeze_encoder="all", device="cpu"):
        wrapper_kwargs = {
            "weights_path": Path(ckpt_path),
            "modality": "optical",
            "do_pool": False,
            "load_state": True,
        }
        super().__init__(DeCurWrapper, wrapper_kwargs, freeze_encoder)
        self.mean = OURS_S2_MEAN
        self.std = OURS_S2_STD
        self.model = self.model.to(device)


class DOFAEncoder(BaseGalileoEncoder):
    """DOFA encoder wrapper."""
    
    def __init__(self, ckpt_path, size="base", freeze_encoder="all", device="cpu"):
        wrapper_kwargs = {
            "weights_path": Path(ckpt_path),
            "size": size,
            "do_pool": False,
            "load_state": True,
        }
        super().__init__(DOFAWrapper, wrapper_kwargs, freeze_encoder)
        self.mean = DOFA_S2_MEAN
        self.std = DOFA_S2_STD
        self.model = self.model.to(device)


class PrithviEncoder(BaseGalileoEncoder):
    """Prithvi encoder wrapper."""
    
    def __init__(self, ckpt_path, freeze_encoder="all", device="cpu"):
        wrapper_kwargs = {
            "weights_path": Path(ckpt_path),
            "do_pool": False,
            "load_state": True,
        }
        super().__init__(PrithviWrapper, wrapper_kwargs, freeze_encoder)
        self.mean = OURS_S2_MEAN
        self.std = OURS_S2_STD
        self.model = self.model.to(device)


class SatlasEncoder(BaseGalileoEncoder):
    """Satlas encoder wrapper.
    
    Note: Satlas expects 13-band input and reorders bands internally.
    """
    
    def __init__(self, ckpt_path, size="base", freeze_encoder="all", device="cpu"):
        wrapper_kwargs = {
            "weights_path": Path(ckpt_path),
            "size": size,
            "do_pool": False,
            "load_state": True,
        }
        super().__init__(SatlasWrapper, wrapper_kwargs, freeze_encoder)
        self.mean = OURS_S2_MEAN
        self.std = OURS_S2_STD
        self.model = self.model.to(device)


class SoftConEncoder(BaseGalileoEncoder):
    """SoftCon encoder wrapper."""
    
    def __init__(self, ckpt_path, size="base", freeze_encoder="all", device="cpu"):
        wrapper_kwargs = {
            "weights_path": Path(ckpt_path),
            "size": size,
            "modality": "optical",
            "do_pool": False,
            "load_state": True,
        }
        super().__init__(SoftConWrapper, wrapper_kwargs, freeze_encoder)
        self.mean = OURS_S2_MEAN
        self.std = OURS_S2_STD
        self.model = self.model.to(device)


class GalileoEncoder(BaseGalileoEncoder):
    """Galileo encoder wrapper."""
    
    def __init__(self, ckpt_path, freeze_encoder="all", device="cpu"):
        wrapper_kwargs = {
            "pretrained_path": Path(ckpt_path),
            "patch_size": 4,
            "month": 6,
            "do_pool": False,
            "load_state": True,
        }
        super().__init__(GalileoWrapper, wrapper_kwargs, freeze_encoder)
        self.mean = OURS_S2_MEAN
        self.std = OURS_S2_STD
        self.model = self.model.to(device)

