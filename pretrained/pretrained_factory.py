import torch
import torch.nn as nn
from ftw_tools.models.segmentor import SegmentationHead
from .models.clay.finetune.segment.factory import SegmentEncoder as ClayEncoder
from .models.TerraFM.terrafm_segment import TerraFMEncoderWrapper as TerraFMEncoder
from .models.dinov3.dinov3_segmentor import SegmentEncoder as DinoV3Encoder
from .models.terramind.terramind import SegmentEncoder as TeraMindEncoder


def get_encoder(model_name: str, device: torch.device, weights_path: str=None):
    model_name = model_name.lower()

    # -------------------- CLAY --------------------
    if model_name == "clay":
        weights = weights_path

        encoder = ClayEncoder(
            mask_ratio=0.0,
            patch_size=8,
            shuffle=False,
            dim=1024,
            depth=24,
            heads=16,
            dim_head=64,
            mlp_ratio=4.0,
            ckpt_path=weights,
            freeze_encoder="all",
        ).to(device)
        encoder.eval()
        return encoder

    # -------------------- TERRAFM --------------------
    elif model_name == "terrafm":
        weights = weights_path
        encoder = TerraFMEncoder(
            ckpt_path=weights, in_chans=4,
            device=device, freeze_encoder="all"
        ).to(device)
        encoder.eval()
        return encoder

    # -------------------- DINOV3 --------------------
    elif model_name == "dinov3":
        weights = weights_path
        encoder = DinoV3Encoder(ckpt_path=weights).to(device)
        encoder.eval()
        return encoder

    # -------------------- TERRAMIND --------------------
    elif model_name == "terramind":
         encoder = TeraMindEncoder().to(device)
         encoder.eval()
         return encoder
    else:
        raise ValueError(f"Unsupported model: {model_name}")
