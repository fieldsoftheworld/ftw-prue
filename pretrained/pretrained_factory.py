import torch
import torch.nn as nn
from ftw_tools.models.segmentor import SegmentationHead
from .models.clay.finetune.segment.factory import SegmentEncoder as ClayEncoder
from .models.TerraFM.terrafm_segment import TerraFMEncoderWrapper as TerraFMEncoder
from .models.dinov3.dinov3_segmentor import SegmentEncoder as DinoV3Encoder
from .models.terramind.terramind import SegmentEncoder as TeraMindEncoder
from .path_config import get_model_path


def get_encoder(model_name: str, device: torch.device, weights_path: str = None):
    """
    Get encoder for a pretrained model.

    Args:
        model_name: Name of the model
        device: torch.device
        weights_path: Path to model weights (defaults to path_config.get_model_path())

    Returns:
        Encoder model
    """
    model_name = model_name.lower()

    if weights_path is None:
        weights_path = str(get_model_path(model_name))

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
        # encoder.eval()
        return encoder

    elif model_name == "terrafm":
        weights = weights_path
        encoder = TerraFMEncoder(ckpt_path=weights, in_chans=4, device=device, freeze_encoder="all").to(device)
        encoder.eval()
        return encoder

    elif model_name == "dinov3":
        weights = weights_path
        encoder = DinoV3Encoder(ckpt_path=weights).to(device)
        encoder.eval()
        return encoder

    elif model_name == "terramind":
        encoder = TeraMindEncoder().to(device)
        encoder.eval()
        return encoder

    elif model_name == "croma":
        from .models.galileo_benchmark.galileo_wrappers import CROMAEncoder

        encoder = CROMAEncoder(ckpt_path=weights_path, size="base", freeze_encoder="all", device=device)
        encoder.eval()
        return encoder

    elif model_name == "decur":
        from .models.galileo_benchmark.galileo_wrappers import DeCurEncoder

        encoder = DeCurEncoder(ckpt_path=weights_path, freeze_encoder="all", device=device)
        encoder.eval()
        return encoder

    elif model_name == "dofa":
        from .models.galileo_benchmark.galileo_wrappers import DOFAEncoder

        encoder = DOFAEncoder(ckpt_path=weights_path, size="base", freeze_encoder="all", device=device)
        encoder.eval()
        return encoder

    elif model_name == "prithvi":
        from .models.galileo_benchmark.galileo_wrappers import PrithviEncoder

        encoder = PrithviEncoder(ckpt_path=weights_path, freeze_encoder="all", device=device)
        encoder.eval()
        return encoder

    elif model_name == "satlas":
        from .models.galileo_benchmark.galileo_wrappers import SatlasEncoder

        encoder = SatlasEncoder(ckpt_path=weights_path, size="base", freeze_encoder="all", device=device)
        encoder.eval()
        return encoder

    elif model_name == "softcon":
        from .models.galileo_benchmark.galileo_wrappers import SoftConEncoder

        encoder = SoftConEncoder(ckpt_path=weights_path, size="base", freeze_encoder="all", device=device)
        encoder.eval()
        return encoder

    elif model_name == "galileo":
        from .models.galileo_benchmark.galileo_wrappers import GalileoEncoder

        encoder = GalileoEncoder(ckpt_path=weights_path, freeze_encoder="all", device=device)
        encoder.eval()
        return encoder

    else:
        raise ValueError(f"Unsupported model: {model_name}")
