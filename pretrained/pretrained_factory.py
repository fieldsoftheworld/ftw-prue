import os

import torch
from .path_config import get_model_path


def get_encoder(model_name: str, device: torch.device, weights_path: str = None):
    """
    Get encoder for a pretrained model.

    Args:
        model_name: Name of the model
        device: torch.device
        weights_path: Path to model weights. If None, uses path_config.get_model_path().
            If that path does not exist (e.g. finetuned full-model checkpoint where
            encoder is inside the same checkpoint), builds encoder without loading
            so the caller can load state from the full checkpoint.

    Returns:
        Encoder model
    """
    model_name = model_name.lower()

    if weights_path is None:
        try:
            default_path = get_model_path(model_name)
            default_path_str = str(default_path)
            if os.path.isfile(default_path_str) or os.path.isdir(default_path_str):
                weights_path = default_path_str
            else:
                # Default path missing: e.g. finetuned full-model ckpt (encoder inside same file)
                weights_path = ""
        except (ValueError, OSError):
            weights_path = ""

    if model_name == "clay":
        from .models.clay.finetune.segment.factory import SegmentEncoder as ClayEncoder

        # Empty path: encoder will be loaded from full-model checkpoint (finetuned clay)
        ckpt_path = weights_path if weights_path and os.path.isfile(weights_path) else None
        encoder = ClayEncoder(
            mask_ratio=0.0,
            patch_size=8,
            shuffle=False,
            dim=1024,
            depth=24,
            heads=16,
            dim_head=64,
            mlp_ratio=4.0,
            ckpt_path=ckpt_path,
            freeze_encoder="all",
        ).to(device)
        # encoder.eval()
        return encoder

    elif model_name == "terrafm":
        from .models.TerraFM.terrafm_segment import TerraFMEncoderWrapper as TerraFMEncoder

        weights = weights_path
        encoder = TerraFMEncoder(ckpt_path=weights, in_chans=4, device=device, freeze_encoder="all").to(device)
        encoder.eval()
        return encoder

    elif model_name == "dinov3":
        from .models.dinov3.dinov3_segmentor import SegmentEncoder as DinoV3Encoder

        weights = weights_path
        encoder = DinoV3Encoder(ckpt_path=weights).to(device)
        encoder.eval()
        return encoder

    elif model_name == "terramind":
        from .models.terramind.terramind import SegmentEncoder as TeraMindEncoder

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
