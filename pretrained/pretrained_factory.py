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


class FullGFMModel(nn.Module):
    """
    Generic encoder→decoder segmentation model.

    Accepts both:
      - dict inputs (CLAY metadata-aware mode)
      - tensor inputs for other backbones
    """

    def __init__(
        self,
        backbone_name: str,
        decoder_kwargs: dict,
        weights_path: str | None,
        device: torch.device,
    ):
        super().__init__()

        # Encoder from factory
        self.encoder = get_encoder(
            model_name=backbone_name,
            device=device,
            weights_path=weights_path
        )
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Decoder from segmentation module
        self.decoder = SegmentationHead(**decoder_kwargs)

    def forward(self, x):
        feats = self.encoder(x)
        # import code; code.interact(local=locals())
        return self.decoder(feats)


def get_full_model(backbone, decoder_kwargs, weights_path, device):
    """
    Exposed factory function so trainer can request full model.
    """
    return FullGFMModel(
        backbone_name=backbone,
        decoder_kwargs=decoder_kwargs,
        weights_path=weights_path,
        device=device,
    )


MODEL_CONFIGS = {
    "croma":    {"input": 120, "patch": 8,  "dim": 768},
    "galileo":  {"input": 256, "patch": 4,  "dim": 768},
    "decur":    {"input": 224, "patch": 16, "dim": 384},
    "dofa":     {"input": 224, "patch": 16, "dim": 1024},
    "prithvi":  {"input": 224, "patch": 16, "dim": 1024},
    "satlas":   {"input": 256, "patch": 16, "dim": 768},
    "softcon":  {"input": 224, "patch": 14, "dim": 384},
    "clay":     {"input": 256, "patch": 8,  "dim": 1024},
    "dinov3":   {"input": 256, "patch": 16, "dim": 1024},
    "terrafm":  {"input": 256, "patch": 16, "dim": 768},
    "terramind": {"input": 256, "patch": 16, "dim": 768},
}


if __name__ == "__main__":

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weights_paths = {
        "clay": "/projects/bdbk/subashk/ckpts/CLAY/clay-v1.5.ckpt",
        "terrafm": "/projects/bdbk/subashk/ckpts/TERRAFM/TerraFM-B.pth",
        "dinov3": "/projects/bdbk/subashk/ckpts/DINOV3/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth",
        "terramind": None,
    }

    B = 2
    num_classes = 3

    for name, cfg in MODEL_CONFIGS.items():
        if name in ["clay", "terrafm", "dinov3", "terramind"]:
            print(f"\n🔹 Testing Backbone={name} | dim={cfg['dim']} | patch={cfg['patch']} ")

            # Build segmentation head configuration
            decoder_kwargs = dict(
                num_classes=num_classes,
                dim=cfg["dim"],
                patch_size=cfg["patch"],
                fusion_type="mlp",
                decoder_type="conv_w_aspp",
                original_input_size=cfg["input"],
            )

            # Instantiate full model
            model = get_full_model(
                backbone=name,
                decoder_kwargs=decoder_kwargs,
                weights_path=weights_paths.get(name),
                device=device
            ).to(device)

            # Create valid encoder input
            if name == "clay":
                # Two windows → channels = 2 * 4
                C = 4 * 2
                images = torch.randn(B, C, cfg["input"], cfg["input"]).to(device)

                time = torch.randn(B, 8).to(device)  # 4+4 split for windows

                latlon = torch.randn(B, 4).to(device)
                gsd = torch.tensor([10.0], dtype=torch.float32).to(device)
                waves = torch.tensor([0.492, 0.559, 0.665, 0.833],
                                    dtype=torch.float32).to(device)

                x = dict(
                    platform="sentinel-2-l2a",
                    image=images,
                    time=time,
                    latlon=latlon,
                    gsd=gsd,
                    waves=waves,
                )
                print(f"   ✔ CLAY test batch constructed")

            else:
                # Two windows means 2 fusion views
                C = 3*2 if name == "dinov3" else 4*2
                x = torch.randn(B, C, cfg["input"], cfg["input"]).to(device)
                print(f"   ✔ Tensor batch constructed: {x.shape}")

            # Forward pass sanity check
            model.eval()
            with torch.no_grad():
                out = model(x)

            print(f"   ✅ Output segmentation logits shape: {tuple(out.shape)}")