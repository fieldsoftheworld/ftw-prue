import torch
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


import torch

# ... (Assume get_encoder is defined or imported, and works for CLAY's dict input)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_names = ["clay", "terrafm", "dinov3", "terramind"]
    weights_paths = {
        "clay": "/projects/bdbk/subashk/ckpts/CLAY/clay-v1.5.ckpt",
        "terrafm": "/projects/bdbk/subashk/ckpts/TERRAFM/TerraFM-B.pth",
        "dinov3": "/projects/bdbk/subashk/ckpts/DINOV3/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth",
        "terramind": None,
    }
    
    # Define batch and image dimensions once
    B, C, H, W = 2, 4, 256, 256
    
    for name in model_names:
        print(f"\n🔹 Testing encoder: {name}")
        encoder = get_encoder(name, device=device, weights_path=weights_paths[name])
        if  name == "dinov3":
            C = 3
        else:
            C = 4
        if name == "clay":
            images = torch.randn(B, C, H, W, dtype=torch.float32).to(device)

            # 2. Time tensor: [B, 4] (temporal encoding: sin/cos of week and hour)
            times = torch.randn(B, 4, dtype=torch.float32).to(device)

            # 3. Lat/Lon tensor: [B, 4] (spatial encoding: sin/cos of lat and lon)
            latlons = torch.randn(B, 4, dtype=torch.float32).to(device)

            # 4. GSD tensor: [1] (Ground Sampling Distance, e.g., 10m)
            gsd = torch.tensor([10.0], dtype=torch.float32).to(device)

            # 5. Wavelengths tensor: [C] -> [4] (Wavelengths of the 4 bands)
            waves = torch.tensor(
                [0.492, 0.559, 0.665, 0.833], dtype=torch.float32
            ).to(device)

            x = {
                "platform": "sentinel-2-l2a",
                "image": images,
                "time": times,
                "latlon": latlons,
                "gsd": gsd,
                "waves": waves,
            }
            print(f"   Created **CLAY dictionary batch**. Input Keys: {list(x.keys())}")
            
        else:
            # --- Other models require only the image tensor: [B, C, H, W] ---
            x = torch.randn(B, C, H, W).to(device)
            print(f"   Created standard tensor batch with shape {x.shape}.")
        
        # Perform inference using the prepared input (x)
        with torch.no_grad():
            feats = encoder(x)
            
        print(f"   ✅ Output feature shape: {feats.shape}")