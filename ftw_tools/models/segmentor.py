#!/usr/bin/env python3
import math
import torch
import torch.nn as nn
from torchvision.models.segmentation.deeplabv3 import ASPP
import torch.nn.functional as F


class MLPCombiner(nn.Module):
    def __init__(self, D=1024, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or 2 * D
        self.mlp = nn.Sequential(
            nn.Linear(2 * D, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, D)
        )
    def forward(self, x_2L_D):
        B, N2, D = x_2L_D.shape
        L = N2 // 2
        AB = torch.cat([x_2L_D[:, :L, :], x_2L_D[:, L:, :]], dim=-1)
        return self.mlp(AB)


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.BatchNorm2d(dim), nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.BatchNorm2d(dim)
        )
        self.act = nn.ReLU(inplace=True)
    def forward(self, x): return self.act(x + self.block(x))

class ConvLightDecoderHead(nn.Module):
    def __init__(self, dim, out_size=256, num_classes=3):
        super().__init__()
        self.out_size = out_size
        self.out_conv = nn.Conv2d(dim, num_classes, kernel_size=1)

    def forward(self, x_L_D):
        B, L, D = x_L_D.shape
        h = w = int(math.sqrt(L))
        x = x_L_D.transpose(1, 2).reshape(B, D, h, w)
        x = self.out_conv(x)
        return F.interpolate(x, size=(self.out_size, self.out_size), mode="bilinear", align_corners=False)


class ConvASPPDecoderHead(nn.Module):
    def __init__(self, dim, patch_size, num_classes):
        super().__init__()
        hidden, C_out, r = 512, 64, patch_size
        self.proj_in = nn.Conv2d(dim, hidden, 3, padding=1)
        self.res1 = ResidualBlock(hidden)
        self.res2 = ResidualBlock(hidden)
        self.aspp = ASPP(in_channels=hidden, atrous_rates=(6, 12, 18), out_channels=hidden)
        self.conv_ps = nn.Conv2d(hidden, C_out * r * r, 3, padding=1)
        self.shuffle = nn.PixelShuffle(r)
        self.out_conv = nn.Conv2d(C_out, num_classes, kernel_size=1)
    def forward(self, x_L_D):
        B, L, D = x_L_D.shape
        h = w = int(math.sqrt(L))
        x = x_L_D.transpose(1, 2).reshape(B, D, h, w)
        x = self.proj_in(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.aspp(x)
        x = self.shuffle(self.conv_ps(x))
        return self.out_conv(x)


class LearnableFinalUpsample(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, in_size=224, out_size=256):
        super().__init__()
        scale = max(1, round(out_size / in_size))
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2 * scale, scale, scale // 2)
        self.out_size = out_size

    def forward(self, x):
        x = self.up(x)
        return F.interpolate(x, size=(self.out_size, self.out_size), mode='bilinear', align_corners=False)


class SegmentationHead(nn.Module):
    """
    fusion_type: 'mlp'
    decoder_type: 'conv_w_aspp'
    """
    def __init__(self, fusion_type="mlp", decoder_type="conv_w_aspp",
                 dim=1024, patch_size=16, num_classes=3,
                original_input_size=256):
        super().__init__()
        assert fusion_type in ["mlp"]
        assert decoder_type in ["conv_w_aspp","conv_light"]

        self.decoder_type = decoder_type
        self.dim = dim
        self.patch_size = patch_size
        self.original_input_size = original_input_size

        if self.original_input_size != 256 and decoder_type == "conv_w_aspp":
            self.final_upsample = LearnableFinalUpsample(
                in_ch=num_classes,
                out_ch=num_classes,
                in_size=self.original_input_size,
                out_size=256
            )
        else:
            self.final_upsample = nn.Identity()
        
        if fusion_type == "mlp":
                self.fuse = MLPCombiner(D=dim)
                self.fused_dim = dim
        else:
            raise NotImplementedError(f"Fusion type '{fusion_type}' not implemented.")
        
        # Decoder
        if decoder_type == "conv_w_aspp":
            self.decoder = ConvASPPDecoderHead(self.fused_dim, patch_size, num_classes)
        elif decoder_type == "conv_light":
            self.decoder = ConvLightDecoderHead(dim=self.fused_dim, out_size=256, num_classes=num_classes)
        else:
            raise NotImplementedError(f"Decoder type '{decoder_type}' not implemented.")

    def forward(self, x):
        # 🔹 STEP 1: Extract features
        if isinstance(x, dict):
            feats = x["feat"]
        else:
            feats = x

        # 🔹 STEP 2: Handle backward compatibility
        if feats.ndim == 4 and feats.shape[1] == 2:
            feats = feats.reshape(feats.shape[0], -1, feats.shape[-1])
        elif feats.ndim != 3:
            raise AssertionError("Input 'feat' must be of shape (B, 2, N, D) or (B, N2, D).")
        # 🔹 STEP 3: Fuse features and decod
        # import code; code.interact(local=dict(globals(), **locals()))
        fused_tokens = self.fuse(feats)
        seg_logits = self.decoder(fused_tokens)
        seg_logits = self.final_upsample(seg_logits)
        return seg_logits


# ============================================================
# ✅ Sanity test — iterate over real model embedding configs
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def count_parameters(model):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        return total_params, trainable_params

    model_configs = {
        "croma":    {"input": 120, "patch": 8,  "dim": 768,  "embed_hw": 15},
        "galileo":  {"input": 256, "patch": 4,  "dim": 768,  "embed_hw": 64},
        "decur":    {"input": 224, "patch": 16, "dim": 384,  "embed_hw": 14},
        "dofa":     {"input": 224, "patch": 16, "dim": 1024, "embed_hw": 14},
        "prithvi":  {"input": 224, "patch": 16, "dim": 1024, "embed_hw": 14},
        "satlas":   {"input": 256, "patch": 16, "dim": 768,  "embed_hw": 16},
        "softcon":  {"input": 224, "patch": 14, "dim": 384,  "embed_hw": 16},
        "clay":     {"input": 256, "patch": 8,  "dim": 1024, "embed_hw": 32},
        "dinov3":   {"input": 256, "patch": 16, "dim": 1024, "embed_hw": 16},
        "terrafm":  {"input": 256, "patch": 16, "dim": 768,  "embed_hw": 16},
        "terramind": {"input": 256, "patch": 16, "dim": 768,  "embed_hw": 16},
    }

    B, num_classes = 2, 3

    for name, cfg in model_configs.items():
        input_size = cfg["input"]
        patch_size = cfg["patch"]
        dim = cfg["dim"]
        embed_hw = cfg["embed_hw"]

        tokens_per_view = embed_hw * embed_hw
        feats = torch.randn(B, 2, tokens_per_view, dim, device=device)

        decoders= ["conv_w_aspp", "conv_light"]
        for decoder_type in decoders:
            print(f"\n🔹 Testing model={name:10s} | decdoer={decoder_type} | input={input_size} | patch={patch_size} | dim={dim} | tokens={embed_hw}x{embed_hw}")
            model = SegmentationHead(
                fusion_type="mlp",
                decoder_type=decoder_type,
                dim=dim,
                patch_size=patch_size,
                num_classes=num_classes,
                original_input_size=input_size,
            ).to(device)
            print(count_parameters(model))

            out = model({"feat": feats})
            print("   ✅ Output:", tuple(out.shape))
