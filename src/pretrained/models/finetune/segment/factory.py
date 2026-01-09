"""
Clay Segmentor for semantic segmentation tasks.

Attribution:
Decoder from Segformer: Simple and Efficient Design for Semantic Segmentation
with Transformers
Paper URL: https://arxiv.org/abs/2105.15203
"""

import re

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn
import sys
from pathlib import Path

# go up two levels: segment → finetune → clay/
clay_root = Path(__file__).resolve().parents[2]  # ../../ from factory.py
if str(clay_root) not in sys.path:
    sys.path.append(str(clay_root))

from src.model import Encoder


class SegmentEncoder(Encoder):
    """
    Encoder class for segmentation tasks, incorporating a feature pyramid
    network (FPN).

    Attributes:
        feature_maps (list): Indices of layers to be used for generating
        feature maps.
        ckpt_path (str): Path to the clay checkpoint file.
    """

    def __init__(  # noqa: PLR0913
        self,
        mask_ratio,
        patch_size,
        shuffle,
        dim,
        depth,
        heads,
        dim_head,
        mlp_ratio,
        ckpt_path=None,
        freeze_encoder="none"
    ):
        # Note: base Encoder.__init__ does not accept `freeze_encoder`.
        # Call base initializer with only the parameters it expects.
        super().__init__(
            mask_ratio,
            patch_size,
            shuffle,
            dim,
            depth,
            heads,
            dim_head,
            mlp_ratio,
        )

        # Handle freeze_encoder locally: options are 'none' or 'all'.
        # Freeze encoder parameters if requested.
        if isinstance(freeze_encoder, str) and freeze_encoder.lower() == "all":
            for param in self.parameters():
                param.requires_grad = False

        # Set device
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        # Load model from checkpoint if provided
        self.load_from_ckpt(ckpt_path)

    def load_from_ckpt(self, ckpt_path):
        """
        Load the model's state from a checkpoint file.

        Args:
            ckpt_path (str): The path to the checkpoint file.
        """
        if ckpt_path:
            # Load checkpoint
            ckpt = torch.load(ckpt_path, map_location=self.device)
            state_dict = ckpt.get("state_dict")

            # Prepare new state dict with the desired subset and naming
            new_state_dict = {
                re.sub(r"^model\.encoder\.", "", name): param
                for name, param in state_dict.items()
                if name.startswith("model.encoder")
            }

            # Load the modified state dict into the model
            model_state_dict = self.state_dict()
            for name, param in new_state_dict.items():
                if (
                    name in model_state_dict
                    and param.size() == model_state_dict[name].size()
                ):
                    model_state_dict[name].copy_(param)
                else:
                    print(f"No matching parameter for {name} with size {param.size()}")


    def forward(self, datacube):
        """
        Forward pass of the SegmentEncoder.

        Args:
            datacube (dict): A dictionary containing the input datacube and
                meta information like time, latlon, gsd & wavelenths.

        Returns:
            list: A list of feature maps extracted from the datacube.
        """
        cube, time, latlon, gsd, waves = (
            datacube["image"],  # [B C 256 256]
            datacube["time"],  # [B 4/8]
            datacube["latlon"],  # [B 4]
            datacube["gsd"],  # 1
            datacube["waves"],  # [N] or [B,N]
        )
        B, C, H, W = cube.shape
        if waves.ndim == 2:
            waves = waves[0]  # [N] assume all batch have same wavelengths
        if gsd.ndim == 1 and len(gsd) == B:
            gsd = gsd[0]  # [1] assume all batch have same gsd
        if C == 4:
            # Patchify and create embeddings per patch
            patches, waves_encoded = self.to_patch_embed(cube, waves)  # [B L D]
            patches = self.add_encodings(patches, time, latlon, gsd)  # [B L D]
            cls_tokens = repeat(self.cls_token, "1 1 D -> B 1 D", B=B)  # [B 1 D]
            patches = torch.cat((cls_tokens, patches), dim=1)  # [B (1 + L) D]
            patches = self.transformer(patches)
        elif C == 8:
            # Patchify and create embeddings per patch
            patches_a, waves_encoded = self.to_patch_embed(cube[:, :len(waves), :, :], waves)
            patches_a = self.add_encodings(patches_a, time[:,:4], latlon, gsd)  # [B L D]
            cls_tokens = repeat(self.cls_token, "1 1 D -> B 1 D", B=B)  # [B 1 D]
            patches_a = torch.cat((cls_tokens, patches_a), dim=1)  # [B (1 + L) D]
            patches_a = self.transformer(patches_a)[:, 1:, :]

            patches_b = self.to_patch_embed(cube[:, 4:, :, :], waves)[0]
            patches_b = self.add_encodings(patches_b, time[:,4:], latlon, gsd)  # [B L D]
            cls_tokens = repeat(self.cls_token, "1 1 D -> B 1 D", B=B)  # [B 1 D]
            patches_b = torch.cat((cls_tokens, patches_b), dim=1)  # [B (1 + L) D]
            patches_b = self.transformer(patches_b)[:, 1:, :]

            patches = torch.cat((patches_a, patches_b), dim=1) # [B 2*L D] #now L is 2*L
        # import code;code.interact(local=dict(globals(), **locals()));

        return patches
