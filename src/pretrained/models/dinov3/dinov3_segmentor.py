"""
DinoV3 Segmentor for semantic segmentation tasks.
"""

import re

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn

import os
import sys


class SegmentEncoder(nn.Module):
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
        ckpt_path=None
    ):
        super().__init__(
        )

        # Set device
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        if ckpt_path == None:
            ckpt_path="/projects/bdbk/subashk/ckpts/DINOV3/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"
        self.encoder = torch.hub.load('facebookresearch/dinov3', 'dinov3_vitl16', weights=ckpt_path)
     

    def forward(self, datacube):
        """
        Forward pass of the SegmentEncoder.

        Args:
            datacube (dict): A dictionary containing the input datacube and
                meta information like time, latlon, gsd & wavelenths.

        Returns:
            list: A list of feature maps extracted from the datacube.
        """
        cube = datacube

        B, C, H, W = cube.shape
       
        if C == 3:
            # Patchify and create embeddings per patch
            patches = self.encoder.forward_features(cube.float())['x_norm_patchtokens'] # [B L D]
        elif C == 2 * 3:
            # Patchify and create embeddings per patch
            patches_a = self.encoder.forward_features(cube[:,:3,:,:].float())['x_norm_patchtokens'] 
            
            patches_b = self.encoder.forward_features(cube[:,3:,:,:].float())['x_norm_patchtokens'] 
            patches = torch.cat((patches_a, patches_b), dim=1) # [B 2*L D] #now L is 2*L

        return patches