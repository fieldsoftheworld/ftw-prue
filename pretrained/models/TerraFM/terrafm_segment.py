import torch
import torch.nn as nn
from .terrafm import terrafm_base, terrafm_large
import os
import sys

def map_to_sentinel12(x):
    """
    Map a [B, 4, H, W] tensor with bands ['red', 'green', 'blue', 'nir']
    to a [B, 12, H, W] Sentinel-2 L2A tensor with correct band positions.

    Unused channels are zero-filled.
    """
    # Sentinel-2 L2A 12-band order
    s2_band_order = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]
    
    # Input band order
    input_bands = ["red", "green", "blue", "nir"]
    
    # Mapping input band names to their Sentinel-2 index
    band_map = {
        "red": s2_band_order.index("B4"),
        "green": s2_band_order.index("B3"),
        "blue": s2_band_order.index("B2"),
        "nir": s2_band_order.index("B8"),
    }
    
    B, _, H, W = x.shape
    x12 = torch.zeros(B, 12, H, W, device=x.device)
    
    for i, band in enumerate(input_bands):
        x12[:, band_map[band]] = x[:, i]

    return x12


class TerraFMEncoderWrapper(nn.Module):
    def __init__(self, ckpt_path, freeze_encoder="none",in_chans=4,device="cpu"):
        super().__init__()
        self.model = terrafm_base(in_chans=in_chans)

        if ckpt_path != None:
            # Load pretrained weights
            state_dict = torch.load(ckpt_path, map_location=device)
            _ = self.model.load_state_dict(state_dict, strict=False)
        
        self.in_chans = in_chans
        self.dim = self.model.embed_dim
        self.patch_size = self.model.patch_embed.patch_size

        if freeze_encoder == 'all':
            for param in self.model.parameters():
                param.requires_grad = False
        
        elif freeze_encoder == 'exceptinput':
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.patch_embed.parameters():
                    param.requires_grad = True
        else:
            pass

        self.freeze_encoder = freeze_encoder

    def forward(self, datacube):
        x = datacube  # [B, C, H, W]
        # import code;code.interact(local=dict(globals(), **locals()));
        if x.shape[1] == self.in_chans:
            if self.freeze_encoder == "all": #Since we are using frozen terrafm, one thing we can try is creating a 12-channel input 
                x = map_to_sentinel12(x)
                patches = self.model.forward(x, return_cls=False,isl2a=True)[:,1:,:]  # [B, L, D]
            else:
                patches = self.model.forward(x, return_cls=False,isl2a=False)[:,1:,:]  # [B, L, D]

        elif x.shape[1] == 2*self.in_chans:
            # Patchify and create embeddings per patch
            x_a = x[:,:4,:,:]
            x_b = x[:,4:,:,:]
            if self.freeze_encoder == "all": #Since we are using frozen terrafm, one thing we can try is creating a 12-channel input 
                x_a = map_to_sentinel12(x_a)
                x_b = map_to_sentinel12(x_b)
                patches_a =  self.model.forward(x_a, return_cls=False,isl2a=True)[:,1:,:]
                patches_b =  self.model.forward(x_b, return_cls=False,isl2a=True)[:,1:,:]
            else:
                patches_a =  self.model.forward(x_a, return_cls=False,isl2a=False)[:,1:,:]
                patches_b =  self.model.forward(x_b, return_cls=False,isl2a=False)[:,1:,:]
            patches = torch.cat((patches_a, patches_b), dim=1)  # [B (L + L) D]
            
        return patches


