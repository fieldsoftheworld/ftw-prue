import torch
import torch.nn as nn
import torch.nn.functional as F

from .scale import UpSample
from .conv2Dnormed import Conv2DNormed


class combine_layers(nn.Module):
    """
    For combining layers with standard concatenation.
    For combining layers with Fusion (i.e. relative attention), see combine_layers_wthFusion
    """
    
    def __init__(self, nfilters, norm_type='BatchNorm', norm_groups=None, **kwargs):
        super().__init__()
        
        # This performs convolution, no BatchNormalization. No need for bias. 
        # The UpSample needs to know the input channels (which are 2*nfilters from the lower resolution)
        self.up = UpSample(2*nfilters, norm_type=norm_type, norm_groups=norm_groups) 

        self.conv_normed = Conv2DNormed(
            channels=nfilters, 
            kernel_size=(1, 1),
            padding=(0, 0), 
            norm_type=norm_type,
            norm_groups=norm_groups,
            in_channels=2*nfilters,  # After concatenation of upsampled and skip connection
            out_channels=nfilters
        )
        
    def forward(self, layer_lo, layer_hi):
        up = self.up(layer_lo)
        up = F.relu(up)
        x = torch.cat([up, layer_hi], dim=1)
        x = self.conv_normed(x)
        return x


class combine_layers_wthFusion(nn.Module):
    """Combining layers with attention fusion"""
    
    def __init__(self, nfilters, nheads, norm_type='BatchNorm', norm_groups=None, ftdepth=5, **kwargs):
        super().__init__()
        
        # This would need to be implemented with attention fusion
        # For now, using standard combination
        self.combine = combine_layers(nfilters, norm_type, norm_groups)
        
    def forward(self, layer_lo, layer_hi):
        # TODO: Implement attention fusion
        return self.combine(layer_lo, layer_hi)
