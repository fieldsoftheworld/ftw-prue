import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2DNormed(nn.Module):
    """2D Convolution with normalization"""
    
    def __init__(self, channels, kernel_size, stride=(1, 1), padding=(0, 0), 
                 dilation=(1, 1), groups=1, bias=True, norm_type='BatchNorm', 
                 norm_groups=None, in_channels=None, out_channels=None, **kwargs):
        super().__init__()
        
        # Allow separate input and output channels, defaulting to same if not specified
        if in_channels is None:
            in_channels = channels
        if out_channels is None:
            out_channels = channels
            
        self.conv = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding, 
            dilation=dilation, 
            groups=groups, 
            bias=bias
        )
        
        # Handle normalization
        if norm_type == 'BatchNorm':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm_type == 'InstanceNorm':
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm_type == 'LayerNorm':
            self.norm = nn.LayerNorm([out_channels, 1, 1])  # For 2D input
        elif norm_type == 'GroupNorm' and norm_groups is not None:
            self.norm = nn.GroupNorm(num_groups=norm_groups, num_channels=out_channels)
        else:
            raise NotImplementedError(f"Normalization {norm_type} not implemented")
        
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x
