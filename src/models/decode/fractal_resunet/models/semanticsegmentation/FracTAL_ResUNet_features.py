import torch
import torch.nn as nn
import torch.nn.functional as F

from ...nn.layers.conv2Dnormed import Conv2DNormed
from ...nn.layers.attention import FTAttention2D
from ...nn.pooling.psp_pooling import PSP_Pooling
from ...nn.layers.scale import DownSample
from ...nn.layers.combine import combine_layers, combine_layers_wthFusion
from ...nn.units.fractal_resnet import FracTALResNet_unit


class FracTAL_ResUNet_features(nn.Module):
    """
    FracTAL_ResUNet features backbone.
    PyTorch implementation of the original MXNet model.
    
    If upFuse == True, then instead of concatenation of the encoder features 
    with the decoder features, the algorithm performs Fusion with relative attention.
    """
    
    def __init__(self, nfilters_init, depth, widths=[1], psp_depth=4, verbose=True, 
                 norm_type='BatchNorm', norm_groups=None, nheads_start=8, 
                 upFuse=False, ftdepth=5, in_channels=None, **kwargs):
        super().__init__()
        
        self.depth = depth
        self.upFuse = upFuse

        if len(widths) == 1 and depth != 1:
            widths = widths * depth
        else:
            assert depth == len(widths), ValueError("depth and length of widths must match, aborting ...")

        self.conv_first = Conv2DNormed(
            nfilters_init, 
            kernel_size=(1, 1), 
            norm_type=norm_type, 
            norm_groups=norm_groups,
            in_channels=in_channels if in_channels is not None else nfilters_init
        )
        
        # List of convolutions and pooling operators 
        self.convs_dn = nn.ModuleList()
        self.pools = nn.ModuleList()

        for idx in range(depth):
            nheads = nheads_start * 2**idx
            nfilters = nfilters_init * 2**idx
            
            if verbose:
                print(f"depth:= {idx}, nfilters: {nfilters}, nheads::{nheads}, widths::{widths[idx]}")
            
            tnet = nn.ModuleList()
            for _ in range(widths[idx]):
                tnet.append(FracTALResNet_unit(
                    nfilters=nfilters, 
                    nheads=nheads, 
                    ngroups=nheads, 
                    norm_type=norm_type, 
                    norm_groups=norm_groups, 
                    ftdepth=ftdepth
                ))
            self.convs_dn.append(tnet)

            if idx < depth - 1:
                self.pools.append(DownSample(nfilters, norm_type=norm_type, norm_groups=norm_groups))
        
        # Middle pooling operator 
        self.middle = PSP_Pooling(nfilters, depth=psp_depth, norm_type=norm_type, norm_groups=norm_groups)
                               
        self.convs_up = nn.ModuleList()
        self.UpCombs = nn.ModuleList()
        
        for idx in range(depth - 1, 0, -1):
            nheads = nheads_start * 2**idx 
            nfilters = nfilters_init * 2**(idx - 1)
            
            if verbose:
                print(f"depth:= {2*depth-idx-1}, nfilters: {nfilters}, nheads::{nheads}, widths::{widths[idx]}")
            
            tnet = nn.ModuleList()
            for _ in range(widths[idx]):
                tnet.append(FracTALResNet_unit(
                    nfilters=nfilters, 
                    nheads=nheads, 
                    ngroups=nheads, 
                    norm_type=norm_type, 
                    norm_groups=norm_groups, 
                    ftdepth=ftdepth
                ))
            self.convs_up.append(tnet)
            
            if upFuse:
                self.UpCombs.append(combine_layers_wthFusion(
                    nfilters=nfilters, 
                    nheads=nheads, 
                    norm_type=norm_type, 
                    norm_groups=norm_groups, 
                    ftdepth=ftdepth
                ))
            else:
                self.UpCombs.append(combine_layers(
                    nfilters, 
                    norm_type=norm_type, 
                    norm_groups=norm_groups
                ))
                
    def forward(self, input):
        conv1_first = self.conv_first(input)
 
        # ******** Going down ***************
        fusions = []

        # Workaround for potential PyTorch issues
        pools = conv1_first.clone()

        for idx in range(self.depth):
            conv1 = pools
            for unit in self.convs_dn[idx]:
                conv1 = unit(conv1)
                
            if idx < self.depth - 1:
                # Evaluate fusions 
                conv1 = conv1.clone()
                fusions.append(conv1)
                # Evaluate pools 
                pools = self.pools[idx](conv1)

        # Middle psppooling
        middle = self.middle(conv1)
        # Activation of middle layer
        middle = F.relu(middle)
        fusions.append(middle) 

        # ******* Coming up ****************
        convs_up = middle
        for idx in range(self.depth - 1):
            convs_up = self.UpCombs[idx](convs_up, fusions[-idx - 2])
            
            for unit in self.convs_up[idx]:
                convs_up = unit(convs_up)
            
        return convs_up, conv1_first
