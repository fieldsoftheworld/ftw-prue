import torch
import torch.nn as nn
import torch.nn.functional as F

from ...nn.activations.sigmoid_crisp import SigmoidCrisp
from ...nn.pooling.psp_pooling import PSP_Pooling
from ...nn.layers.conv2Dnormed import Conv2DNormed


class HeadSingle(nn.Module):
    """Helper classification head, for a single layer output"""
    
    def __init__(self, nfilters, NClasses, depth=2, norm_type='BatchNorm', norm_groups=None, in_channels=None, **kwargs):
        super().__init__()
        
        if in_channels is None:
            in_channels = nfilters

        self.logits = nn.Sequential()
        for i in range(depth):
            if i == 0:
                self.logits.append(Conv2DNormed(
                    channels=nfilters, 
                    kernel_size=(3, 3), 
                    padding=(1, 1), 
                    norm_type=norm_type, 
                    norm_groups=norm_groups,
                    in_channels=in_channels,
                    out_channels=nfilters
                ))
            else:
                self.logits.append(Conv2DNormed(
                    channels=nfilters, 
                    kernel_size=(3, 3), 
                    padding=(1, 1), 
                    norm_type=norm_type, 
                    norm_groups=norm_groups
                ))
            self.logits.append(nn.ReLU())
        self.logits.append(nn.Conv2d(nfilters, NClasses, kernel_size=1, padding=0))

    def forward(self, input):
        return self.logits(input)


class Head_CMTSK_BC(nn.Module):
    """
    BC: Balanced (features) Crisp (boundaries)
    Conditioned multitask head for segmentation, boundaries, and distance maps.
    """
    
    def __init__(self, nfilters_init, NClasses, norm_type='BatchNorm', norm_groups=None, **kwargs):
        super().__init__()
        
        self.model_name = "Head_CMTSK_BC"
        self.nfilters = nfilters_init  
        self.NClasses = NClasses
        
        self.psp_2ndlast = PSP_Pooling(self.nfilters, norm_type=norm_type, norm_groups=norm_groups)

        for conv in self.psp_2ndlast.convs:
            conv.conv = nn.Conv2d(
                in_channels=2*self.nfilters,  
                out_channels=self.nfilters,
                kernel_size=(1, 1),
                padding=(0, 0)
            )

            if hasattr(conv, 'norm'):
                if isinstance(conv.norm, nn.BatchNorm2d):
                    conv.norm = nn.BatchNorm2d(self.nfilters)
                elif isinstance(conv.norm, nn.InstanceNorm2d):
                    conv.norm = nn.InstanceNorm2d(self.nfilters)
                elif isinstance(conv.norm, nn.LayerNorm):
                    conv.norm = nn.LayerNorm([self.nfilters, 1, 1])
                elif isinstance(conv.norm, nn.GroupNorm):
                    conv.norm = nn.GroupNorm(num_groups=norm_groups, num_channels=self.nfilters)
        
        self.psp_2ndlast.conv_norm_final = Conv2DNormed(
            channels=self.nfilters,
            kernel_size=(1, 1),
            padding=(0, 0),
            norm_type=norm_type, 
            norm_groups=norm_groups,
            in_channels=64 + 32 * (self.psp_2ndlast.depth),
            out_channels=self.nfilters
        )

        self.bound_logits = HeadSingle(self.nfilters, self.NClasses, norm_type=norm_type, norm_groups=norm_groups, in_channels=2*self.nfilters)
        self.bound_Equalizer = Conv2DNormed(
            channels=self.nfilters, 
            kernel_size=1, 
            norm_type=norm_type, 
            norm_groups=norm_groups,
            in_channels=self.NClasses,  
            out_channels=self.nfilters
        )
        
        self.distance_logits = HeadSingle(self.nfilters, 1, norm_type=norm_type, norm_groups=norm_groups, in_channels=2*self.nfilters) 
        self.dist_Equalizer = Conv2DNormed(
            channels=self.nfilters, 
            kernel_size=1, 
            norm_type=norm_type, 
            norm_groups=norm_groups,
            in_channels= 1,  #self.NClasses,  # From distance logits output 
            out_channels=self.nfilters
        )

        self.Comb_bound_dist = Conv2DNormed(
            channels=self.nfilters, 
            kernel_size=1, 
            norm_type=norm_type, 
            norm_groups=norm_groups,
            in_channels=2*self.nfilters,  # After concatenation of boundEq and distEq
            out_channels=self.nfilters
        )

        self.final_segm_logits = HeadSingle(self.nfilters, self.NClasses, norm_type=norm_type, norm_groups=norm_groups, in_channels=2*self.nfilters)
     
        self.CrispSigm = SigmoidCrisp()

        if self.NClasses == 1:
            self.ChannelAct = lambda x: torch.sigmoid(x)
        else:
            self.ChannelAct = lambda x: F.softmax(x, dim=1)

        self.DistanceAct = lambda x: torch.sigmoid(x)
        
    def forward(self, UpConv4, conv1):
 
        convl = torch.cat([conv1, UpConv4], dim=1)
        conv = self.psp_2ndlast(convl)

        conv = F.relu(conv)
    
        dist = self.distance_logits(convl)  # do not use max pooling for distance
        # dist = self.ChannelAct(dist)
        dist = self.DistanceAct(dist)
        distEq = F.relu(self.dist_Equalizer(dist))  

        bound = torch.cat([conv, distEq], dim=1)
        bound = self.bound_logits(bound)
        bound = self.CrispSigm(bound) 
        boundEq = F.relu(self.bound_Equalizer(bound))

        comb_bd = self.Comb_bound_dist(torch.cat([boundEq, distEq], dim=1))
        comb_bd = F.relu(comb_bd)

        all_layers = torch.cat([comb_bd, conv], dim=1)
        final_segm = self.final_segm_logits(all_layers)
        final_segm = self.ChannelAct(final_segm)

        return final_segm, bound, dist
