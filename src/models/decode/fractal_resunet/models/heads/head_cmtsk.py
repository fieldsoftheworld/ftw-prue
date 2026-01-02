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
        
        # Use provided in_channels or default to nfilters
        if in_channels is None:
            in_channels = nfilters

        self.logits = nn.Sequential()
        for i in range(depth):
            if i == 0:
                # First layer needs to handle the input channels
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
                # Subsequent layers maintain the same channel count
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
        self.nfilters = nfilters_init  # Initial number of filters 
        self.NClasses = NClasses
        
        # The PSP pooling needs to handle 2*nfilters input channels (from concatenation)
        self.psp_2ndlast = PSP_Pooling(self.nfilters, norm_type=norm_type, norm_groups=norm_groups)
        # Override the PSP pooling to accept the correct input channels
        # Fix all the individual conv layers
        for conv in self.psp_2ndlast.convs:
            conv.conv = nn.Conv2d(
                in_channels=2*self.nfilters,  # From concatenation of conv1 and UpConv4
                out_channels=self.nfilters,
                kernel_size=(1, 1),
                padding=(0, 0)
            )
            # Update the normalization layer to match the output channels
            if hasattr(conv, 'norm'):
                if isinstance(conv.norm, nn.BatchNorm2d):
                    conv.norm = nn.BatchNorm2d(self.nfilters)
                elif isinstance(conv.norm, nn.InstanceNorm2d):
                    conv.norm = nn.InstanceNorm2d(self.nfilters)
                elif isinstance(conv.norm, nn.LayerNorm):
                    conv.norm = nn.LayerNorm([self.nfilters, 1, 1])
                elif isinstance(conv.norm, nn.GroupNorm):
                    conv.norm = nn.GroupNorm(num_groups=norm_groups, num_channels=self.nfilters)
        
        # Fix the final conv layer
        self.psp_2ndlast.conv_norm_final = Conv2DNormed(
            channels=self.nfilters,
            kernel_size=(1, 1),
            padding=(0, 0),
            norm_type=norm_type, 
            norm_groups=norm_groups,
            in_channels=64 + 32 * (self.psp_2ndlast.depth),  # 64 + 32*4 = 192 channels
            out_channels=self.nfilters
        )
        
        # bound logits 
        # This processes the concatenation of conv (32) and distEq (32) = 64 channels
        self.bound_logits = HeadSingle(self.nfilters, self.NClasses, norm_type=norm_type, norm_groups=norm_groups, in_channels=2*self.nfilters)
        self.bound_Equalizer = Conv2DNormed(
            channels=self.nfilters, 
            kernel_size=1, 
            norm_type=norm_type, 
            norm_groups=norm_groups,
            in_channels=self.NClasses,  # From boundary logits output
            out_channels=self.nfilters
        )
        
        # distance logits -- deeper for better reconstruction 
        # This processes convl which has 64 channels (32+32 from concatenation)
        self.distance_logits = HeadSingle(self.nfilters, 1, norm_type=norm_type, norm_groups=norm_groups, in_channels=2*self.nfilters) # self.NClasses replaced it with 1
        self.dist_Equalizer = Conv2DNormed(
            channels=self.nfilters, 
            kernel_size=1, 
            norm_type=norm_type, 
            norm_groups=norm_groups,
            in_channels= 1,  #self.NClasses,  # From distance logits output # self.NClasses replaced it with 1
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

        # Segmentation logits -- deeper for better reconstruction 
        # This processes the concatenation of comb_bd (32) and conv (32) = 64 channels
        self.final_segm_logits = HeadSingle(self.nfilters, self.NClasses, norm_type=norm_type, norm_groups=norm_groups, in_channels=2*self.nfilters)
     
        self.CrispSigm = SigmoidCrisp()

        # Last activation, customization for binary results
        if self.NClasses == 1:
            self.ChannelAct = lambda x: torch.sigmoid(x)
        else:
            self.ChannelAct = lambda x: F.softmax(x, dim=1)

        self.DistanceAct = lambda x: torch.sigmoid(x)
        
    def forward(self, UpConv4, conv1):
        # second last layer 
        convl = torch.cat([conv1, UpConv4], dim=1)
        # print(f"Head input shapes - conv1: {conv1.shape}, UpConv4: {UpConv4.shape}, convl: {convl.shape}")
        conv = self.psp_2ndlast(convl)
        # print(f"After PSP pooling: {conv.shape}")
        conv = F.relu(conv)
    
        # logits 
        # 1st find distance map, skeleton like, topology info
        dist = self.distance_logits(convl)  # do not use max pooling for distance
        # dist = self.ChannelAct(dist)
        dist = self.DistanceAct(dist)
        distEq = F.relu(self.dist_Equalizer(dist))  # makes nfilters equals to conv and convl  

        # Then find boundaries 
        bound = torch.cat([conv, distEq], dim=1)
        bound = self.bound_logits(bound)
        bound = self.CrispSigm(bound)  # Boundaries are not mutually exclusive 
        boundEq = F.relu(self.bound_Equalizer(bound))

        # Now combine all predictions in a final segmentation mask 
        # Balance first boundary and distance transform, with the features
        comb_bd = self.Comb_bound_dist(torch.cat([boundEq, distEq], dim=1))
        comb_bd = F.relu(comb_bd)

        all_layers = torch.cat([comb_bd, conv], dim=1)
        final_segm = self.final_segm_logits(all_layers)
        final_segm = self.ChannelAct(final_segm)

        return final_segm, bound, dist
