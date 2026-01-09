import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers.conv2Dnormed import Conv2DNormed


class PSP_Pooling(nn.Module):
    """Pyramid Scene Parsing Pooling"""
    
    def __init__(self, nfilters, depth=4, norm_type='BatchNorm', norm_groups=None, mob=False, **kwargs):
        super().__init__()
               
        self.nfilters = nfilters
        self.depth = depth 
        
        self.convs = nn.ModuleList()
        for _ in range(depth):
            self.convs.append(Conv2DNormed(
                self.nfilters, 
                kernel_size=(1, 1), 
                padding=(0, 0), 
                norm_type=norm_type, 
                norm_groups=norm_groups
            )) 
        
        self.conv_norm_final = Conv2DNormed(
            channels=self.nfilters,
            kernel_size=(1, 1),
            padding=(0, 0),
            norm_type=norm_type, 
            norm_groups=norm_groups,
            in_channels=self.nfilters * (depth + 1),
            out_channels=self.nfilters
        )

    def HalfSplit(self, a):
        """
        Returns a list of half split arrays. Useful for HalfPooling 
        """
        b = torch.split(a, a.shape[2] // 2, dim=2) 
        c1 = torch.split(b[0], b[0].shape[3] // 2, dim=3)
        c2 = torch.split(b[1], b[1].shape[3] // 2, dim=3)
    
        d11 = c1[0]
        d12 = c1[1]
        d21 = c2[0]
        d22 = c2[1]
    
        return [d11, d12, d21, d22]
    
    def QuarterStitch(self, Dss):
        """
        INPUT:
            A list of [d11,d12,d21,d22] block matrices.
        OUTPUT:
            A single matrix joined of these submatrices
        """
        temp1 = torch.cat([Dss[0], Dss[1]], dim=-1)
        temp2 = torch.cat([Dss[2], Dss[3]], dim=-1)
        result = torch.cat([temp1, temp2], dim=2)
        return result
    
    def HalfPooling(self, a):
        Ds = self.HalfSplit(a)
    
        Dss = []
        for x in Ds:
            Dss.append(torch.ones_like(x) * F.adaptive_avg_pool2d(x, (1, 1)))
     
        return self.QuarterStitch(Dss)    
      
    def SplitPooling(self, a, depth):
        """
        A recursive function that produces the Pooling you want - in particular depth (powers of 2)
        """
        if depth == 1:
            return self.HalfPooling(a)
        else:
            D = self.HalfSplit(a)
            return self.QuarterStitch([self.SplitPooling(d, depth - 1) for d in D])

    def forward(self, input):
        p = [input]
        p.append(self.convs[0](torch.ones_like(input) * F.adaptive_avg_pool2d(input, (1, 1))))
        p.extend([self.convs[d](self.SplitPooling(input, d)) for d in range(1, self.depth)])
        out = torch.cat(p, dim=1)
        out = self.conv_norm_final(out)
        return out
