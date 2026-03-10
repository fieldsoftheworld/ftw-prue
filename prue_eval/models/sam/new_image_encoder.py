import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple, Type

from segment_anything.modeling.common import LayerNorm2d, MLPBlock
from segment_anything.modeling.image_encoder import ImageEncoderViT

class newImageEncoderViT(ImageEncoderViT):
    def __init__(
        self,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args,**kwargs)
        self.patch_size = kwargs['patch_size']
        self.in_chans = kwargs['in_chans']
        self.embed_dim = kwargs['embed_dim']

    def ckpt_blk(self, blk):
        def custom_forward(x):
            x = blk(x)
            return x
        return custom_forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed

        for blk in self.blocks:
            x = torch.utils.checkpoint.checkpoint( #blk(x)
                self.ckpt_blk(blk),
                x,
                use_reentrant=False,
            )

        x = self.neck(x.permute(0, 3, 1, 2))

        return x
