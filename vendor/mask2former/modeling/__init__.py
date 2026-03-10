# Copyright (c) Facebook, Inc. and its affiliates.
# Adapted for AutoFocusFormer by Ziwen 2023

# from .backbone.aff import AutoFocusFormer
from .backbone.swin import D2SwinTransformer, Mask2FormerD2SwinTransformer
from .pixel_decoder.fpn import BasePixelDecoder
# from .pixel_decoder.msdeformattn_pc import MSDeformAttnPixelDecoder # AFF
from .pixel_decoder.msdeformattn import MSDeformAttnPixelDecoder # Swin
from .meta_arch.mask_former_head import MaskFormerHead
from .meta_arch.per_pixel_baseline import PerPixelBaselineHead, PerPixelBaselinePlusHead