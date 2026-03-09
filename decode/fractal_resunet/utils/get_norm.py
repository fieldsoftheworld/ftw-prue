import torch.nn as nn


def get_norm(name, axis=1, norm_groups=None):
    """Get normalization layer based on name"""
    if name == "BatchNorm":
        return nn.BatchNorm2d if axis == 1 else nn.BatchNorm1d
    elif name == "InstanceNorm":
        return nn.InstanceNorm2d if axis == 1 else nn.InstanceNorm1d
    elif name == "LayerNorm":
        return nn.LayerNorm
    elif name == "GroupNorm" and norm_groups is not None:
        return lambda channels: nn.GroupNorm(num_groups=norm_groups, num_channels=channels)
    else:
        raise NotImplementedError(f"Normalization {name} not implemented")
