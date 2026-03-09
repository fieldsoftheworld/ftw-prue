import cv2
import itertools
import numpy as np
import torch
import torch.nn.functional as F


class ParamsRange(dict):
    """Parameter ranges for data augmentation"""

    def __init__(self):
        self["center_range"] = [0, 256]
        self["rot_range"] = [-85.0, 85.0]
        self["zoom_range"] = [0.75, 1.25]
        self["noise_mean"] = [0] * 5
        self["noise_var"] = [10] * 5


class SemSegAugmentor_CV:
    """
    INPUTS:
        parameters range for all transformations
        probability of transformation to take place - default to 1.
        Nrot: number of rotations in comparison with reflections x,y,xy. Default to equal the number of reflections.
    """

    def __init__(self, params_range, prob=1.0, Nrot=5, norm=None, one_hot=True):
        self.norm = norm
        self.one_hot = one_hot
        self.range = params_range
        self.prob = prob
        assert self.prob <= 1, f"prob must be in range [0,1], you gave prob::{prob}"

        self.operations = [self.reflect_x, self.reflect_y, self.reflect_xy, self.random_brightness, self.random_shadow]
        self.operations += [self.rand_shift_rot_zoom] * Nrot
        self.iterator = itertools.cycle(self.operations)

    def _shift_rot_zoom(self, img, mask, center, angle, scale):
        imgT = img.transpose([1, 2, 0])
        if self.one_hot:
            maskT = mask.transpose([1, 2, 0])
        else:
            maskT = mask

        cols, rows = imgT.shape[:-1]

        tRotMat = cv2.getRotationMatrix2D(center, angle, scale)

        img_trans = cv2.warpAffine(imgT, tRotMat, (cols, rows), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REFLECT_101)
        mask_trans = cv2.warpAffine(
            maskT, tRotMat, (cols, rows), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REFLECT_101
        )

        img_trans = img_trans.transpose([2, 0, 1])
        if self.one_hot:
            mask_trans = mask_trans.transpose([2, 0, 1])

        return img_trans, mask_trans

    def reflect_x(self, img, mask):
        img_z = img[:, ::-1, :]
        if self.one_hot:
            mask_z = mask[:, ::-1, :]
        else:
            mask_z = mask[::-1, :]
        return img_z, mask_z

    def reflect_y(self, img, mask):
        img_z = img[:, :, ::-1]
        if self.one_hot:
            mask_z = mask[:, :, ::-1]
        else:
            mask_z = mask[:, ::-1]
        return img_z, mask_z

    def reflect_xy(self, img, mask):
        img_z = img[:, ::-1, ::-1]
        if self.one_hot:
            mask_z = mask[:, ::-1, ::-1]
        else:
            mask_z = mask[::-1, ::-1]
        return img_z, mask_z

    def rand_shift_rot_zoom(self, img, mask):
        center = np.random.randint(low=self.range["center_range"][0], high=self.range["center_range"][1], size=2)

        angle = np.random.uniform(low=self.range["rot_range"][0], high=self.range["rot_range"][1])
        scale = np.random.uniform(low=self.range["zoom_range"][0], high=self.range["zoom_range"][1])

        return self._shift_rot_zoom(img, mask, center, angle, scale)

    def random_brightness(self, img, mask):
        brightness_factor = np.random.uniform(0.8, 1.2)
        img_bright = img * brightness_factor
        img_bright = np.clip(img_bright, 0, 255)
        return img_bright, mask

    def random_shadow(self, img, mask):
        shadow_intensity = np.random.uniform(0.3, 0.7)
        img_shadow = img * shadow_intensity
        return img_shadow, mask

    def augment(self, img, mask):
        if np.random.random() < self.prob:
            op = next(self.iterator)
            return op(img, mask)
        return img, mask
