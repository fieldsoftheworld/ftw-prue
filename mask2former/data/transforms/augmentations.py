# Custom augmentations for satellite imagery
import numpy as np
import torch
from detectron2.data import transforms as T
from detectron2.data.transforms import TransformGen, ResizeShortestEdge
import random
from scipy.ndimage import uniform_filter
import cv2

class TransformValidator:
    """Validates transforms and their outputs."""
    
    @staticmethod
    def is_valid_image(image, min_nonzero_fraction=0.001, min_unique_values=5):
        """
        Check if an image is valid based on criteria appropriate for satellite imagery.
        Relaxed thresholds to account for valid dark/uniform areas.
        """
        if torch.is_tensor(image):
            image_array = image.cpu().numpy()
        else:
            image_array = image
            
        # For multi-channel images, check each channel
        if len(image_array.shape) <= 8:
            # Check if any channel is valid
            return any(TransformValidator.is_valid_image(image_array[..., i])[0] 
                      for i in range(image_array.shape[-1]))
            
        # Check for NaN or inf values
        if np.any(np.isnan(image_array)) or np.any(np.isinf(image_array)):
            return False, "Image contains NaN or inf values"
            
        # More permissive check for zero values
        nonzero_fraction = np.count_nonzero(image_array) / image_array.size
        if nonzero_fraction < min_nonzero_fraction:
            return False, f"Image has too few non-zero values ({nonzero_fraction:.3f})"
            
        # Check for sufficient unique values, but be more permissive
        unique_values = len(np.unique(image_array))
        if unique_values < min_unique_values:
            return False, f"Image has too few unique values ({unique_values})"
            
        return True, "Image is valid"

class SafeTransformMixin:
    """Enhanced mixin class to track transform effects."""

    def apply_transform_safely(self, transform, image, max_attempts=3):
        """Apply transform with multiple attempts if needed."""
        validator = TransformValidator()
        
        try:
            # Apply the transform
            transformed = transform.apply_image(image)

            # Validate the result
            is_valid, message = validator.is_valid_image(transformed)
            if is_valid:
                return transformed
            else:
                print(f"Transform validation failed: {message}")
                return image
                
        except Exception as e:
            print(f"Transform failed with error: {str(e)}")
            return image

class SafeRandomRotation(T.Augmentation, SafeTransformMixin):
    def __init__(self, angle_list, **kwargs):
        super().__init__()
        self.angle_list = angle_list
        self.kwargs = kwargs
        
    def get_transform(self, image):
        angle = random.choice(self.angle_list)
        return T.RotationTransform(h=image.shape[0], w=image.shape[1], angle=angle, **self.kwargs)


# ============================================================================
# Translation and Shear Transforms (for SatTrivial)
# ============================================================================

class SafeTranslateX(T.Augmentation, SafeTransformMixin):
    """Safe translation in X direction that handles segmentation properly."""
    
    def __init__(self, max_translate_ratio=0.1, **kwargs):
        super().__init__()
        self.max_translate_ratio = max_translate_ratio
        self.kwargs = kwargs
        
    def get_transform(self, image):
        h, w = image.shape[:2]
        max_translate = int(self.max_translate_ratio * w)
        translate_x = np.random.randint(-max_translate, max_translate + 1)
        return TranslateXTransform(h=h, w=w, translate_x=translate_x, **self.kwargs)


class SafeTranslateY(T.Augmentation, SafeTransformMixin):
    """Safe translation in Y direction that handles segmentation properly."""
    
    def __init__(self, max_translate_ratio=0.1, **kwargs):
        super().__init__()
        self.max_translate_ratio = max_translate_ratio
        self.kwargs = kwargs
        
    def get_transform(self, image):
        h, w = image.shape[:2]
        max_translate = int(self.max_translate_ratio * h)
        translate_y = np.random.randint(-max_translate, max_translate + 1)
        return TranslateYTransform(h=h, w=w, translate_y=translate_y, **self.kwargs)


class SafeShearX(T.Augmentation, SafeTransformMixin):
    """Safe shear in X direction that handles segmentation properly."""
    
    def __init__(self, max_shear=0.3, **kwargs):
        super().__init__()
        self.max_shear = max_shear
        self.kwargs = kwargs
        
    def get_transform(self, image):
        h, w = image.shape[:2]
        shear_x = np.random.uniform(-self.max_shear, self.max_shear)
        return ShearXTransform(h=h, w=w, shear_x=shear_x, **self.kwargs)


class SafeShearY(T.Augmentation, SafeTransformMixin):
    """Safe shear in Y direction that handles segmentation properly."""
    
    def __init__(self, max_shear=0.3, **kwargs):
        super().__init__()
        self.max_shear = max_shear
        self.kwargs = kwargs
        
    def get_transform(self, image):
        h, w = image.shape[:2]
        shear_y = np.random.uniform(-self.max_shear, self.max_shear)
        return ShearYTransform(h=h, w=w, shear_y=shear_y, **self.kwargs)


class TranslateXTransform(T.Transform):
    """Transform for X-direction translation."""
    
    def __init__(self, h, w, translate_x, **kwargs):
        super().__init__()
        self.h = h
        self.w = w
        self.translate_x = translate_x
        
    def apply_image(self, img):
        if self.translate_x == 0:
            return img
        M = np.float32([[1, 0, self.translate_x], [0, 1, 0]])
        return cv2.warpAffine(img, M, (self.w, self.h), borderMode=cv2.BORDER_REFLECT)
    
    def apply_coords(self, coords):
        if self.translate_x == 0:
            return coords
        coords[:, 0] += self.translate_x
        return coords
    
    def apply_segmentation(self, segmentation):
        if self.translate_x == 0:
            return segmentation
        M = np.float32([[1, 0, self.translate_x], [0, 1, 0]])
        return cv2.warpAffine(segmentation, M, (self.w, self.h), borderMode=cv2.BORDER_REFLECT, flags=cv2.INTER_NEAREST)


class TranslateYTransform(T.Transform):
    """Transform for Y-direction translation."""
    
    def __init__(self, h, w, translate_y, **kwargs):
        super().__init__()
        self.h = h
        self.w = w
        self.translate_y = translate_y
        
    def apply_image(self, img):
        if self.translate_y == 0:
            return img
        M = np.float32([[1, 0, 0], [0, 1, self.translate_y]])
        return cv2.warpAffine(img, M, (self.w, self.h), borderMode=cv2.BORDER_REFLECT)
    
    def apply_coords(self, coords):
        if self.translate_y == 0:
            return coords
        coords[:, 1] += self.translate_y
        return coords
    
    def apply_segmentation(self, segmentation):
        if self.translate_y == 0:
            return segmentation
        M = np.float32([[1, 0, 0], [0, 1, self.translate_y]])
        return cv2.warpAffine(segmentation, M, (self.w, self.h), borderMode=cv2.BORDER_REFLECT, flags=cv2.INTER_NEAREST)


class ShearXTransform(T.Transform):
    """Transform for X-direction shear."""
    
    def __init__(self, h, w, shear_x, **kwargs):
        super().__init__()
        self.h = h
        self.w = w
        self.shear_x = shear_x
        
    def apply_image(self, img):
        if abs(self.shear_x) < 0.01:
            return img
        M = np.float32([[1, self.shear_x, 0], [0, 1, 0]])
        return cv2.warpAffine(img, M, (self.w, self.h), borderMode=cv2.BORDER_REFLECT)
    
    def apply_coords(self, coords):
        if abs(self.shear_x) < 0.01:
            return coords
        coords[:, 0] += self.shear_x * coords[:, 1]
        return coords
    
    def apply_segmentation(self, segmentation):
        if abs(self.shear_x) < 0.01:
            return segmentation
        M = np.float32([[1, self.shear_x, 0], [0, 1, 0]])
        return cv2.warpAffine(segmentation, M, (self.w, self.h), borderMode=cv2.BORDER_REFLECT, flags=cv2.INTER_NEAREST)


class ShearYTransform(T.Transform):
    """Transform for Y-direction shear."""
    
    def __init__(self, h, w, shear_y, **kwargs):
        super().__init__()
        self.h = h
        self.w = w
        self.shear_y = shear_y
        
    def apply_image(self, img):
        if abs(self.shear_y) < 0.01:
            return img
        M = np.float32([[1, 0, 0], [self.shear_y, 1, 0]])
        return cv2.warpAffine(img, M, (self.w, self.h), borderMode=cv2.BORDER_REFLECT)
    
    def apply_coords(self, coords):
        if abs(self.shear_y) < 0.01:
            return coords
        coords[:, 1] += self.shear_y * coords[:, 0]
        return coords
    
    def apply_segmentation(self, segmentation):
        if abs(self.shear_y) < 0.01:
            return segmentation
        M = np.float32([[1, 0, 0], [self.shear_y, 1, 0]])
        return cv2.warpAffine(segmentation, M, (self.w, self.h), borderMode=cv2.BORDER_REFLECT, flags=cv2.INTER_NEAREST)
    
class MissingValueImputation(T.Augmentation):
    """Transform that handles missing data imputation for satellite imagery."""
    
    def __init__(self, missing_value=None, strategy='mean', window_size=5):
        """
        Args:
            missing_value: Value to consider as missing (e.g., 0, NaN)
            strategy: Imputation strategy ('mean', 'median', 'neighbor')
            window_size: Size of window for neighbor-based imputation
        """
        super().__init__()
        self.missing_value = missing_value
        self.strategy = strategy
        self.window_size = window_size
        
    def get_transform(self, image):
        return MissingValueImputationTransform(
            missing_value=self.missing_value,
            strategy=self.strategy,
            window_size=self.window_size
        )

class MissingValueImputationTransform(T.Transform):
    def __init__(self, missing_value=None, strategy='neighbor', window_size=5):
        super().__init__()
        self.missing_value = missing_value
        self.strategy = strategy
        self.window_size = window_size
        
    def _get_neighbor_mean(self, img, mask, window_size):
        """Calculate mean of non-missing neighbors within window."""
        from scipy.ndimage import uniform_filter
        
        # Create weight matrix for valid pixels
        valid_pixels = ~mask
        
        # Calculate sum and count of valid pixels in neighborhood
        weights = uniform_filter(valid_pixels.astype(float), size=window_size)
        weighted_sum = uniform_filter(
            np.where(valid_pixels, img, 0).astype(float), 
            size=window_size
        )
        
        # Avoid division by zero
        neighbor_mean = np.where(
            weights > 0,
            weighted_sum / weights,
            img  # Keep original value where no valid neighbors
        )
        
        return neighbor_mean
        
    def apply_image(self, img):
        """
        Apply imputation to image.
        Args:
            img: numpy array of shape (H,W,C)
        Returns:
            imputed image of same shape
        """
        result = img.copy()
        
        # Handle different missing value indicators
        if self.missing_value is None:
            mask = np.isnan(img)
        else:
            mask = img == self.missing_value
            
        # Process each channel separately
        for c in range(img.shape[2]):
            channel_mask = mask[..., c]
            if not np.any(channel_mask):
                continue
                
            if self.strategy == 'mean':
                # Calculate mean of non-missing values per channel
                channel_mean = np.mean(img[~channel_mask, c])
                result[channel_mask, c] = channel_mean
                
            elif self.strategy == 'median':
                channel_median = np.median(img[~channel_mask, c])
                result[channel_mask, c] = channel_median
                
            elif self.strategy == 'neighbor':
                result[..., c] = self._get_neighbor_mean(
                    img[..., c],
                    channel_mask,
                    self.window_size
                )
                
        return result
    
    def apply_coords(self, coords):
        return coords

    def apply_segmentation(self, segmentation):
        """
        Apply no transformation to segmentation.
        """
        return segmentation

class RandomErase(T.Augmentation):
    def __init__(self, max_erased_groups=9, erase_size=(10, 20), prob=0.1):
        super().__init__()
        self.max_erased_groups = max_erased_groups
        self.erase_size = erase_size
        self.prob = prob
        
    def get_transform(self, image):
        n_groups = np.random.randint(0, self.max_erased_groups + 1)
        return EraseTransform(n_groups, self.erase_size, self.prob)

class EraseTransform(T.Transform):
    def __init__(self, n_groups, erase_size, prob):
        super().__init__()
        self.n_groups = n_groups
        self.erase_size = erase_size
        self.prob = prob
    
    def apply_image(self, img):
        if np.random.random() > self.prob:
            return img

        result = img.copy()
        h, w = img.shape[:2]
        
        for _ in range(self.n_groups):
            erase_h = np.random.randint(self.erase_size[0], self.erase_size[1])
            erase_w = np.random.randint(self.erase_size[0], self.erase_size[1])
            
            x = np.random.randint(0, w - erase_w)
            y = np.random.randint(0, h - erase_h)
            
            result[y:y+erase_h, x:x+erase_w] = 0
            
        return result

    
    def apply_coords(self, coords):
        return coords

    def apply_segmentation(self, segmentation):
        """
        Apply no transformation to segmentation.
        """
        return segmentation

class RandomOversaturate(T.Augmentation):
    def __init__(self, prob=0.1, max_regions=3, max_value=65535):  # 2^16 - 1
        super().__init__()
        self.prob = prob
        self.max_regions = max_regions
        self.max_value = max_value
    
    def get_transform(self, image):
        return OversaturateTransform(self.prob, self.max_regions, self.max_value)

class OversaturateTransform(T.Transform):
    def __init__(self, prob, max_regions, max_value):
        super().__init__()
        self.prob = prob
        self.max_regions = max_regions
        self.max_value = max_value
    
    def apply_image(self, img):
        if np.random.random() > self.prob:
            return img
            
        result = img.copy()
        h, w = img.shape[:2]
        n_regions = np.random.randint(1, self.max_regions + 1)
        
        for _ in range(n_regions):
            size = np.random.randint(20, 50)
            x = np.random.randint(0, w - size)
            y = np.random.randint(0, h - size)
            result[y:y+size, x:x+size] = self.max_value
            
        return result

    def apply_coords(self, coords):
        return coords

    def apply_segmentation(self, segmentation):
        """
        Apply no transformation to segmentation.
        """
        return segmentation

class GaussianNoise(T.Augmentation):
    def __init__(self, max_std=0.04, prob=0.1):
        super().__init__()
        self.max_std = max_std
        self.prob = prob
        
    def get_transform(self, image):
        # Scale std based on image dynamic range
        img_max = np.max(image)
        scaled_std = self.max_std * img_max
        return NoiseTransform(scaled_std, self.prob)

class NoiseTransform(T.Transform):
    def __init__(self, std, prob):
        super().__init__()
        self.std = std
        self.prob = prob
    
    def apply_image(self, img):
        if np.random.random() > self.prob:
            return img

        noise = np.random.normal(0, self.std, img.shape)
        return np.clip(img + noise, 0, np.max(img))        

    def apply_coords(self, coords):
        return coords

    def apply_segmentation(self, segmentation):
        """
        Apply no transformation to segmentation.
        """
        return segmentation

def is_geometric_transform(transform):
    """
    Determine if a transform is geometric (affects spatial coordinates).
    
    Geometric transforms need to be applied to both image and segmentation labels.
    Radiometric transforms only affect pixel values and should NOT be applied to labels.
    
    Args:
        transform: Transform augmentation object
        
    Returns:
        bool: True if geometric, False if radiometric or other
    """
    # Geometric transforms that affect spatial coordinates
    geometric_types = (
        T.RandomFlip,
        T.RotationTransform,
        T.RandomCrop,
        T.ResizeScale,
        T.FixedSizeCrop,
        T.ResizeShortestEdge,
        T.RandomApply,
        T.RandomRotation,
        SafeRandomRotation,
        SafeTranslateX,
        SafeTranslateY,
        SafeShearX,
        SafeShearY,
        SatTrivialAugmentation,
        RandomResizedCrop,
    )
    
    # Check if transform is an instance of geometric types
    if isinstance(transform, geometric_types):
        return True
    
    # Check by class name as fallback (for transforms wrapped in other classes)
    transform_class_name = transform.__class__.__name__
    geometric_class_names = (
        'RandomFlip',
        'RotationTransform',
        'RandomCrop',
        'ResizeScale',
        'FixedSizeCrop',
        'ResizeShortestEdge',
        'RandomApply',
        'RandomRotation',
        'SafeRandomRotation',
        'TranslateXTransform',
        'TranslateYTransform',
        'ShearXTransform',
        'ShearYTransform',
        'SatTrivialTransform',
        'SatTrivialAugmentation',
        'RandomResizedCrop',
        'RandomResizedCropTransform',
    )
    
    if any(name in transform_class_name for name in geometric_class_names):
        return True
    
    # Default: assume non-geometric (radiometric or utility)
    return False


# ============================================================================
# Additional Transforms for FTW/Prue Augmentation Sets
# ============================================================================

class RandomSharpness(T.Augmentation):
    """Random sharpness augmentation for satellite imagery."""
    
    def __init__(self, prob=0.5, sharpness_range=(0.5, 2.0)):
        """
        Args:
            prob: Probability of applying sharpness
            sharpness_range: Tuple of (min, max) sharpness values
        """
        super().__init__()
        self.prob = prob
        self.sharpness_range = sharpness_range
        
    def get_transform(self, image):
        return SharpnessTransform(self.prob, self.sharpness_range)


class SharpnessTransform(T.Transform):
    """Transform for sharpness adjustment."""
    
    def __init__(self, prob, sharpness_range):
        super().__init__()
        self.prob = prob
        self.sharpness_range = sharpness_range
        
    def apply_image(self, img):
        if np.random.random() > self.prob:
            return img
        
        from PIL import Image, ImageEnhance
        # Convert numpy array to PIL Image
        if img.dtype != np.uint8:
            # Normalize to 0-255 for PIL
            img_normalized = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)
        else:
            img_normalized = img.copy()
        
        # Apply sharpness to each channel separately for multi-channel images
        if len(img_normalized.shape) == 3:
            result = np.zeros_like(img_normalized)
            for c in range(img_normalized.shape[2]):
                channel_img = Image.fromarray(img_normalized[:, :, c])
                enhancer = ImageEnhance.Sharpness(channel_img)
                factor = np.random.uniform(self.sharpness_range[0], self.sharpness_range[1])
                enhanced = enhancer.enhance(factor)
                result[:, :, c] = np.array(enhanced)
            
            # Convert back to original dtype and scale
            if img.dtype != np.uint8:
                result = (result.astype(np.float32) / 255.0 * (img.max() - img.min()) + img.min()).astype(img.dtype)
            
            return result
        else:
            # Single channel
            channel_img = Image.fromarray(img_normalized)
            enhancer = ImageEnhance.Sharpness(channel_img)
            factor = np.random.uniform(self.sharpness_range[0], self.sharpness_range[1])
            enhanced = enhancer.enhance(factor)
            result = np.array(enhanced)
            
            if img.dtype != np.uint8:
                result = (result.astype(np.float32) / 255.0 * (img.max() - img.min()) + img.min()).astype(img.dtype)
            
            return result
    
    def apply_coords(self, coords):
        return coords
    
    def apply_segmentation(self, segmentation):
        # Sharpness doesn't affect segmentation
        return segmentation


class ChannelShuffle(T.Augmentation):
    """Shuffle channels for stacked multi-temporal images (e.g., swap window A and B)."""
    
    def __init__(self, prob=0.5, num_channels=8):
        """
        Args:
            prob: Probability of shuffling
            num_channels: Total number of channels (should be 8 for stacked images)
        """
        super().__init__()
        self.prob = prob
        self.num_channels = num_channels
        
    def get_transform(self, image):
        return ChannelShuffleTransform(self.prob, self.num_channels)


class ChannelShuffleTransform(T.Transform):
    """Transform for channel shuffling."""
    
    def __init__(self, prob, num_channels):
        super().__init__()
        self.prob = prob
        self.num_channels = num_channels
        
    def apply_image(self, img):
        if np.random.random() > self.prob:
            return img
        
        # For 8-channel stacked images: swap first 4 channels with last 4 channels
        if len(img.shape) == 3 and img.shape[2] == self.num_channels:
            result = img.copy()
            # Swap window A and window B
            result[:, :, :4], result[:, :, 4:] = result[:, :, 4:].copy(), result[:, :, :4].copy()
            return result
        
        return img
    
    def apply_coords(self, coords):
        return coords
    
    def apply_segmentation(self, segmentation):
        # Channel shuffle doesn't affect segmentation
        return segmentation


class RandomResizedCrop(T.Augmentation):
    """Random resized crop for satellite imagery."""
    
    def __init__(self, prob=0.5, size=(256, 256), scale=(0.3, 0.9), ratio=(0.75, 1.33)):
        """
        Args:
            prob: Probability of applying crop
            size: Output size (height, width)
            scale: Scale range (min, max)
            ratio: Aspect ratio range (min, max)
        """
        super().__init__()
        self.prob = prob
        self.size = size
        self.scale = scale
        self.ratio = ratio
        
    def get_transform(self, image):
        return RandomResizedCropTransform(self.prob, self.size, self.scale, self.ratio)


class RandomResizedCropTransform(T.Transform):
    """Transform for random resized crop."""
    
    def __init__(self, prob, size, scale, ratio):
        super().__init__()
        self.prob = prob
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self._crop_params = None
        
    def _get_crop_params(self, img_h, img_w):
        """Calculate crop parameters."""
        if self._crop_params is not None:
            return self._crop_params
        
        if np.random.random() > self.prob:
            # No crop - use full image
            self._crop_params = (0, 0, img_w, img_h, self.size[1], self.size[0])
            return self._crop_params
        
        # Calculate scale and aspect ratio
        area = img_h * img_w
        target_area = np.random.uniform(self.scale[0], self.scale[1]) * area
        aspect_ratio = np.random.uniform(self.ratio[0], self.ratio[1])
        
        # Calculate crop dimensions
        w = int(np.round(np.sqrt(target_area * aspect_ratio)))
        h = int(np.round(np.sqrt(target_area / aspect_ratio)))
        
        # Ensure crop fits within image
        w = min(w, img_w)
        h = min(h, img_h)
        
        # Random crop position
        if img_w > w:
            x = np.random.randint(0, img_w - w)
        else:
            x = 0
            
        if img_h > h:
            y = np.random.randint(0, img_h - h)
        else:
            y = 0
        
        self._crop_params = (x, y, w, h, self.size[1], self.size[0])
        return self._crop_params
    
    def apply_image(self, img):
        img_h, img_w = img.shape[:2]
        x, y, w, h, out_w, out_h = self._get_crop_params(img_h, img_w)
        
        # Crop
        cropped = img[y:y+h, x:x+w]
        
        # Resize to target size
        resized = cv2.resize(cropped, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        return resized
    
    def apply_coords(self, coords):
        if self._crop_params is None:
            return coords
        
        x, y, w, h, out_w, out_h = self._crop_params
        img_h, img_w = out_h, out_w  # Use output size for scaling
        
        # Translate coordinates by crop offset
        coords[:, 0] -= x
        coords[:, 1] -= y
        
        # Scale coordinates to new size
        scale_x = out_w / w
        scale_y = out_h / h
        coords[:, 0] *= scale_x
        coords[:, 1] *= scale_y
        
        return coords
    
    def apply_segmentation(self, segmentation):
        if self._crop_params is None:
            return segmentation
        
        x, y, w, h, out_w, out_h = self._get_crop_params(segmentation.shape[0], segmentation.shape[1])
        
        # Crop
        cropped = segmentation[y:y+h, x:x+w]
        
        # Resize to target size (use nearest neighbor for segmentation)
        resized = cv2.resize(cropped, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
        return resized


# ============================================================================
# SatTrivial Augmentation (Single Random Augmentation Per Image)
# ============================================================================

class SatTrivialAugmentation(T.Augmentation):
    """Sat-Trivial style augmentation that randomly selects ONE augmentation per image."""
    
    def __init__(self, prob=0.8, max_std=0.04, erase_prob=0.05, saturate_prob=0.01, noise_prob=0.05):
        super().__init__()
        self.prob = prob
        self.max_std = max_std
        self.erase_prob = erase_prob
        self.saturate_prob = saturate_prob
        self.noise_prob = noise_prob
        
    def get_transform(self, image):
        if np.random.random() > self.prob:
            return SatTrivialTransform('identity', 0, 0, 0, 0)
        
        # Randomly select one augmentation (Sat-Trivial style)
        augmentations = ['identity', 'translateX', 'translateY', 'shearX', 'shearY', 
                        'flip', 'rotate', 'erase', 'saturate', 'noise']
        selected = np.random.choice(augmentations)
        
        return SatTrivialTransform(selected, self.max_std, self.erase_prob, 
                                 self.saturate_prob, self.noise_prob)


class SatTrivialTransform(T.Transform, SafeTransformMixin):
    """Single Sat-Trivial augmentation transform with proper segmentation handling."""
    
    def __init__(self, augmentation_type, max_std, erase_prob, saturate_prob, noise_prob):
        super().__init__()
        self.augmentation_type = augmentation_type
        self.max_std = max_std
        self.erase_prob = erase_prob
        self.saturate_prob = saturate_prob
        self.noise_prob = noise_prob
        
        # Store transform parameters to ensure consistency between image and segmentation
        self._transform_params = {}
        self._initialize_transform_params()
    
    def _initialize_transform_params(self):
        """Initialize transform parameters once to ensure consistency."""
        if self.augmentation_type == 'translateX':
            h, w = 256, 256  # Default size, will be updated in apply methods
            max_translate = int(0.1 * w)
            self._transform_params['translate_x'] = np.random.randint(-max_translate, max_translate + 1)
        elif self.augmentation_type == 'translateY':
            h, w = 256, 256  # Default size, will be updated in apply methods
            max_translate = int(0.1 * h)
            self._transform_params['translate_y'] = np.random.randint(-max_translate, max_translate + 1)
        elif self.augmentation_type == 'shearX':
            self._transform_params['shear_x'] = np.random.uniform(-0.3, 0.3)
        elif self.augmentation_type == 'shearY':
            self._transform_params['shear_y'] = np.random.uniform(-0.3, 0.3)
        elif self.augmentation_type == 'flip':
            self._transform_params['flip_h'] = np.random.random() < 0.5
            self._transform_params['flip_v'] = np.random.random() < 0.5
        elif self.augmentation_type == 'rotate':
            self._transform_params['rotate_k'] = np.random.choice([0, 1, 2, 3])
        elif self.augmentation_type == 'noise':
            self._transform_params['noise_std'] = np.random.uniform(0, self.max_std)
        elif self.augmentation_type == 'erase':
            self._transform_params['erase_applied'] = np.random.random() <= self.erase_prob
        elif self.augmentation_type == 'saturate':
            self._transform_params['saturate_applied'] = np.random.random() <= self.saturate_prob
        
    def apply_image(self, img):
        if self.augmentation_type == 'identity':
            return img
        elif self.augmentation_type == 'translateX':
            return self._translate_x(img)
        elif self.augmentation_type == 'translateY':
            return self._translate_y(img)
        elif self.augmentation_type == 'shearX':
            return self._shear_x(img)
        elif self.augmentation_type == 'shearY':
            return self._shear_y(img)
        elif self.augmentation_type == 'flip':
            return self._random_flip(img)
        elif self.augmentation_type == 'rotate':
            return self._random_rotate(img)
        elif self.augmentation_type == 'erase':
            return self._erase(img)
        elif self.augmentation_type == 'saturate':
            return self._saturate(img)
        elif self.augmentation_type == 'noise':
            return self._gaussian_noise(img)
        else:
            return img
    
    def apply_segmentation(self, segmentation):
        """Apply the same transform to segmentation with proper handling."""
        if self.augmentation_type == 'identity':
            return segmentation
        elif self.augmentation_type == 'translateX':
            return self._translate_x_seg(segmentation)
        elif self.augmentation_type == 'translateY':
            return self._translate_y_seg(segmentation)
        elif self.augmentation_type == 'shearX':
            return self._shear_x_seg(segmentation)
        elif self.augmentation_type == 'shearY':
            return self._shear_y_seg(segmentation)
        elif self.augmentation_type == 'flip':
            return self._random_flip_seg(segmentation)
        elif self.augmentation_type == 'rotate':
            return self._random_rotate_seg(segmentation)
        elif self.augmentation_type in ['erase', 'saturate', 'noise']:
            # These don't affect segmentation
            return segmentation
        else:
            return segmentation
    
    def _translate_x(self, img):
        """Translate image in x-direction (up to 10% of width)."""
        h, w = img.shape[:2]
        # Update stored parameters with actual image dimensions
        max_translate = int(0.1 * w)
        if 'translate_x' not in self._transform_params:
            self._transform_params['translate_x'] = np.random.randint(-max_translate, max_translate + 1)
        
        translate_x = self._transform_params['translate_x']
        if translate_x == 0:
            return img
            
        # Create translation matrix
        M = np.float32([[1, 0, translate_x], [0, 1, 0]])
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    def _translate_x_seg(self, segmentation):
        """Translate segmentation in x-direction with proper interpolation."""
        h, w = segmentation.shape[:2]
        translate_x = self._transform_params.get('translate_x', 0)
        
        if translate_x == 0:
            return segmentation
            
        # Create translation matrix
        M = np.float32([[1, 0, translate_x], [0, 1, 0]])
        return cv2.warpAffine(segmentation, M, (w, h), borderMode=cv2.BORDER_REFLECT, flags=cv2.INTER_NEAREST)
    
    def _translate_y(self, img):
        """Translate image in y-direction (up to 10% of height)."""
        h, w = img.shape[:2]
        translate_y = self._transform_params.get('translate_y', 0)
        
        if translate_y == 0:
            return img
            
        # Create translation matrix
        M = np.float32([[1, 0, 0], [0, 1, translate_y]])
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    def _translate_y_seg(self, segmentation):
        """Translate segmentation in y-direction with proper interpolation."""
        h, w = segmentation.shape[:2]
        translate_y = self._transform_params.get('translate_y', 0)
        
        if translate_y == 0:
            return segmentation
            
        # Create translation matrix
        M = np.float32([[1, 0, 0], [0, 1, translate_y]])
        return cv2.warpAffine(segmentation, M, (w, h), borderMode=cv2.BORDER_REFLECT, flags=cv2.INTER_NEAREST)
    
    def _shear_x(self, img):
        """Shear image in x-direction."""
        h, w = img.shape[:2]
        shear_x = self._transform_params.get('shear_x', 0)
        
        if abs(shear_x) < 0.01:
            return img
            
        # Create shear matrix
        M = np.float32([[1, shear_x, 0], [0, 1, 0]])
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    def _shear_x_seg(self, segmentation):
        """Shear segmentation in x-direction with proper interpolation."""
        h, w = segmentation.shape[:2]
        shear_x = self._transform_params.get('shear_x', 0)
        
        if abs(shear_x) < 0.01:
            return segmentation
            
        # Create shear matrix
        M = np.float32([[1, shear_x, 0], [0, 1, 0]])
        return cv2.warpAffine(segmentation, M, (w, h), borderMode=cv2.BORDER_REFLECT, flags=cv2.INTER_NEAREST)
    
    def _shear_y(self, img):
        """Shear image in y-direction."""
        h, w = img.shape[:2]
        shear_y = self._transform_params.get('shear_y', 0)
        
        if abs(shear_y) < 0.01:
            return img
            
        # Create shear matrix
        M = np.float32([[1, 0, 0], [shear_y, 1, 0]])
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    def _shear_y_seg(self, segmentation):
        """Shear segmentation in y-direction with proper interpolation."""
        h, w = segmentation.shape[:2]
        shear_y = self._transform_params.get('shear_y', 0)
        
        if abs(shear_y) < 0.01:
            return segmentation
            
        # Create shear matrix
        M = np.float32([[1, 0, 0], [shear_y, 1, 0]])
        return cv2.warpAffine(segmentation, M, (w, h), borderMode=cv2.BORDER_REFLECT, flags=cv2.INTER_NEAREST)
    
    def _random_flip(self, img):
        """Random horizontal and vertical flips."""
        flip_h = self._transform_params.get('flip_h', False)
        flip_v = self._transform_params.get('flip_v', False)
        
        if flip_h:
            img = np.fliplr(img)
        if flip_v:
            img = np.flipud(img)
        return img
    
    def _random_flip_seg(self, segmentation):
        """Random horizontal and vertical flips for segmentation."""
        flip_h = self._transform_params.get('flip_h', False)
        flip_v = self._transform_params.get('flip_v', False)
        
        if flip_h:
            segmentation = np.fliplr(segmentation)
        if flip_v:
            segmentation = np.flipud(segmentation)
        return segmentation
    
    def _random_rotate(self, img):
        """Random rotation (0, 90, 180, 270 degrees)."""
        k = self._transform_params.get('rotate_k', 0)
        if k > 0:
            img = np.rot90(img, k=k)
        return img
    
    def _random_rotate_seg(self, segmentation):
        """Random rotation for segmentation."""
        k = self._transform_params.get('rotate_k', 0)
        if k > 0:
            segmentation = np.rot90(segmentation, k=k)
        return segmentation
    
    def _erase(self, img):
        """Random erase (0-9 patches set to 0)."""
        if np.random.random() > self.erase_prob:
            return img
            
        result = img.copy()
        h, w = img.shape[:2]
        num_patches = np.random.randint(1, 10)
        
        for _ in range(num_patches):
            # Random patch size and position
            patch_h = np.random.randint(1, min(20, h//4))
            patch_w = np.random.randint(1, min(20, w//4))
            x = np.random.randint(0, w - patch_w)
            y = np.random.randint(0, h - patch_h)
            
            result[y:y+patch_h, x:x+patch_w] = 0
            
        return result
    
    def _saturate(self, img):
        """Random saturate (0-9 patches set to max value)."""
        if np.random.random() > self.saturate_prob:
            return img
            
        result = img.copy()
        h, w = img.shape[:2]
        max_val = np.max(img)
        num_patches = np.random.randint(1, 10)
        
        for _ in range(num_patches):
            # Random patch size and position
            patch_h = np.random.randint(1, min(20, h//4))
            patch_w = np.random.randint(1, min(20, w//4))
            x = np.random.randint(0, w - patch_w)
            y = np.random.randint(0, h - patch_h)
            
            result[y:y+patch_h, x:x+patch_w] = max_val
            
        return result
    
    def _gaussian_noise(self, img):
        """Gaussian noise."""
        if np.random.random() > self.noise_prob:
            return img
            
        std = np.random.uniform(0, self.max_std)
        noise = np.random.normal(0, std, img.shape)
        return np.clip(img + noise, 0, np.max(img))
    
    def apply_coords(self, coords):
        return coords


# ============================================================================
# Augmentation Set Builders
# ============================================================================

def build_augmentation_set_sattrivial(cfg):
    """
    Build SatTrivial augmentation set.
    Single random augmentation per image: translateX, translateY, shearX, shearY,
    flip, rotate, erase, saturate, noise.
    """
    geometric = []
    radiometric = []
    
    # Basic geometric transforms that are always applied
    if cfg.INPUT.RANDOM_FLIP in ['both', 'horizontal']:
        geometric.append(T.RandomFlip(prob=0.5, horizontal=True, vertical=False))
    if cfg.INPUT.RANDOM_FLIP in ['both', 'vertical']:
        geometric.append(T.RandomFlip(prob=0.5, horizontal=False, vertical=True))
    
    geometric.append(SafeRandomRotation([90, 180, 270], expand=False, interp=cv2.INTER_LINEAR, center=None))
    
    if cfg.INPUT.CROP.ENABLED:
        geometric.append(T.RandomCrop(
            crop_type=cfg.INPUT.CROP.TYPE,
            crop_size=(cfg.INPUT.CROP.SIZE[0], cfg.INPUT.CROP.SIZE[1])
        ))
    
    # Sat-Trivial style: randomly sample ONE additional augmentation
    # This combines geometric and radiometric augmentations in a single transform.
    # It goes in the geometric list, but SatTrivialTransform.apply_segmentation()
    # correctly handles radiometric augmentations (erase, saturate, noise) by
    # returning segmentation unchanged, while geometric augmentations (translateX,
    # translateY, shearX, shearY, flip, rotate) are properly applied to segmentation.
    geometric.append(SatTrivialAugmentation(
        prob=0.8,  # 80% chance to apply one of the Sat-Trivial augmentations
        max_std=0.04,  # Gaussian noise max std
        erase_prob=0.05,  # Reduced erase probability
        saturate_prob=0.01,  # Reduced saturate probability
        noise_prob=0.05  # Reduced noise probability
    ))
    
    # Resize
    geometric.append(ResizeShortestEdge(cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST, sample_style="choice"))
    
    # Utility transforms (radiometric)
    radiometric.extend([
        MissingValueImputation(missing_value=0, window_size=7),
        MissingValueImputation(missing_value=None, window_size=7),
    ])
    
    return geometric, radiometric


def build_augmentation_set_ftw(cfg):
    """
    Build FTW augmentation set.
    Basic geometric augmentations: RandomRotation (90 degrees), RandomHorizontalFlip,
    RandomVerticalFlip, RandomSharpness.
    """
    geometric = []
    radiometric = []
    
    # Geometric transforms
    geometric.append(SafeRandomRotation([90], expand=False, interp=cv2.INTER_LINEAR, center=None))
    geometric.append(T.RandomFlip(prob=0.5, horizontal=True, vertical=False))
    geometric.append(T.RandomFlip(prob=0.5, horizontal=False, vertical=True))
    
    # Resize
    geometric.append(ResizeShortestEdge(cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST, sample_style="choice"))
    
    # Radiometric transforms
    radiometric.append(RandomSharpness(prob=0.5, sharpness_range=(0.5, 2.0)))
    
    # Utility transforms
    radiometric.extend([
        MissingValueImputation(missing_value=0, window_size=7),
        MissingValueImputation(missing_value=None, window_size=7),
    ])
    
    return geometric, radiometric


def build_augmentation_set_prue(cfg):
    """
    Build Prue augmentation set.
    FTW + additional augmentations: Channel shuffle, RandomBrightness, RandomResizedCrop.
    """
    geometric = []
    radiometric = []
    
    # Build FTW augmentations directly (avoid recursive call)
    # Geometric transforms
    geometric.append(SafeRandomRotation([90], expand=False, interp=cv2.INTER_LINEAR, center=None))
    geometric.append(T.RandomFlip(prob=0.5, horizontal=True, vertical=False))
    geometric.append(T.RandomFlip(prob=0.5, horizontal=False, vertical=True))
    
    # Additional geometric: RandomResizedCrop (before resize)
    image_size = getattr(cfg.INPUT, 'IMAGE_SIZE', 256)
    geometric.append(RandomResizedCrop(
        prob=0.5,
        size=(image_size, image_size),
        scale=(0.3, 0.9),
        ratio=(0.75, 1.33)
    ))
    
    # Final resize
    geometric.append(ResizeShortestEdge(cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST, sample_style="choice"))
    
    # Radiometric transforms (FTW + Prue additions)
    radiometric.append(RandomSharpness(prob=0.5, sharpness_range=(0.5, 2.0)))
    radiometric.append(ChannelShuffle(prob=0.5, num_channels=8))  # For 8-channel stacked images
    radiometric.append(T.RandomBrightness(0.5, 1.5))  # brightness=(0.5, 1.5) from Prue config
    
    # Utility transforms
    radiometric.extend([
        MissingValueImputation(missing_value=0, window_size=7),
        MissingValueImputation(missing_value=None, window_size=7),
    ])
    
    return geometric, radiometric


def build_augmentation_set_lsj(cfg):
    """
    Build LSJ (Large Scale Jittering) augmentation set.
    Current default when ENABLE_TRAIN_AUGS=False: ResizeScale, FixedSizeCrop, RandomFlip.
    """
    geometric = []
    radiometric = []
    
    if cfg.INPUT.RANDOM_FLIP != "none":
        # Handle "both" by creating two separate RandomFlip transforms
        if cfg.INPUT.RANDOM_FLIP in ['both', 'horizontal']:
            geometric.append(T.RandomFlip(horizontal=True, vertical=False))
        if cfg.INPUT.RANDOM_FLIP in ['both', 'vertical']:
            geometric.append(T.RandomFlip(horizontal=False, vertical=True))
    
    geometric.extend([
        T.ResizeScale(
            min_scale=cfg.INPUT.MIN_SCALE, max_scale=cfg.INPUT.MAX_SCALE, 
            target_height=cfg.INPUT.IMAGE_SIZE, target_width=cfg.INPUT.IMAGE_SIZE
        ),
        T.FixedSizeCrop(crop_size=(cfg.INPUT.IMAGE_SIZE, cfg.INPUT.IMAGE_SIZE)),
        ResizeShortestEdge(cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST, sample_style="choice")
    ])
    
    # No radiometric transforms for LSJ
    return geometric, radiometric


def build_augmentation_set_current(cfg):
    """
    Build Current augmentation set.
    Current ag-seg augmentations (when ENABLE_TRAIN_AUGS=True): RandomFlip, SafeRandomRotation,
    RandomCrop, RandomErase, RandomOversaturate, GaussianNoise, MissingValueImputation.
    """
    geometric = []
    radiometric = []
    
    # Geometric transforms
    if cfg.INPUT.RANDOM_FLIP == 'horizontal':
        geometric.append(T.RandomFlip(prob=0.25, horizontal=True, vertical=False))
    elif cfg.INPUT.RANDOM_FLIP == 'vertical':
        geometric.append(T.RandomFlip(prob=0.25, horizontal=False, vertical=True))
    
    geometric.append(SafeRandomRotation(
        [90, 180, 270],
        expand=False,
        interp=cv2.INTER_LINEAR,
        center=None
    ))
    
    if cfg.INPUT.CROP.ENABLED:
        geometric.append(T.RandomCrop(
            crop_type=cfg.INPUT.CROP.TYPE,
            crop_size=(cfg.INPUT.CROP.SIZE[0], cfg.INPUT.CROP.SIZE[1])
        ))
    
    geometric.append(ResizeShortestEdge(cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST, sample_style="choice"))
    
    # Radiometric transforms
    radiometric.extend([
        RandomErase(max_erased_groups=9, prob=0.1),
        RandomOversaturate(prob=0.01),
        GaussianNoise(max_std=0.04, prob=0.1),
        MissingValueImputation(missing_value=0, window_size=7),
        MissingValueImputation(missing_value=None, window_size=7),
    ])
    
    return geometric, radiometric


def build_augmentation_set_none(cfg):
    """
    Build no-augmentation set (for inference or baseline).
    Only essential transforms: resize and utility imputation.
    """
    geometric = []
    radiometric = []
    
    geometric.append(ResizeShortestEdge(
        cfg.INPUT.MIN_SIZE_TEST, 
        cfg.INPUT.MAX_SIZE_TEST,
        sample_style="choice"
    ))
    
    radiometric.extend([
        MissingValueImputation(missing_value=0, window_size=7),
        MissingValueImputation(missing_value=None, window_size=7)
    ])
    
    return geometric, radiometric


def build_transform_gen(cfg, is_train):
    """
    Build transforms for satellite imagery (B,G,R,NIR).
    
    Returns a list of all transforms (maintains backward compatibility).
    For proper separation, use build_transform_gen_separated() instead.
    """
    geometric, radiometric = build_transform_gen_separated(cfg, is_train)
    # Combine for backward compatibility (geometric first, then radiometric)
    return geometric + radiometric


def build_transform_gen_separated(cfg, is_train):
    """
    Build transforms separated into geometric and radiometric categories.
    
    Geometric transforms affect spatial coordinates and must be applied to both
    image and segmentation labels. Radiometric transforms only affect pixel values
    and should only be applied to the image.
    
    Augmentation sets can be selected via cfg.INPUT.AUGMENTATION_SET:
    - "SatTrivial": Single random augmentation per image (translateX, translateY, shearX, shearY, flip, rotate, erase, saturate, noise)
    - "FTW": Basic geometric (RandomRotation, RandomHorizontalFlip, RandomVerticalFlip, RandomSharpness)
    - "Prue": FTW + Channel shuffle, RandomBrightness, RandomResizedCrop
    - "LSJ": Large Scale Jittering (ResizeScale, FixedSizeCrop, RandomFlip)
    - "Current": Current ag-seg augmentations (RandomFlip, SafeRandomRotation, RandomCrop, RandomErase, etc.)
    - "None": No augmentations (only resize and imputation)
    
    If AUGMENTATION_SET is not set, falls back to old behavior using ENABLE_TRAIN_AUGS.
    
    Args:
        cfg: Detectron2 config
        is_train: Whether this is for training
        
    Returns:
        tuple: (geometric_transforms, radiometric_transforms)
    """
    # Check if augmentation set is specified
    augmentation_set = getattr(cfg.INPUT, 'AUGMENTATION_SET', None)
    
    if is_train and augmentation_set is not None:
        # Use augmentation set builders
        augmentation_set_lower = augmentation_set.lower()
        
        if augmentation_set_lower == "sattrivial":
            return build_augmentation_set_sattrivial(cfg)
        elif augmentation_set_lower == "ftw":
            return build_augmentation_set_ftw(cfg)
        elif augmentation_set_lower == "prue":
            return build_augmentation_set_prue(cfg)
        elif augmentation_set_lower == "lsj":
            return build_augmentation_set_lsj(cfg)
        elif augmentation_set_lower == "current":
            return build_augmentation_set_current(cfg)
        elif augmentation_set_lower == "none":
            return build_augmentation_set_none(cfg)
        else:
            raise ValueError(
                f"Unknown augmentation set: {augmentation_set}. "
                f"Must be one of: SatTrivial, FTW, Prue, LSJ, Current, None"
            )
    
    # Fall back to old behavior (backward compatibility)
    geometric_transforms = []
    radiometric_transforms = []
    
    if is_train:
        if cfg.INPUT.ENABLE_TRAIN_AUGS:
            # Geometric transforms (affect spatial coordinates - apply to image AND labels)
            geometric_candidates = [
                T.RandomFlip(prob=0.25, horizontal=True, vertical=False) if cfg.INPUT.RANDOM_FLIP == 'horizontal' else None,
                T.RandomFlip(prob=0.25, horizontal=False, vertical=True) if cfg.INPUT.RANDOM_FLIP == 'vertical' else None,
                SafeRandomRotation(
                    [90, 180, 270],
                    expand=False,
                    interp=cv2.INTER_LINEAR,
                    center=None
                ),
                T.RandomCrop(
                    crop_type=cfg.INPUT.CROP.TYPE,
                    crop_size=(cfg.INPUT.CROP.SIZE[0], cfg.INPUT.CROP.SIZE[1])
                ) if cfg.INPUT.CROP.ENABLED else None,
                ResizeShortestEdge(cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST, sample_style="choice")
            ]
            geometric_transforms = [t for t in geometric_candidates if t is not None]
            
            # Radiometric transforms (only affect pixel values - apply to image ONLY)
            radiometric_transforms = [
                RandomErase(max_erased_groups=9, prob=0.1),
                RandomOversaturate(prob=0.01),
                GaussianNoise(max_std=0.04, prob=0.1),
                MissingValueImputation(missing_value=0, window_size=7),
                MissingValueImputation(missing_value=None, window_size=7),
            ]
        else:
            # LSJ (Large Scale Jittering) - all geometric
            if cfg.INPUT.RANDOM_FLIP != "none":
                # Handle "both" by creating two separate RandomFlip transforms
                if cfg.INPUT.RANDOM_FLIP in ['both', 'horizontal']:
                    geometric_transforms.append(T.RandomFlip(horizontal=True, vertical=False))
                if cfg.INPUT.RANDOM_FLIP in ['both', 'vertical']:
                    geometric_transforms.append(T.RandomFlip(horizontal=False, vertical=True))
            geometric_transforms.extend([
                T.ResizeScale(
                    min_scale=cfg.INPUT.MIN_SCALE, max_scale=cfg.INPUT.MAX_SCALE, 
                    target_height=cfg.INPUT.IMAGE_SIZE, target_width=cfg.INPUT.IMAGE_SIZE
                ),
                T.FixedSizeCrop(crop_size=(cfg.INPUT.IMAGE_SIZE, cfg.INPUT.IMAGE_SIZE)),
                ResizeShortestEdge(cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST, sample_style="choice")
            ])
            # No radiometric transforms for LSJ
            radiometric_transforms = []
    else:
        # Inference: only geometric (resize) and utility (imputation)
        geometric_transforms = [
            ResizeShortestEdge(
                cfg.INPUT.MIN_SIZE_TEST, 
                cfg.INPUT.MAX_SIZE_TEST,
                sample_style="choice"
            )
        ]
        radiometric_transforms = [
            MissingValueImputation(missing_value=0, window_size=7),
            MissingValueImputation(missing_value=None, window_size=7)
        ]
    
    return geometric_transforms, radiometric_transforms    