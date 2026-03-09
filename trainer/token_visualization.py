import torch
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os
from PIL import Image
import tempfile


class TokenVisualizer:
    """
    Utility class for visualizing token positions from AutoFocusFormer backbone
    at different stages of the network.
    """

    def __init__(self, output_dir, max_images=8, log_to_tensorboard=True):
        """
        Initialize the token visualizer.

        Args:
            output_dir (str): Directory to save visualizations
            max_images (int): Maximum number of images to visualize from batch
            log_to_tensorboard (bool): Whether to log to tensorboard
        """
        self.output_dir = output_dir
        self.max_images = max_images
        self.log_to_tensorboard = log_to_tensorboard

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Create a subdirectory for token visualizations
        self.viz_dir = os.path.join(output_dir, "token_visualizations")
        if not os.path.exists(self.viz_dir):
            os.makedirs(self.viz_dir)

        if log_to_tensorboard:
            self.writer = SummaryWriter(os.path.join(output_dir, "tensorboard"))

    def visualize_tokens(self, original_images, token_positions_list, iteration, stage_names=None):
        """
        Create visualization of token positions overlaid on original images.

        Args:
            original_images (torch.Tensor): Original input images [B, C, H, W]
            token_positions_list (list): List of token positions at different stages
                Each element is a tensor of shape [B, N, 2] where N is number of tokens
            iteration (int): Current training iteration (for naming)
            stage_names (list): Optional list of stage names
        """
        if stage_names is None:
            stage_names = [f"stage_{i}" for i in range(len(token_positions_list))]

        batch_size = original_images.shape[0]
        max_images = min(batch_size, self.max_images)

        # Process a subset of images from the batch
        for img_idx in range(max_images):
            image = original_images[img_idx].detach().clone()

            # Normalize image for visualization (assuming 4-channel satellite image)
            # Typically first 3 channels are BGR, convert to RGB
            if image.shape[0] >= 3:
                rgb_image = image[:3].clone()
                # Convert BGR to RGB if that's your format
                rgb_image = rgb_image.flip(0)

                # Normalize each channel independently to [0, 1]
                for c in range(rgb_image.shape[0]):
                    if rgb_image[c].max() > rgb_image[c].min():
                        rgb_image[c] = (rgb_image[c] - rgb_image[c].min()) / (rgb_image[c].max() - rgb_image[c].min())
            else:
                # For grayscale or fewer than 3 channels, repeat the first channel
                rgb_image = image[0:1].repeat(3, 1, 1)
                if rgb_image.max() > rgb_image.min():
                    rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())

            # Create a figure with subplots for each stage
            num_stages = len(token_positions_list)
            fig, axes = plt.subplots(1, num_stages + 1, figsize=(4 * (num_stages + 1), 4))

            # Plot original image
            axes[0].imshow(rgb_image.permute(1, 2, 0).cpu().numpy())
            axes[0].set_title("Original Image")
            axes[0].axis("off")

            # For each stage, create a token map
            for stage_idx, token_positions in enumerate(token_positions_list):
                stage_name = stage_names[stage_idx]
                ax = axes[stage_idx + 1]

                # Get token positions for this image
                pos = token_positions[img_idx]  # [N, 2]

                # Create empty token map at the size of the original image
                H, W = original_images.shape[2], original_images.shape[3]
                token_map = torch.zeros((H, W), device=image.device)

                # Convert token positions to image coordinates
                # AFF positions are in the downsampled coordinate space, need to scale back up
                if stage_idx == 0:
                    scale_factor = 4  # First stage is downsampled by 4
                else:
                    scale_factor = 2 ** (stage_idx + 1)  # Subsequent stages double the downsampling

                # Scale positions to match original image coordinates
                scaled_pos = pos.clone() * scale_factor

                # Round to nearest integer and clamp to image boundaries
                pos_y = scaled_pos[:, 1].long().clamp(0, H - 1)
                pos_x = scaled_pos[:, 0].long().clamp(0, W - 1)

                # Mark token positions
                for y, x in zip(pos_y, pos_x):
                    # Draw a small square around each token position
                    y1, y2 = max(0, y - 1), min(H - 1, y + 1)
                    x1, x2 = max(0, x - 1), min(W - 1, x + 1)
                    token_map[y1 : y2 + 1, x1 : x2 + 1] = 1.0

                # Create overlay image
                token_map_rgb = torch.zeros((3, H, W), device=image.device)
                token_map_rgb[0] = token_map  # Red channel

                # Blend with original image
                alpha = 0.7
                blended = alpha * token_map_rgb + (1 - alpha * token_map.unsqueeze(0)) * rgb_image

                # Display
                ax.imshow(blended.permute(1, 2, 0).cpu().numpy())
                ax.set_title(f"{stage_name} - {pos.shape[0]} tokens")
                ax.axis("off")

            # Save figure
            plt.tight_layout()
            fig_path = os.path.join(self.viz_dir, f"tokens_iter{iteration}_img{img_idx}.png")
            plt.savefig(fig_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

            # Log to tensorboard
            if self.log_to_tensorboard:
                self.writer.add_figure(f"token_maps/image_{img_idx}", fig, global_step=iteration)

        # Also log as a grid of images
        if self.log_to_tensorboard:
            for stage_idx, token_positions in enumerate(token_positions_list):
                stage_name = stage_names[stage_idx]

                # Create token maps for all selected images
                token_maps = []
                for img_idx in range(max_images):
                    image = original_images[img_idx]
                    pos = token_positions[img_idx]

                    # Create empty token map
                    H, W = image.shape[1], image.shape[2]
                    token_map = torch.zeros((3, H, W), device=image.device)

                    # Scale positions
                    if stage_idx == 0:
                        scale_factor = 4
                    else:
                        scale_factor = 2 ** (stage_idx + 1)

                    scaled_pos = pos.clone() * scale_factor
                    pos_y = scaled_pos[:, 1].long().clamp(0, H - 1)
                    pos_x = scaled_pos[:, 0].long().clamp(0, W - 1)

                    # Create RGB visualization
                    rgb_image = self._normalize_image(image)
                    for y, x in zip(pos_y, pos_x):
                        y1, y2 = max(0, y - 1), min(H - 1, y + 1)
                        x1, x2 = max(0, x - 1), min(W - 1, x + 1)
                        token_map[0, y1 : y2 + 1, x1 : x2 + 1] = 1.0  # Red dots

                    # Blend
                    alpha = 0.7
                    blended = alpha * token_map + (1 - alpha * token_map[0:1].repeat(3, 1, 1)) * rgb_image
                    token_maps.append(blended)

                # Create grid
                grid = torchvision.utils.make_grid(token_maps, nrow=4)
                self.writer.add_image(f"token_maps/{stage_name}", grid, global_step=iteration)

    def _normalize_image(self, image):
        """Normalize image for visualization"""
        if image.shape[0] >= 3:
            rgb_image = image[:3].clone()
            # Convert BGR to RGB if needed
            rgb_image = rgb_image.flip(0)

            # Normalize each channel independently
            for c in range(rgb_image.shape[0]):
                if rgb_image[c].max() > rgb_image[c].min():
                    rgb_image[c] = (rgb_image[c] - rgb_image[c].min()) / (rgb_image[c].max() - rgb_image[c].min())
            return rgb_image
        else:
            # For fewer channels
            rgb_image = image[0:1].repeat(3, 1, 1)
            if rgb_image.max() > rgb_image.min():
                rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())
            return rgb_image
