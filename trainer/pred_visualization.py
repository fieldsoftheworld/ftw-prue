from pathlib import Path
import logging
from typing import Optional, Tuple, Dict, Any
import matplotlib.pyplot as plt
import numpy as np
from detectron2.utils.colormap import random_color
from detectron2.utils.visualizer import Visualizer, ColorMode, _PanopticPrediction
import torch

logger = logging.getLogger(__name__)


_OFF_WHITE = (1.0, 1.0, 240.0 / 255)

class NoTextVisualizer(Visualizer):
    """
    A visualizer that doesn't draw any text labels on instances or segmentation
    """
    def overlay_instances(self, *args, **kwargs):
        # Force labels to be None regardless of what was passed
        kwargs['labels'] = None
        return super().overlay_instances(*args, **kwargs)

    def draw_binary_mask(self, binary_mask, color=None, *, edge_color=None, text=None, alpha=0.5, area_threshold=10):
        """Override to remove text from binary masks"""
        return super().draw_binary_mask(
            binary_mask, 
            color=color, 
            edge_color=edge_color, 
            text=None,  # Always pass None as text
            alpha=alpha, 
            area_threshold=area_threshold
        )
        
    def draw_panoptic_seg(self, panoptic_seg, segments_info, area_threshold=None, alpha=0.7):
        """
        Override to use our version of PanopticPrediction that won't draw text
        """
        # Create our custom prediction
        pred = _PanopticPrediction(panoptic_seg, segments_info, self.metadata)

        if self._instance_mode == ColorMode.IMAGE_BW:
            self.output.reset_image(self._create_grayscale_image(pred.non_empty_mask()))

        # Draw mask for all semantic segments first i.e. "stuff"
        for mask, sinfo in pred.semantic_masks():
            category_idx = sinfo["category_id"]
            try:
                mask_color = [x / 255 for x in self.metadata.stuff_colors[category_idx]]
            except (AttributeError, IndexError):
                mask_color = None

            # Pass text=None to draw_binary_mask
            self.draw_binary_mask(
                mask,
                color=mask_color,
                edge_color=_OFF_WHITE,
                text=None,  # No text
                alpha=alpha,
                area_threshold=area_threshold,
            )

        # Draw mask for all instances second
        all_instances = list(pred.instance_masks())
        if len(all_instances) == 0:
            return self.output
        
        masks, sinfo = list(zip(*all_instances))
        
        # We don't create labels here - pass None
        self.overlay_instances(masks=masks, labels=None, assigned_colors=None, alpha=alpha)

        return self.output
        
    # Also alias the predictions method
    draw_panoptic_seg_predictions = draw_panoptic_seg
    
    def _draw_text_in_mask(self, binary_mask, text, color):
        """
        Override to prevent drawing text in masks completely.
        """
        # Do nothing - don't draw any text
        pass

class SatelliteVisualizer:
    """Enhanced visualizer for satellite imagery predictions."""
    
    def __init__(self, metadata, instance_mode=ColorMode.IMAGE):
        """
        Args:
            metadata: Detectron2 metadata
            instance_mode: ColorMode for instance visualization
        """
        self.metadata = metadata
        self.instance_mode = instance_mode
        self.cpu_device = torch.device("cpu")
        
        # Validate metadata
        required_fields = [
            "thing_classes",
            "thing_colors",
            "thing_dataset_id_to_contiguous_id",
            "stuff_classes",
            "stuff_colors",
            "stuff_dataset_id_to_contiguous_id"
        ]
        
        missing_fields = [field for field in required_fields if not hasattr(metadata, field)]
        if missing_fields:
            logger.warning(f"Metadata missing required fields: {missing_fields}")
            # Set defaults if needed
            if "stuff_classes" in missing_fields:
                self.metadata.stuff_classes = self.metadata.thing_classes
            if "stuff_colors" in missing_fields:
                self.metadata.stuff_colors = [random_color(rgb=True, maximum=255) 
                                           for _ in self.metadata.stuff_classes]
            if "stuff_dataset_id_to_contiguous_id" in missing_fields:
                self.metadata.stuff_dataset_id_to_contiguous_id = \
                    self.metadata.thing_dataset_id_to_contiguous_id
        
    def _prepare_satellite_image(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare RGB and NIR false color composites from satellite image.
        
        Args:
            image (np.ndarray): Satellite image in (H,W,C) format.
                Can be 8-channel RGBNRGBN (two temporal windows) or 4-channel RGBN.
                For 8-channel images, uses the first temporal window (first 4 channels).
            
        Returns:
            tuple: (rgb_image, nir_composite) normalized to 0-255 range
        """
        # Handle 8-channel RGBNRGBN format by using first temporal window
        if image.shape[2] == 8:
            image = image[..., :4]
        elif image.shape[2] != 4:
            raise ValueError(f"Expected 4 or 8-channel image, got {image.shape[2]} channels")
            
        # Normalize each band to 0-255
        image_norm = np.zeros_like(image, dtype=np.float32)
        for i in range(4):
            band = image[..., i]
            if band.max() > band.min():
                image_norm[..., i] = 255 * (band - band.min()) / (band.max() - band.min())
                
        # Create RGB (using BGR bands)
        rgb_image = np.stack([
            image_norm[..., 2],  # R
            image_norm[..., 1],  # G
            image_norm[..., 0]   # B
        ], axis=2).astype(np.uint8)
        
        # Create NIR false color
        nir_composite = np.stack([
            image_norm[..., 3],  # NIR
            image_norm[..., 1],  # G
            image_norm[..., 2]   # R
        ], axis=2).astype(np.uint8)
        
        return rgb_image, nir_composite
    
    def visualize_predictions(self, 
                            image: np.ndarray, 
                            predictions: Dict[str, Any],
                            ground_truth: Optional[Dict[str, Any]] = None,
                            output_dir: Optional[str] = None,
                            image_id: Optional[str] = None) -> Optional[Path]:
        """
        Create comprehensive visualization of model predictions.
        
        Args:
            image (np.ndarray): Input BGRN image
            predictions (dict): Model predictions
            output_dir (str, optional): Directory to save visualizations
            image_id (str, optional): Identifier for the image
            
        Returns:
            Path to saved visualization (if output_dir provided), otherwise None
        """
        # Prepare output directory
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare RGB and NIR composites
        rgb_image, nir_composite = self._prepare_satellite_image(image)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))
        
        gs = fig.add_gridspec(2, 3) # 2 rows, 3 columns
        
        # 1. RGB Composite
        ax_rgb = fig.add_subplot(gs[0, 0])
        ax_rgb.imshow(rgb_image)
        ax_rgb.set_title("RGB Composite")
        ax_rgb.axis('off')
        
        # 2. NIR False Color
        ax_nir = fig.add_subplot(gs[0, 1])
        ax_nir.imshow(nir_composite)
        ax_nir.set_title("NIR False Color")
        ax_nir.axis('off')
        print('Images added')
        
        # 3. Semantic Segmentation
        ax_sem = fig.add_subplot(gs[0, 2])
        if "sem_seg" in predictions:
            print("Sem seg identified")
            sem_seg = predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
            v = Visualizer(rgb_image, self.metadata)
            sem_vis = v.draw_sem_seg(sem_seg, area_threshold=10, alpha=0.7)
            ax_sem.imshow(sem_vis.get_image())
            print("Sem seg plotted")
        else:
            ax_sem.imshow(rgb_image)
        ax_sem.set_title("Semantic Segmentation (Pred)")
        ax_sem.axis('off')
        
        # 4. Instance Segmentation
        ax_inst = fig.add_subplot(gs[1, 0])
        if "instances" in predictions:
            print("Instance seg identified")
            instances = predictions["instances"].to(self.cpu_device)
            v = NoTextVisualizer(rgb_image, self.metadata, instance_mode=self.instance_mode)
            inst_vis = v.draw_instance_predictions(predictions=instances)
            ax_inst.imshow(inst_vis.get_image())
            print("Instance seg plotted")
        else:
            ax_inst.imshow(rgb_image)
        ax_inst.set_title("Instance Segmentation (Pred)")
        ax_inst.axis('off')

        # 5. Panoptic Segmentation
        ax_pan = fig.add_subplot(gs[1, 1])
        if "panoptic_seg" in predictions:
            print("Panoptic seg identified")
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            print("panoptic_seg: ", panoptic_seg)
            print("segments_info: ", segments_info)
            v = NoTextVisualizer(rgb_image, self.metadata)
            for seg in segments_info:
                # if seg['id'] == 4988569:
                #     seg['category_id'] = 1
                # if seg['isthing'] == True:
                #     seg['category_id'] = 0
                if seg['id'] == 4988569:
                    seg['category_id'] = 2  # Background

            pan_vis = v.draw_panoptic_seg_predictions(
                panoptic_seg.to(self.cpu_device), segments_info, alpha=0.7#, labels=None
            )
            ax_pan.imshow(pan_vis.get_image())
            print("Panoptic seg plotted")
        else:
            ax_pan.imshow(rgb_image)
        ax_pan.set_title("Panoptic Segmentation (Pred)")
        ax_pan.axis('off')
        
        # Save plots if output directory provided
        print("Trying to save to output_dir: ", output_dir)
        if output_dir:
            # Adjust layout and save
            plt.tight_layout()
            vis_path = output_dir / f"predictions_{image_id or 'unknown'}.png"
            plt.savefig(vis_path, bbox_inches='tight', dpi=150)
            plt.close()
            
            return vis_path
        
        return None



