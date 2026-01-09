import os
import io
import logging
import numpy as np
import time, datetime
import torch
import glob
import itertools
from torch.utils.tensorboard import SummaryWriter
import detectron2.utils.comm as comm
from detectron2.engine import HookBase
from detectron2.utils.logger import log_every_n_seconds
import matplotlib.pyplot as plt
from PIL import Image
import psutil
from contextlib import contextmanager


'''
Custom hooks for training, validation, and monitoring.
'''

class CheckpointCleanupHook(HookBase):
    def __init__(self, checkpoint_dir, keep_last=5):
        """
        Args:
            checkpoint_dir (str): Directory where checkpoints are saved
            keep_last (int): Number of latest checkpoints to keep (default: 5)
        """
        self.checkpoint_dir = checkpoint_dir
        self.keep_last = keep_last

    def after_step(self):
        # Only run after evaluation
        if not self.trainer.iter % self.trainer.cfg.TEST.EVAL_PERIOD == 0:
            return

        # Get list of checkpoint files
        checkpoint_files = glob.glob(os.path.join(self.checkpoint_dir, "model_*.pth"))
        
        # Sort by modification time (newest first)
        checkpoint_files.sort(key=os.path.getmtime, reverse=True)
        
        # Keep only the latest N checkpoints
        for checkpoint_file in checkpoint_files[self.keep_last:]:
            try:
                os.remove(checkpoint_file)
            except OSError as e:
                print(f"Error deleting {checkpoint_file}: {e}")

class MemoryMonitorHook(HookBase):
    """
    Monitors CPU usage, RAM, and shared memory usage during training; also GPU usage.
    Designed for debugging memory issues, particularly when working with large datasets
    or complex models like Mask2Former with different backbones.
    """
    def __init__(self, log_period=20):
        """
        Args:
            log_period (int): The number of iterations between logging memory statistics
        """
        self._log_period = log_period
        self._logger = logging.getLogger(__name__)
        self._step_times = []
        self._peak_memory = 0
        self._start_time = time.time()
    
    @contextmanager
    def _timed_region(self):
        """Context manager to time a region of code."""
        start = time.time()
        yield
        end = time.time()
        self._step_times.append(end - start)
        # Keep only recent times
        if len(self._step_times) > 50:
            self._step_times.pop(0)
    
    def before_step(self):
        """Start timing the step."""
        self._step_start_time = time.time()
    
    def after_step(self):
        """Log memory statistics after each step based on the specified period."""
        # Record step time
        step_time = time.time() - self._step_start_time
        self._step_times.append(step_time)
        if len(self._step_times) > 20:
            self._step_times.pop(0)
        
        # Only log at specified iterations
        if self.trainer.iter % self._log_period != 0:
            return
        
        # Get process and system information
        process = psutil.Process()
        
        # Get CPU usage (average over recent period)
        cpu_percent = process.cpu_percent()
        
        # Get memory usage stats
        virtual_mem = psutil.virtual_memory()
        swap_mem = psutil.swap_memory()
        process_mem = process.memory_info()
        
        # Calculate memory usage in GB
        ram_used_gb = process_mem.rss / (1024 * 1024 * 1024)
        total_ram_gb = virtual_mem.total / (1024 * 1024 * 1024)
        ram_percent = virtual_mem.percent
        
        # Track peak memory
        if ram_used_gb > self._peak_memory:
            self._peak_memory = ram_used_gb
        
        # Calculate shared memory usage
        shared_mem_mb = process_mem.shared / (1024 * 1024)
        
        # Calculate average step time
        avg_step_time = sum(self._step_times) / len(self._step_times) if self._step_times else 0
        
        # Calculate ETA based on average step time
        steps_remaining = self.trainer.max_iter - self.trainer.iter
        eta_seconds = avg_step_time * steps_remaining
        hours, remainder = divmod(eta_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        eta = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
        
        # Calculate elapsed time
        elapsed_seconds = time.time() - self._start_time
        elapsed_hours, remainder = divmod(elapsed_seconds, 3600)
        elapsed_minutes, elapsed_seconds = divmod(remainder, 60)
        elapsed = f"{int(elapsed_hours)}h {int(elapsed_minutes)}m {int(elapsed_seconds)}s"
        
        # Log basic information
        self._logger.info(
            f"[Iter {self.trainer.iter}/{self.trainer.max_iter}] "
            f"CPU: {cpu_percent:.1f}% | "
            f"RAM: {ram_used_gb:.2f}GB/{total_ram_gb:.1f}GB ({ram_percent:.1f}%) | "
            f"Shared: {shared_mem_mb:.2f}MB | "
            f"Peak: {self._peak_memory:.2f}GB | "
            f"Step: {avg_step_time:.3f}s | "
            f"Elapsed: {elapsed} | "
            f"ETA: {eta}"
        )
        
        # Log GPU information separately if available
        if torch.cuda.is_available():
            '''
            Monitor GPU memory usage and write to TensorBoard.
            https://pytorch.org/docs/stable/notes/cuda.html#cuda-memory-management

            memory_allocated(), max_memory_allocated(): monitor memory occupied by tensors
            memory_reserved(), max_memory_reserved(): monitor total memory managed by caching allocator

            Notes on max_memory_allocated(): maximum GPU memory occupied by tensors in bytes for a given device
            default returns peak allocated memory since the beginning of the program;
            reset_peak_memory_stats() can be used to reset the starting point in tracking this metric
            Combine functions to measure peak allocated memory usage of each iteration in a training loop
            
            '''
            gpu_info = []
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / (1024 * 1024 * 1024)
                reserved = torch.cuda.memory_reserved(i) / (1024 * 1024 * 1024)
                utilization = torch.cuda.utilization(i)  # GPU utilization percentage
                gpu_info.append(f"GPU {i}: {allocated:.2f}GB/{reserved:.2f}GB ({utilization}%)")
                
                max_allocated = torch.cuda.max_memory_allocated(i) / (1024 * 1024 * 1024)
                max_reserved = torch.cuda.max_memory_reserved(i) / (1024 * 1024 * 1024) 

                # Add metrics to storage for TensorBoard
                self.trainer.storage.put_scalar(f"system/gpu_{i}_allocated_gb", allocated)
                self.trainer.storage.put_scalar(f"system/gpu_{i}_reserved_gb", reserved)
                self.trainer.storage.put_scalar(f"system/gpu_{i}_utilization", utilization)
                self.trainer.storage.add_scalar(f"system/gpu_{i}_max_allocated_memory_gb", max_allocated, self.trainer.iter)
                self.trainer.storage.add_scalar(f"system/gpu_{i}_max_reserved_memory_gb", max_reserved, self.trainer.iter)
            
            self._logger.info(" | ".join(gpu_info))
        
        # Add metrics to storage for TensorBoard/visualization
        self.trainer.storage.put_scalar("system/cpu_percent", cpu_percent)
        self.trainer.storage.put_scalar("system/ram_gb", ram_used_gb)
        self.trainer.storage.put_scalar("system/ram_percent", ram_percent)
        self.trainer.storage.put_scalar("system/shared_memory_mb", shared_mem_mb)
        self.trainer.storage.put_scalar("system/step_time", avg_step_time)
        self.trainer.storage.put_scalar("system/peak_memory_gb", self._peak_memory)
        
        # Monitor swap usage as well
        swap_used_gb = swap_mem.used / (1024 * 1024 * 1024)
        swap_total_gb = swap_mem.total / (1024 * 1024 * 1024)
        swap_percent = swap_mem.percent
        
        self.trainer.storage.put_scalar("system/swap_gb", swap_used_gb)
        self.trainer.storage.put_scalar("system/swap_percent", swap_percent)

class SimpleTimingHook(HookBase):
    """
    A simpler timing hook that just measures overall iteration time and components.
    """
    def __init__(self, log_period=20):
        self._log_period = log_period
        self._timings = {
            "data_loading": [],
            "forward": [],
            "backward": [],
            "optimizer": []
        }
        self._step_start = None
        self._tb_logger = None
        
    def before_train(self):
        """Set up TensorBoard writer - use shared writer from trainer if available"""
        if comm.is_main_process():
            # Try to get shared writer from trainer, otherwise create our own
            if hasattr(self.trainer, '_shared_tb_writer') and self.trainer._shared_tb_writer is not None:
                self._tb_logger = self.trainer._shared_tb_writer
            else:
                # Use same directory as Detectron2's TensorboardXWriter for consistency
                self._tb_logger = SummaryWriter(self.trainer.cfg.OUTPUT_DIR)
    
    def before_step(self):
        """Mark time before step begins"""
        self._step_start = time.time()
        self._data_load_end = time.time()  # Data loading just finished
        
        # Record data loading time
        if hasattr(self, "_prev_step_end"):
            data_time = self._data_load_end - self._prev_step_end
            self._timings["data_loading"].append(data_time)
    
    def after_step(self):
        """Record timing information after step"""
        step_end = time.time()
        
        # Skip for warmup iterations
        if self.trainer.iter < 5:
            self._prev_step_end = step_end
            return
            
        # Record timings
        iter_time = step_end - self._step_start
        
        # Log periodically
        if self.trainer.iter % self._log_period == 0 and comm.is_main_process():
            # Log basic timing info
            for k, v in self._timings.items():
                if v:
                    # Calculate statistics
                    avg_time = sum(v) / len(v)
                    if self._tb_logger:
                        self._tb_logger.add_scalar(f"timing/{k}", avg_time, self.trainer.iter)
                    # Reset list
                    self._timings[k] = []
                    
            # Log total iteration time
            if self._tb_logger:
                self._tb_logger.add_scalar("timing/iteration", iter_time, self.trainer.iter)
                
        # Mark the end of this step
        self._prev_step_end = step_end
        
    def after_train(self):
        """Clean up TensorBoard writer - only close if we created it"""
        if self._tb_logger and not (hasattr(self.trainer, '_shared_tb_writer') and self._tb_logger is self.trainer._shared_tb_writer):
            self._tb_logger.close()

class ValidationHook(HookBase):
    """
    Hook for running validation during training and logging validation losses to TensorBoard.
    Provides validation loss logging during training with configurable period.
    """
    def __init__(self, cfg, model, data_loader, tb_writer=None, period=1000):
        self.cfg = cfg
        self._model = model
        self._data_loader = data_loader
        self._tb_writer = tb_writer
        self._period = period
        self._cpu_device = torch.device("cpu")
        self.logger = logging.getLogger(__name__)
        # validate only a subset each run to reduce cost and introduce variability
        self._max_val_batches = 8
        try:
            # cfg may not be a config node in some call paths
            test_cfg = getattr(cfg, "TEST", None)
            if test_cfg is not None:
                self._max_val_batches = int(getattr(test_cfg, "VAL_MAX_BATCHES", 8))
        except Exception:
            pass
    
    def before_train(self):
        """Set up TensorBoard writer - write to same directory as Detectron2's TensorBoard writer"""
        if comm.is_main_process():
            # Write to same directory as Detectron2's FilteredTensorboardXWriter (OUTPUT_DIR)
            self._tb_writer = SummaryWriter(self.trainer.cfg.OUTPUT_DIR)
    
    def _do_validation(self):
        """Run validation and log losses"""
        # Compute validation loss by running the model in training mode under no_grad.
        # Detectron2 models only return losses when self.training is True.
        was_training = self._model.training
        self._model.train()  # Must be train mode for losses
        total_losses = {}
        total_loss_sum = 0.0
        num_batches = 0
        used_files = []
        
        # re-iterable loader subset with randomized start offset
        try:
            dl_len_hint = len(self._data_loader)
        except Exception:
            dl_len_hint = None
        start_skip = 0
        if dl_len_hint and dl_len_hint > 0:
            # use iter-dependent offset to vary samples across validations
            start_skip = (max(0, self.trainer.iter) // max(1, self._period)) % dl_len_hint
        
        with torch.no_grad():
            # create a fresh iterator each call and skip some samples to vary the subset
            it = iter(self._data_loader)
            for _ in range(start_skip):
                try:
                    next(it)
                except StopIteration:
                    it = iter(self._data_loader)
                    break
            for inputs in itertools.islice(it, self._max_val_batches):
                # Record filenames for visibility
                try:
                    if isinstance(inputs, dict):
                        fn = inputs.get("file_name")
                        if fn:
                            used_files.append(fn)
                    elif isinstance(inputs, (list, tuple)) and len(inputs) > 0 and isinstance(inputs[0], dict):
                        fn = inputs[0].get("file_name")
                        if fn:
                            used_files.append(fn)
                except Exception:
                    pass
                
                # Forward pass
                outputs = self._model(inputs)
                
                # Extract losses
                batch_loss_sum = 0.0
                if isinstance(outputs, dict):
                    for key, value in outputs.items():
                        if 'loss' in key.lower():
                            if key not in total_losses:
                                total_losses[key] = 0.0
                            v = float(value.detach().item()) if hasattr(value, 'detach') else float(value)
                            total_losses[key] += v
                            batch_loss_sum += v
                total_loss_sum += batch_loss_sum
                num_batches += 1
        
        if was_training is False:
            self._model.eval()
        
        if num_batches == 0:
            self.logger.warning("ValidationHook: no batches evaluated; check data_loader is re-iterable.")
            return {}
        
        # Average and aggregate
        for k in list(total_losses.keys()):
            total_losses[k] /= num_batches
        total_loss_avg = total_loss_sum / num_batches
        
        # Derive aggregates
        val_ce_loss = sum(v for k, v in total_losses.items() if 'ce' in k.lower())
        val_mask_loss = sum(v for k, v in total_losses.items() if 'mask' in k.lower() and 'dice' not in k.lower())
        val_dice_loss = sum(v for k, v in total_losses.items() if 'dice' in k.lower())
        
        # Log which files contributed this round (first few)
        if len(used_files) > 0:
            sample_list = used_files[:min(5, len(used_files))]
            self.logger.info(f"[Validation] iter={self.trainer.iter} used_files(sample)={sample_list} batches={num_batches}")
        
        # Log to TensorBoard - use "validation/" prefix for separate plots
        if self._tb_writer and comm.is_main_process():
            self._tb_writer.add_scalar("validation/total_loss", total_loss_avg, self.trainer.iter)
            self._tb_writer.add_scalar("validation/loss_ce", val_ce_loss, self.trainer.iter)
            self._tb_writer.add_scalar("validation/loss_mask", val_mask_loss, self.trainer.iter)
            self._tb_writer.add_scalar("validation/loss_dice", val_dice_loss, self.trainer.iter)
        
        # Log to storage for console output - keep "validation/" prefix for console logs
        self.trainer.storage.put_scalar("validation/loss_ce", val_ce_loss)
        self.trainer.storage.put_scalar("validation/loss_mask", val_mask_loss)
        self.trainer.storage.put_scalar("validation/loss_dice", val_dice_loss)
        self.trainer.storage.put_scalar("validation/total_loss", total_loss_avg)
        
        return total_losses
    
    def after_step(self):
        """Run validation at specified intervals"""
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_validation()
    
    def after_train(self):
        """Clean up TensorBoard writer"""
        if self._tb_writer:
            self._tb_writer.close()


class PredictionVisualizationHook(HookBase):
    """Hook to run model on validation samples and log visualizations to TensorBoard"""
    def __init__(self, cfg, model, data_loader, metadata=None, tb_writer=None, period=1000, panoptic_threshold=None, max_images=8):
        self.cfg = cfg
        self._model = model
        self._data_loader = data_loader
        # resolve metadata if not provided
        if metadata is None:
            try:
                ds_names = []
                try:
                    if hasattr(cfg, "DATASETS"):
                        if hasattr(cfg.DATASETS, "TEST") and len(cfg.DATASETS.TEST) > 0:
                            ds_names = list(cfg.DATASETS.TEST)
                        elif hasattr(cfg.DATASETS, "TRAIN") and len(cfg.DATASETS.TRAIN) > 0:
                            ds_names = list(cfg.DATASETS.TRAIN)
                except Exception:
                    ds_names = []
                meta_name = ds_names[0] if len(ds_names) > 0 else None
                from detectron2.data import MetadataCatalog
                self._metadata = MetadataCatalog.get(meta_name) if meta_name is not None else MetadataCatalog.get("_default")
            except Exception:
                self._metadata = metadata
        else:
            self._metadata = metadata
        self._tb_writer = tb_writer
        self._period = period
        self._cpu_device = torch.device("cpu")
        # Use config threshold if available, otherwise use provided or default
        if panoptic_threshold is None:
            try:
                # Use the same threshold as model inference for consistency
                self._panoptic_threshold = cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD
            except (AttributeError, KeyError):
                self._panoptic_threshold = 0.1  # Default to match current config
        else:
            self._panoptic_threshold = panoptic_threshold
        self._max_images = int(max_images) if max_images is not None else 8
        self.logger = logging.getLogger(__name__)
    
    def _create_error_image(self, error_msg, height=512, width=512):
        """Create a simple error image with text message"""
        try:
            from PIL import Image, ImageDraw, ImageFont
            # Create a white image
            img = Image.new('RGB', (width, height), color='white')
            draw = ImageDraw.Draw(img)
            
            # Try to use a default font, fallback to basic if not available
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
            except:
                try:
                    font = ImageFont.load_default()
                except:
                    font = None
            
            # Draw error message (split into multiple lines if needed)
            text_lines = str(error_msg).split('\n')
            y_offset = height // 2 - (len(text_lines) * 25) // 2
            for line in text_lines:
                # Truncate if too long
                if len(line) > 50:
                    line = line[:47] + "..."
                bbox = draw.textbbox((0, 0), line, font=font) if font else (0, 0, len(line) * 10, 20)
                text_width = bbox[2] - bbox[0] if font else len(line) * 10
                x = (width - text_width) // 2
                draw.text((x, y_offset), line, fill='red', font=font)
                y_offset += 25
            
            # Convert PIL Image to numpy array
            return np.array(img)
        except Exception as e:
            # Fallback: create a simple colored image with numpy
            self.logger.warning(f"Failed to create error image with PIL: {e}")
            img = np.ones((height, width, 3), dtype=np.uint8) * 255
            # Add a red border
            img[:5, :] = [255, 0, 0]  # Top border
            img[-5:, :] = [255, 0, 0]  # Bottom border
            img[:, :5] = [255, 0, 0]  # Left border
            img[:, -5:] = [255, 0, 0]  # Right border
            return img
    
    def before_train(self):
        """Set up TensorBoard writer and metadata - use shared writer from trainer if available"""
        if comm.is_main_process():
            # Try to get shared writer from trainer, otherwise create our own
            if hasattr(self.trainer, '_shared_tb_writer') and self.trainer._shared_tb_writer is not None:
                self._tb_writer = self.trainer._shared_tb_writer
            else:
                # Use same directory as Detectron2's TensorboardXWriter for consistency
                self._tb_writer = SummaryWriter(self.trainer.cfg.OUTPUT_DIR)
                # Store as shared writer for other hooks to use
                self.trainer._shared_tb_writer = self._tb_writer
            
            # Get metadata for visualization - try cfg first, then trainer.cfg
            try:
                from detectron2.data import MetadataCatalog
                if hasattr(self, 'cfg') and self.cfg is not None:
                    cfg_to_use = self.cfg
                else:
                    cfg_to_use = self.trainer.cfg
                
                if hasattr(cfg_to_use, "DATASETS") and hasattr(cfg_to_use.DATASETS, "TEST") and len(cfg_to_use.DATASETS.TEST) > 0:
                    dataset_name = cfg_to_use.DATASETS.TEST[0]
                else:
                    dataset_name = None
                
                if dataset_name:
                    self._metadata = MetadataCatalog.get(dataset_name)
                else:
                    # Fallback: try to get from data loader
                    try:
                        dataset_name = self._data_loader.dataset.dataset.name if hasattr(self._data_loader.dataset, 'dataset') else None
                        if dataset_name:
                            self._metadata = MetadataCatalog.get(dataset_name)
                    except Exception:
                        pass
                
                # Log metadata initialization once
                if dataset_name:
                    self.logger.info(f"[VizInit] Initialized metadata for dataset: {dataset_name}")
            except Exception as e:
                self.logger.warning(f"[VizInit] Failed to initialize metadata in before_train: {e}", exc_info=True)
    
    def _visualize_predictions(self):
        """Generate and log prediction visualizations"""
        if not comm.is_main_process():
            return
            
        try:
            self._model.eval()
        
            with torch.no_grad():
                # Get a batch of validation data
                for inputs in self._data_loader:
                    try:
                        # Run inference - need to wrap single input in list if needed
                        if isinstance(inputs, dict):
                            inputs = [inputs]
                        
                        # Preserve original inputs for GT visualization (before model modifies them)
                        original_inputs = []
                        for inp in inputs:
                            orig_inp = {}
                            # Preserve keys needed for GT visualization
                            if "pan_seg_file_name" in inp:
                                orig_inp["pan_seg_file_name"] = inp["pan_seg_file_name"]
                            if "segments_info" in inp:
                                orig_inp["segments_info"] = inp["segments_info"]
                            if "file_name" in inp:
                                orig_inp["file_name"] = inp["file_name"]
                            original_inputs.append(orig_inp)
                        
                        # Run inference on the batch
                        outputs = self._model(inputs)
                        
                        # Handle both list and dict outputs
                        if isinstance(outputs, dict):
                            outputs = [outputs]
                        
                        # Process up to max_images
                        num_images = min(len(inputs), self._max_images, len(outputs))
                
                        for i in range(num_images):
                            try:
                                input_dict = inputs[i]
                                orig_input_dict = original_inputs[i]
                                output = outputs[i]
                                
                                # Get image identifier for logging
                                image_id = input_dict.get("image_id", f"idx_{i}")
                                file_name = input_dict.get("file_name", orig_input_dict.get("file_name", "unknown"))
                                
                                # Get image and map to 3-channel RGB for visualization
                                image = input_dict["image"].cpu().numpy()
                                # Detect if config declares BGR-style ordering
                                fmt = getattr(getattr(self.trainer, "cfg", None), "INPUT", None)
                                img_format = getattr(fmt, "FORMAT", "RGB")
                                is_bgr = str(img_format).upper().startswith("BGR")

                                c = image.shape[0]
                                if c >= 3:
                                    # For 8-channel stacked (BGRN BGRN...), use first window BGR
                                    if c == 8:
                                        rgb_src = image[:3]  # B,G,R
                                    # For 4-channel BGRN, use B,G,R
                                    elif c == 4:
                                        rgb_src = image[:3]  # B,G,R
                                    else:
                                        rgb_src = image[:3]

                                    # If stored as BGR*, swap to RGB for viz
                                    if is_bgr and rgb_src.shape[0] >= 3:
                                        rgb_src = rgb_src[[2, 1, 0], ...]

                                    rgb_image = rgb_src.transpose(1, 2, 0)
                                else:
                                    # Fallback: just transpose whatever channels exist
                                    rgb_image = image.transpose(1, 2, 0)
                                # Ensure we only keep 3 channels for vis to avoid broadcasting issues
                                if rgb_image.shape[2] > 3:
                                    rgb_image = rgb_image[..., :3]
                                
                                # Normalize image for visualization
                                rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min() + 1e-8)
                                rgb_image = (rgb_image * 255).astype(np.uint8)
                                
                                # Create visualizations - pass original dict for GT
                                self._create_prediction_visualization(
                                    rgb_image, input_dict, orig_input_dict, output, self.trainer.iter, i
                                )
                            except Exception as e:
                                logging.warning(f"Error visualizing image {i}: {e}", exc_info=True)
                    
                    except Exception as e:
                        logging.warning(f"Error in visualization batch processing: {e}", exc_info=True)
                
                    break  # Only process one batch
        except Exception as e:
            logging.error(f"Error in _visualize_predictions: {e}", exc_info=True)
        finally:
            self._model.train()
    
    def _create_prediction_visualization(self, rgb_image, input_dict, orig_input_dict, output, iteration, img_idx):
        """Create simple red/green/yellow overlay visualization comparing GT and predictions"""
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import cv2
        from panopticapi.utils import rgb2id
        
        h, w = rgb_image.shape[:2]
        img_size = (h, w)
        
        # Create binary masks for GT and predictions
        gt_mask = np.zeros(img_size, dtype=np.uint8)
        pred_mask = np.zeros(img_size, dtype=np.uint8)
        
        # Helper to determine if a segment is a "thing" when isthing is missing
        def _is_thing(seg_info):
            if seg_info.get("isthing") is not None:
                return bool(seg_info["isthing"])
            # Fallback: treat category_id==1 as thing (ag_field) and others as stuff
            return seg_info.get("category_id") == 1

        # Extract GT mask from panoptic segmentation
        if "pan_seg_file_name" in orig_input_dict:
            try:
                pan_rgb = cv2.imread(orig_input_dict["pan_seg_file_name"], cv2.IMREAD_COLOR)
                if pan_rgb is not None:
                    pan_rgb = cv2.cvtColor(pan_rgb, cv2.COLOR_BGR2RGB)
                    gt_seg_np = rgb2id(pan_rgb)
                    segments_info = orig_input_dict.get("segments_info", [])
                    
                    # Resize GT to match image size
                    if gt_seg_np.shape[:2] != img_size:
                        gt_seg_np_int32 = gt_seg_np.astype(np.int32)
                        gt_seg_np_resized = cv2.resize(gt_seg_np_int32, (w, h), interpolation=cv2.INTER_NEAREST)
                        gt_seg_np = gt_seg_np_resized.astype(gt_seg_np.dtype)
                    
                    # Create binary mask: 1 for ag_field, 0 for background
                    # In GT, ag_field is identified by isthing=True (thing classes)
                    # Background has isthing=False (stuff class)
                    for seg_info in segments_info:
                        if _is_thing(seg_info):
                            seg_id = seg_info['id']
                            gt_mask[gt_seg_np == seg_id] = 1
            except Exception as e:
                self.logger.warning(f"Error loading GT mask: {e}")
        
        # Extract prediction mask from panoptic segmentation
        if "panoptic_seg" in output:
            try:
                pan_seg, segments_info = output["panoptic_seg"]
                # Convert to numpy if needed
                if isinstance(pan_seg, torch.Tensor):
                    pan_seg_np = pan_seg.cpu().numpy()
                else:
                    pan_seg_np = pan_seg
                
                # Ensure correct shape
                if pan_seg_np.shape[:2] != img_size:
                    if isinstance(pan_seg_np, np.ndarray):
                        pan_seg_np = cv2.resize(pan_seg_np.astype(np.int32), (w, h), interpolation=cv2.INTER_NEAREST)
                
                # Create binary mask: 1 for ag_field predictions, 0 for background
                # In predictions, ag_field is identified by isthing=True (thing classes)
                # Background has isthing=False (stuff class)
                for seg_info in segments_info:
                    if _is_thing(seg_info):
                        seg_id = seg_info['id']
                        pred_mask[pan_seg_np == seg_id] = 1
            except Exception as e:
                self.logger.warning(f"Error extracting prediction mask: {e}")
        
        # Create red/green/yellow overlay visualization
        vis = rgb_image.copy().astype(np.float32)
        
        # Define colors with transparency
        gt_color = np.array([0, 255, 0])  # Green
        pred_color = np.array([255, 0, 0])  # Red
        overlap_color = np.array([255, 255, 0])  # Yellow
        alpha = 0.6
        
        # Overlay ground truth only (green)
        gt_only = gt_mask & ~pred_mask
        vis[gt_only == 1] = vis[gt_only == 1] * (1 - alpha) + gt_color * alpha
        
        # Overlay predictions only (red)
        pred_only = pred_mask & ~gt_mask
        vis[pred_only == 1] = vis[pred_only == 1] * (1 - alpha) + pred_color * alpha
        
        # Overlay overlap (yellow)
        overlap = gt_mask & pred_mask
        vis[overlap == 1] = vis[overlap == 1] * (1 - alpha) + overlap_color * alpha
        
        # Convert back to uint8
        vis = np.clip(vis, 0, 255).astype(np.uint8)
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(vis)
        ax.axis('off')
        ax.set_title(f"GT: Green, Pred: Red, Overlap: Yellow (iter={iteration})")
        
        # Add legend
        legend_patches = [
            mpatches.Patch(color='green', alpha=alpha, label='Ground Truth Only'),
            mpatches.Patch(color='red', alpha=alpha, label='Prediction Only'),
            mpatches.Patch(color='yellow', alpha=alpha, label='Overlap'),
        ]
        ax.legend(handles=legend_patches, loc='lower right')
        
        plt.tight_layout()
        
        # Log to TensorBoard
        if self._tb_writer:
            self._tb_writer.add_figure(
                f"predictions/validation_image_{img_idx}", 
                fig, 
                global_step=iteration
            )
        
        plt.close(fig)
    
    def after_step(self):
        """Run visualization at specified intervals"""
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._visualize_predictions()
    
    def after_train(self):
        """Clean up TensorBoard writer - only close if we created it"""
        if self._tb_writer and not (hasattr(self.trainer, '_shared_tb_writer') and self._tb_writer is self.trainer._shared_tb_writer):
            self._tb_writer.close()