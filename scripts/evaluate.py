import argparse
import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from setup import shared_setup
from trainer.custom_trainer import CustomTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def main(args):
    """
    Main function to run model evaluation using the full test set.
    """
    # Setup configuration and register datasets
    cfg = shared_setup(args)
    os.makedirs(args.output, exist_ok=True)
    
    logger.info(f"Starting full evaluation. Results will be saved to: {args.output}")
    
    # Run the full test using the custom trainer's class method
    results = CustomTrainer.full_test(cfg, eval_output_dir=args.output)
    
    logger.info("Full evaluation complete.")
    logger.info(f"Final results: {results}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a panoptic segmentation model on the full test dataset.")
    parser.add_argument("--config-file", required=True, help="Path to the model config file.")
    parser.add_argument("--weights", required=True, help="Path to the model weights.")
    parser.add_argument("--output", required=True, help="Directory to save evaluation results.")
    parser.add_argument("--coco-root", required=True, help="Root directory of the COCO dataset for metadata and images.")
    parser.add_argument("--opts", default=[], nargs=argparse.REMAINDER, help="Additional config options to override, e.g., DATASETS.COUNTRIES_EVAL 'austria,belgium' TEST.EVAL_SUBSET_SIZE 200")
    
    args = parser.parse_args()
    # Add output dir to opts so it's available in shared_setup
    args.opts.extend(['OUTPUT_DIR', args.output])
    
    main(args)