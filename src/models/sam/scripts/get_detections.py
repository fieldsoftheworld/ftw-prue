import os
import sys
import yaml
import argparse
from pathlib import Path

# Add parent directory to path to import sam_controller
sys.path.insert(0, str(Path(__file__).parent.parent))

from sam_controller import SAMEval

def main():
    parser = argparse.ArgumentParser(description="Train SAM")
    parser.add_argument("config", type=str, help="path to config file")
    args = parser.parse_args()

    config_file = args.config

    with open(config_file, 'r') as f:
        config_data = yaml.safe_load(f)

    evaler = SAMEval(config_data)
    detections = evaler.get_detections()

    # then do whatever you want with the detections
    print('Done!')

if __name__ == '__main__':
    main()
