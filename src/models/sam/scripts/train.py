import os
import sys
import yaml
import argparse
from pathlib import Path

# Add parent directory to path to import sam_controller
sys.path.insert(0, str(Path(__file__).parent.parent))

from sam_controller import SAMTrainer

def main():
    parser = argparse.ArgumentParser(description="Train SAM")
    parser.add_argument("config", type=str, help="path to config file") # required=True
    args = parser.parse_args()

    config_file = args.config

    with open(config_file, 'r') as f:
        config_data = yaml.safe_load(f)

    trainer = SAMTrainer(config_data)
    trainer.train()

    print('Done!')

if __name__ == '__main__':
    main()
