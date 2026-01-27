#!/bin/bash
#SBATCH --job-name=ftw_sam2
#SBATCH --output=logs/jobs/ftw_sam2%j.out
#SBATCH --error=logs/jobs/ftw_sam2%j.err
#SBATCH --time=0-15:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gmuhawenayo@asu.edu

# Activate environment
source ~/.bashrc
mamba activate ftw



cd /u/gmuhawenayo/projects/PRUE-CVPR/ftw-prue

python -m ftw_tools.cli model fit --config sam2_ftw/config_sam_rebuttal.yaml