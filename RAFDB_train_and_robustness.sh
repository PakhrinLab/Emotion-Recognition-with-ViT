#!/bin/bash
#SBATCH --job-name=RAFDB_robustness_analysis
#SBATCH --output=RAFDB_robustness_analysis_output.out
#SBATCH --error=RAFDB_robustness_analysis_error.out
#SBATCH --time=4-00:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=volta:1                # single GPU
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8            # dataloader workers for augmentation
#SBATCH --mem=128G

cd /project/pakhrin/subash

# GPU sanity check
/project/pakhrin/subash/exp_acetylation/bin/python check_gpu_available.py

# Single-GPU run (train once + robustness analysis, no DDP)
/project/pakhrin/subash/exp_acetylation/bin/python \
    /project/pakhrin/subash/RAFDB_train_and_robustness.py