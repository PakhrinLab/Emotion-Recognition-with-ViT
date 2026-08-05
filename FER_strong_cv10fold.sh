#!/bin/bash
#SBATCH --job-name=FER_strong_cv10fold
#SBATCH --output=FER_strong_cv10fold_output.out
#SBATCH --error=FER_strong_cv10fold_error.out
#SBATCH --time=4-00:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=volta:2                # 2-GPU system
#SBATCH --ntasks=1                    # accelerate spawns the 2 procs itself
#SBATCH --cpus-per-task=8            # ~4 dataloader workers per GPU (augmentation is CPU-heavy)
#SBATCH --mem=128G

cd /project/pakhrin/subash

# GPU sanity check
/project/pakhrin/subash/exp_acetylation/bin/python check_gpu_available.py

# 2-GPU DDP
/project/pakhrin/subash/exp_acetylation/bin/accelerate launch \
    --num_processes=2 \
    --mixed_precision=fp16 \
    /project/pakhrin/subash/FER_strong_cv10fold.py
