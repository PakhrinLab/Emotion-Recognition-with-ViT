#!/bin/bash
#SBATCH --job-name=RAF_DB_vit_Subash_strong_cv10fold_
#SBATCH --output=RAF_DB_vit_Subash_strong_cv10fold__output.out
#SBATCH --error=RAF_DB_vit_Subash_strong_cv10fold__error.out
#SBATCH --time=4-00:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=volta:1                # 1-GPU system
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4             # ~4 dataloader workers for the single GPU
#SBATCH --mem=64G
cd /project/pakhrin/subash
# GPU sanity check
/project/pakhrin/subash/exp_acetylation/bin/python check_gpu_available.py
# single-GPU run
/project/pakhrin/subash/exp_acetylation/bin/accelerate launch \
    --num_processes=1 \
    --mixed_precision=fp16 \
    /project/pakhrin/subash/RAF_DB_vit_cv10fold.py
