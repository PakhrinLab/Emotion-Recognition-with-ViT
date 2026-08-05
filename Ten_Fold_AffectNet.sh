#!/bin/bash
#SBATCH --job-name=Mason_AffectNet_Train
#SBATCH --output=Mason_AffectNet_Train_output.out
#SBATCH --error=Mason_AffectNet_Train_error.out
#SBATCH --time=4-00:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=volta:4                # <-- CHANGED FROM 8 TO 7 (Matches your hardware)
#SBATCH --cpus-per-task=4            # <-- INCREASING THIS to support image loading (2 CPUs per GPU)
#SBATCH --mem=48G                     # <-- OPTIONAL: Increased RAM to handle 7 data-loaders comfortably

module add python

# 1. Ensure you are in the correct project directory
cd /project/pakhrin/subash

# 2. Run your GPU check script just to be sure
/project/pakhrin/subash/exp_acetylation/bin/python check_gpu_available.py

# 3. To run your Vision-Language Model on ALL 7 GPUs, 
# use 'accelerate launch' instead of plain python:
/project/pakhrin/subash/exp_acetylation/bin/python /project/pakhrin/subash/Ten_Fold_AffectNet.py
