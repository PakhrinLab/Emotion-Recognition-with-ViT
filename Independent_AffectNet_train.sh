#!/bin/bash
#SBATCH --job-name=Indi_AffectNet_MultiSeed
#SBATCH --output=Indi_AffectNet_MultiSeed_output.out
#SBATCH --error=Indi_AffectNet_MultiSeed_error.out
#SBATCH --time=4-00:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=volta:2                # <-- 2-GPU system
#SBATCH --ntasks=1                    # accelerate spawns the 2 procs itself; keep ntasks=1
#SBATCH --cpus-per-task=8            # ~4 dataloader workers per GPU
#SBATCH --mem=128G                    # generous headroom; earlier jobs were RAM OOM-killed

# 1. Move to the project directory
cd /project/pakhrin/subash

# 2. GPU sanity check
/project/pakhrin/subash/exp_acetylation/bin/python check_gpu_available.py

# 3. Launch 2-GPU DDP with accelerate.
#    The .py loops over 5 seeds internally, so ONE submission runs all 5 trainings
#    sequentially and prints the mean +/- std summary at the end.
/project/pakhrin/subash/exp_acetylation/bin/accelerate launch \
    --num_processes=2 \
    --mixed_precision=fp16 \
    /project/pakhrin/subash/Independent_AffectNet_train.py

# ---- Optional: override the seed list without editing the .py ----
# SEEDS="1,2,3,4,5" /project/pakhrin/subash/exp_acetylation/bin/accelerate launch \
#     --num_processes=2 --mixed_precision=fp16 \
#     /project/pakhrin/subash/Independent_AffectNet_train.py
