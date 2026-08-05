#!/bin/bash
#SBATCH --job-name=PA_AffectNet_vit_token_head
#SBATCH --output=PA_AffectNet_vit_token_head_output.out
#SBATCH --error=PA_AffectNet_vit_token_head_error.out
#SBATCH --time=4-00:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=volta:2
#SBATCH --ntasks=1                    # accelerate spawns the 4 procs itself; keep ntasks=1
#SBATCH --cpus-per-task=8            # ~4 dataloader workers per GPU
#SBATCH --mem=128G                    # RAISED 64G->128G (jobs were RAM OOM-killed)

cd /project/pakhrin/subash

# GPU sanity check
/project/pakhrin/subash/exp_acetylation/bin/python check_gpu_available.py

# Launch with accelerate for real DDP across all 4 GPUs
/project/pakhrin/subash/exp_acetylation/bin/accelerate launch \
    --num_processes=2 \
    --mixed_precision=fp16 \
    /project/pakhrin/subash/PA_AffectNet_vit_token_head.py