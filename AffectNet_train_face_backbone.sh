#!/bin/bash
#SBATCH --job-name=AffectNet_train_face_backbone
#SBATCH --output=AffectNet_train_face_backbone_output.out
#SBATCH --error=AffectNet_train_face_backbone_error.out
#SBATCH --time=4-00:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=volta:2
#SBATCH --ntasks=1                    # accelerate spawns the 4 procs itself; keep ntasks=1
#SBATCH --cpus-per-task=8            # ~4 dataloader workers per GPU
#SBATCH --mem=64G

cd /project/pakhrin/subash

# GPU sanity check
/project/pakhrin/subash/exp_acetylation/bin/python check_gpu_available.py

# Launch with accelerate for real DDP across all 4 GPUs
/project/pakhrin/subash/exp_acetylation/bin/accelerate launch \
    --num_processes=2 \
    --mixed_precision=fp16 \
    /project/pakhrin/subash/AffectNet_train_face_backbone.py