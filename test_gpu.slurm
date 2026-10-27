#!/bin/bash
#SBATCH --job-name=gputest
#SBATCH --partition=gpu4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=logs/gputest_%j.out
source ~/miniconda3/etc/profile.d/conda.sh && conda activate esmprobe
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
