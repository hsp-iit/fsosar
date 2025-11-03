#!/bin/bash
 
#SBATCH --nodes=1
#SBATCH --ntasks=1                        # Number of tasks (processes)
#SBATCH --cpus-per-task=4                 # CPUs per task
#SBATCH --time=24:00:00                   # Max runtime hh:mm:ss
#SBATCH --partition=gpuv                 # Partition name
#SBATCH --gres=gpu:4                      # GPUs requested
 
# activate conda environment
source miniconda3/etc/profile.d/conda.sh
conda activate rgbskl
cd /fastwork/mnasato/fsosar

python train.py \
    --model SAFSAR \
    --data NTURGBD120 \
    --os_loss 