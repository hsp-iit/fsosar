#!/bin/bash
#SBATCH --job-name=fsosar_train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Initialize conda
source /home/sberti/.bashrc
conda activate fsosar

# Change to the fsosar directory
cd /fastwork/sberti/fsosar

# Run the training script with MODEL, DATA, and OS_LOSS set from environment variables
# Defaults if not defined
MODEL="${MODEL:-STRM}"
DATA="${DATA:-SSv2}"
OS_LOSS="${OS_LOSS:-discriminator}"

echo "Starting training with:"
echo "Model: $MODEL"
echo "Dataset: $DATA" 
echo "Open Set Loss: $OS_LOSS"
echo "Job started at: $(date)"
echo "Running on node: $SLURM_NODELIST"
echo "Job ID: $SLURM_JOB_ID"

python train.py --model "$MODEL" --data "$DATA" --os_loss "$OS_LOSS"

echo "Job finished at: $(date)"
