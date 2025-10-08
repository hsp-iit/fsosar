#!/bin/bash
# This script is called by submit_batch_slurm.sh
# All SLURM parameters are controlled by the submission script

# Initialize conda
source /home/sberti/.bashrc
conda activate fsosar_gnode

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
