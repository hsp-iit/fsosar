# SPDX-FileCopyrightText: 2025 Humanoid Sensing and Perception, Istituto Italiano di Tecnologia
# SPDX-License-Identifier: BSD-3-Clause

#!/bin/bash
#SBATCH --job-name=SAFSAR_Diving48_discriminator
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Initialize conda
source /home/sberti/.bashrc
conda activate fsosar

# Change to the fsosar directory
cd /fastwork/sberti/fsosar

echo "Starting single training job:"
echo "Model: SAFSAR"
echo "Dataset: Diving48" 
echo "Open Set Loss: discriminator"
echo "Job started at: $(date)"
echo "Running on node: $SLURM_NODELIST"
echo "Job ID: $SLURM_JOB_ID"

# Run the training script
python train.py --model SAFSAR --data Diving48 --os_loss discriminator

echo "Job finished at: $(date)"
