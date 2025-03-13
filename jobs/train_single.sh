#!/bin/bash
#PBS -l select=1:ncpus=16:ngpus=4
#PBS -l walltime=24:00:00
#PBS -N SAFSAR_Diving48_discriminator
#PBS -q gpu
#PBS -j oe

# Initialize conda
/applications/sw/miniforge/condabin/conda init bash
source /home/sberti/.bashrc
conda activate fsosar

# Change to the fsosar directory
cd /home/sberti/fsosar

# Run the training script with OS_LOSS set from the environment variable
# Defaults to 'softmax' if OS_LOSS is not defined.
python train.py --model SAFSAR --data Diving48 --os_loss discriminator
