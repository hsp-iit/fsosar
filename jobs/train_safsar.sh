#!/bin/bash
#PBS -l select=1:ncpus=20:ngpus=4
#PBS -l walltime=24:00:00
#PBS -N safsar
#PBS -q gpu
#PBS -j oe

# Initialize conda
/applications/sw/miniforge/condabin/conda init bash
source /home/sberti/.bashrc
conda activate fsosar

# Change to the fsosar directory
cd /home/sberti/fsosar

# Run the training script with OS_LOSS and DATA set from the environment variables.
# Defaults to 'discriminator' for os_loss and 'SSv2' for data if not defined.
python train.py --model SAFSAR --data "${DATA:-SSv2}" --os_loss "${OS_LOSS:-discriminator}"
