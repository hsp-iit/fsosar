#/bin/bash
#PBS -l select=1:ncpus=10:ngpus=1
#PBS -l walltime=24:00:00
#PBS -N fsosar
#PBS -q gpu
#PBS -j oe

/applications/sw/miniforge/condabin/conda init bash
source /home/sberti/.bashrc
conda activate fsosar
cd /home/sberti/fsosar
python train.py --model SAFSAR --data SSv2 --os_loss None
