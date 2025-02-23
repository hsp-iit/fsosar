#/bin/bash
#PBS -l select=1:ncpus=4:ngpus=1
#PBS -l walltime=24:00:00
#PBS -N safsar
#PBS -q gpu
#PBS -j oe

/applications/sw/miniforge/condabin/conda init bash
source /home/sberti/.bashrc
conda activate fsosar
cd /home/sberti/fsosar
python train.py --model SAFSAR --data UCF101 --os_loss discriminator
