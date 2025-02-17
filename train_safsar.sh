#/bin/bash
#PBS -l select=1:ncpus=10:ngpus=4
#PBS -l walltime=24:00:00
#PBS -N safsar
#PBS -q gpu
#PBS -j oe
#PBS -J 0-4

list=(softmax eos gc discriminator)
/applications/sw/miniforge/condabin/conda init bash
source /home/sberti/.bashrc
conda activate fsosar
cd /home/sberti/fsosar
python train.py --model SAFSAR --data SSv2 --os_loss ${list[$PBS_ARRAY_INDEX]}
