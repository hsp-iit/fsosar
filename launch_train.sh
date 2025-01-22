#/bin/bash
#PBS -l select=1:ncpus=10:ngpus=4
#PBS -l walltime=24:00:00
#PBS -N strm
#PBS -q gpu
#PBS -j oe

source /home/sberti/.bashrc
conda activate fsosar
cd /home/sberti/fsosar
python train.py --model STRM --data SSv2 --os_loss None
