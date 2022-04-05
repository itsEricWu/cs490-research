#!/bin/bash

#SBATCH --nodes=1 
#SBATCH -A gpu
#SBATCH --gres=gpu:1
#SBATCH --time=2:30:00
# cd $SLURM_SUBMIT_DIR

source /etc/profile.d/modules.sh
module purge
module load anaconda/2020.02-py37
module load use.own
module load conda-env/cs490cuda102-py3.7.6 

for i in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18
do
    python resnet_main.py $i /home/lu677/cs490/cs490-research/Resnet/generated/gaussian
done
