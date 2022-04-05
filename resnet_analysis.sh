#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=2:00:00
# cd $SLURM_SUBMIT_DIR

source /etc/profile.d/modules.sh
module purge
module load anaconda/2020.02-py37
module load use.own
module load conda-env/cs490cuda102-py3.7.6 


for i in 0
do
    python analysis.py $i /home/lu677/cs490/cs490-research/Resnet/generated/gaussian
done
