#!/bin/bash

#SBATCH --nodes=1 
#SBATCH -A gpu
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
cd $SLURM_SUBMIT_DIR

source /etc/profile.d/modules.sh
module purge
module load anaconda/2020.02-py37
module load use.own
module load conda-env/cs490cuda102-py3.7.6 

python test.py
