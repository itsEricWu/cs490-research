#!/bin/bash

#SBATCH --nodes=2
#SBATCH -A gpu
#SBATCH --gres=gpu:1
#SBATCH --time=2:30:00

cd $SLURM_SUBMIT_DIR

source /etc/profile.d/modules.sh
module load learning/conda-5.1.0-py36-gpu
source activate cs490

python merge.py
