#!/bin/bash

#SBATCH --nodes=1 
#SBATCH -A gpu
#SBATCH --gres=gpu:1

source /etc/profile.d/modules.sh
module load learning/conda-5.1.0-py36-gpu
source activate cs490
cd $SLURM_SUBMIT_DIR
python main.py
