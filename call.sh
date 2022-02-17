#!/bin/bash

#SBATCH --nodes=1 
#SBATCH -A gpu
#SBATCH --gres=gpu:1
# cd $SLURM_SUBMIT_DIR

source main.sh
source analysis.sh
source merge.sh
