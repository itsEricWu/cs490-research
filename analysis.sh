#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=2:00:00
# cd $SLURM_SUBMIT_DIR

source /etc/profile.d/modules.sh
module load learning/conda-5.1.0-py36-gpu
source activate cs490
for i in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18
do
    python analysis.py $i LeNet/speckle
done
