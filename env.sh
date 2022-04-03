#!bin/bash

source /etc/profile.d/modules.sh
module load learning/conda-5.1.0-py36-gpu
module load ml-toolkit-gpu/pytorch/1.4.0
module load cuda
source activate cs490
