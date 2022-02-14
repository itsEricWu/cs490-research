#!/bin/bash

source /etc/profile.d/modules.sh
module load anaconda/5.1.0-py36
source activate cs490dsc
cd $SLURM_SUBMIT_DIR
python main.py
