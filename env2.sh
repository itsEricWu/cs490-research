#!bin/bash

source /etc/profile.d/modules.sh
module purge
module load anaconda/2020.02-py37
module load cuda/10.1.168 
module load use.own
module load conda-env/cs490DSCpy37-py3.7.6 
