#!bin/bash

source /etc/profile.d/modules.sh
module purge
module load anaconda/2020.02-py37
# module load cuda/10.1.168 
module load use.own
conda-env-mod create -n cs490cuda102
module load conda-env/cs490cuda102-py3.7.6 
conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=10.2 -c pytorch
pip install pandas scikit-learn matplotlib tqdm
pip install ray ray[tune]
