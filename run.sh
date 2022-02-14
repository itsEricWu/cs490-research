#!/bin/bash

sbatch  -A gpu --nodes=1 --gres=gpu:1 -t 24:00:00 main.sh
