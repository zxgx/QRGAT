#!/bin/bash

#SBATCH --account=liujin2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00

# check the result for webqsp
python -u scripts/webqsp_experiment_xlnet.py
