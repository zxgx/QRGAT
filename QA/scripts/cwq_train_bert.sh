#!/bin/bash

#SBATCH --account=liujin2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00

# check the result for cwq
python -u scripts/cwq_experiment_bert.py
