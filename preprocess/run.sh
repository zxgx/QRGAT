#!/bin/bash

#SBATCH --account=liujin2
#SBATCH --partition=fat
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=72:00:00

cd $SLURM_SUBMIT_DIR

#python manual_filter_rel.py

#python -u preprocess_webqsp_step1.py webqsp

#python -u preprocess_step2.py webqsp

# construct question-specific subgraph
#python -u preprocess_step3.py webqsp 0

# wasting time to write multiprocess codes in python
# python -u multi_preprocess_step3.py webqsp 15

# split raw QA dataset, convert entities & relations to ids to compact the final QA datasets
python -u preprocess_step4.py webqsp 16 ../nsm_dataset/webqsp

python -u build_vocab.py webqsp
#python -u test.py
