#!/bin/bash

#SBATCH --account=liujin2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --output=webqsp_train.log

# check the result for webqsp
python -u main.py --train --eval --dataset webqsp \
    --gat_skip --batch_size 32 --decay_rate 0.5 --direction all \
    --fact_dropout 0.0 --gat_dropout 0.0 --gat_head_dim 25 --gat_head_size 8 \
    --graph_encoder_type NSM --hidden_dim 100 --label_smooth 0.2 \
    --linear_dropout 0.2 --lr 0.001 --num_step 3 --question_dropout 0.3 \
    --relation_dim 200 --weight_decay 1e-05 --word_dim 300 --word_emb_path word_emb.npy