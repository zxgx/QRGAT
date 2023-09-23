
# check the result for CWQ
python -u main.py --eval --dataset CWQ \
    --gat_skip --batch_size 24 --decay_rate 0.5 --direction all \
    --fact_dropout 0.0 --gat_dropout 0.0 --gat_head_dim 25 --gat_head_size 8 \
    --graph_encoder_type NSM --hidden_dim 100 --label_smooth 0.2 \
    --linear_dropout 0.2 --lr 0.001 --num_step 4 --question_dropout 0.3 \
    --relation_dim 200 --weight_decay 1e-05 --word_dim 300 --word_emb_path word_emb.npy \
    --checkpoint cache/nsm_cwq/CWQ-1639151506.pt
