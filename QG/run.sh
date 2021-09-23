#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

python main.py --seed 1234 --train --eval --levi_graph --bidir_encoder --lr 5e-3

