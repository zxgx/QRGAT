#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export MKL_THREADING_LAYER=GNU

python -u experiment.py
