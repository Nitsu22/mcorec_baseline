#!/bin/bash

# 環境変数の設定
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 推論実行
python script/inference_face.py \
    --model_type avsr_cocktail \
    --session_dir "data-bin/dev/*" \
    --checkpoint_path ./model-bin/mcorec_finetuning \
    --beam_size 3 \
    --max_length 15 \
    --output_dir_name output_finetuning_face 

# 評価実行
python script/evaluate.py \
    --session_dir "data-bin/dev/*" \
    --output_dir_name output_finetuning_face  

