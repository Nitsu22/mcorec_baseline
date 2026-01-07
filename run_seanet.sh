#!/bin/bash

# SEANet AV-TSE (Audio-Visual Target Speaker Extraction) 学習スクリプト
# MixITを使用した混合音声からのターゲット話者抽出

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1

# GPU設定（環境変数から取得、なければデフォルトで0）
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

printf "SEANet AV-TSE training (CHIME9 MixIT, all session pairs with plusnull, max8 speakers, max2 session3)\n"

python script/train_tse.py \
    --save_path exps/seanet_chime9_mixit_all_half_ft_plusnull_max8_2 \
    --data_list /net/midgar/work/nitsu/work/chime9/data/datalist_train \
    --backbone seanet \
    --n_cpu 4 \
    --length 4 \
    --batch_size 1 \
    --max_epoch 150 \
    --lr 0.00100 \
    --alpha 1.0 \
    --val_step 3 \
    --lr_decay 0.97 \
    --init_model /net/midgar/work/nitsu/work/chime9/SEANet/configs/exps/seanet_chime9_mixit_all_half_ft_plusnull_max8_2/model/model_0147.model