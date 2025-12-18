#!/bin/bash

# 環境変数の設定
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=4,5

# 学習実行
torchrun --nproc_per_node 2 script/train_face.py \
    --streaming_dataset \
    --include_mcorec \
    --batch_size 6 \
    --max_steps 400000 \
    --gradient_accumulation_steps 2 \
    --save_steps 2000 \
    --eval_steps 2000 \
    --learning_rate 1e-4 \
    --warmup_steps 4000 \
    --checkpoint_name mcorec_finetuning_face \
    --model_name_or_path ./model-bin/avsr_cocktail \
    --output_dir ./model-bin