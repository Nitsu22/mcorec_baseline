#!/bin/bash

# Stage 2: Joint multitask training (ASR + SEANet)
# Effective batch size = per_device_batch_size * grad_accum * num_gpus = 24

export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

# per_device_batch_size=3, grad_accum=2, num_gpus=4 => effective batch size 24

torchrun --nproc_per_node 4 script/train_sep_nitsu_all_step2.py \
    --streaming_dataset \
    --batch_size 3 \
    --max_steps 250000 \
    --gradient_accumulation_steps 2 \
    --save_steps 2000 \
    --eval_steps 2000 \
    --warmup_steps 4000 \
    --learning_rate 1e-4 \
    --model_name_or_path ./model-bin/sep_all_sisnr_stage1/checkpoint-100000 \
    --checkpoint_name sep_all_sisnr_stage2 \
    --output_dir ./model-bin
    # --include_mcorec
