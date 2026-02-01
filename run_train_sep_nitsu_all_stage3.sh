#!/bin/bash

# Stage 3: ASR-only fine-tuning (SEANet frozen)
# Effective batch size = per_device_batch_size * grad_accum * num_gpus = 24

export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

# per_device_batch_size=3, grad_accum=2, num_gpus=4 => effective batch size 24

torchrun --nproc_per_node 4 script/train_sep_nitsu_all_step3.py \
    --streaming_dataset \
    --include_mcorec \
    --batch_size 3 \
    --max_steps 100000 \
    --gradient_accumulation_steps 2 \
    --save_steps 2000 \
    --eval_steps 2000 \
    --warmup_steps 0 \
    --learning_rate 4e-5 \
    --model_name_or_path ./model-bin/sep_all_sisnr_stage2/checkpoint-250000 \
    --checkpoint_name sep_all_sisnr_stage3 \
    --output_dir ./model-bin
