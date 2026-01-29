#!/bin/bash

# Set environment variables for distributed training
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0

torchrun --nproc_per_node 1 --rdzv-endpoint localhost:29400 script/train_sep_ft_mcorec_step3.py \
    --streaming_dataset \
    --include_mcorec \
    --batch_size 3 \
    --max_steps 100000 \
    --gradient_accumulation_steps 8 \
    --save_steps 250 \
    --eval_steps 250 \
    --warmup_steps 0 \
    --learning_rate 4e-5 \
    --model_name_or_path ./model-bin/mcorec_finetuning_2spk_sep_input_step2_lr4/checkpoint-112000 \
    --checkpoint_name mcorec_finetuning_2spk_sep_input_step3_lr4 \
    --output_dir ./model-bin
