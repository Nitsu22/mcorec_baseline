#!/bin/bash

# Stage 1: AV-TSE (SEANet) training only (Hydrogen)
# Effective batch size = per_device_batch_size * grad_accum * num_gpus = 24

export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

DATA_ROOT=/raid5/nitsu/research/chime-9/data
MCOREC_ROOT=/raid5/nitsu/research/chime-9/data/mcorec

# per_device_batch_size=3, grad_accum=2, num_gpus=4 => effective batch size 24

torchrun --nproc_per_node 4 script/train_sep_nitsu_all.py \
    --streaming_dataset \
    --batch_size 3 \
    --max_steps 100000 \
    --gradient_accumulation_steps 2 \
    --save_steps 2000 \
    --eval_steps 2000 \
    --learning_rate 1e-3 \
    --model_name_or_path ./model-bin-phuong/avsr_cocktail \
    --checkpoint_name sep_all_sisnr_stage1 \
    --output_dir ./model-bin \
    --data_root "${DATA_ROOT}" \
    --mcorec_data_root "${MCOREC_ROOT}"
    # --include_mcorec \
    # --warmup_steps 0
