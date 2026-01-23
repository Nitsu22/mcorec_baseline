# !/bin/bash

# Set environment variables for distributed training
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0

# freeze backbone script
torchrun --nproc_per_node 1 script/train_sep.py \
    --streaming_dataset \
    --batch_size 1 \
    --max_steps 100000 \
    --gradient_accumulation_steps 6 \
    --save_steps 2000 \
    --eval_steps 2000 \
    --learning_rate 1e-3 \
    --model_name_or_path ./model-bin-phuong/avsr_cocktail \
    --checkpoint_name mcorec_finetuning_2spk_sep_bs_3_gradacc_6_no_mcorec_lyr1 \
    --output_dir ./model-bin
    #--resume_from_checkpoint
    # --model_name_or_path ./model-bin-phuong/avsr_cocktail \
    # --warmup_steps 4000 \
    # --include_mcorec \

# torchrun --nproc_per_node 1 script/train_sep.py \
#     --streaming_dataset \
#     --include_mcorec \
#     --batch_size 3 \
#     --max_steps 400000 \
#     --gradient_accumulation_steps 16 \
#     --save_steps 2000 \
#     --eval_steps 2000 \
#     --learning_rate 1e-3 \
#     --model_name_or_path ./model-bin-phuong/avsr_cocktail \
#     --checkpoint_name mcorec_finetuning_2spk_sep \
#     --output_dir ./model-bin