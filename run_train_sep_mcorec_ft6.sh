# !/bin/bash

# Set environment variables for distributed training
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0

torchrun --nproc_per_node 1 --rdzv-endpoint localhost:29400 script/train_sep_ft_mcorec.py \
    --streaming_dataset \
    --include_mcorec \
    --batch_size 3 \
    --max_steps 100000 \
    --gradient_accumulation_steps 8 \
    --save_steps 250 \
    --eval_steps 250 \
    --warmup_steps 0 \
    --learning_rate 4e-5 \
    --model_name_or_path ./model-bin/mcorec_finetuning_2spk_sep_bs_3_gradacc_6_no_mcorec_lyr1_train_all/checkpoint-232000 \
    --checkpoint_name mcorec_finetuning_2spk_sep_bs_3_gradacc_16_lyr1_train_all_232000 \
    --output_dir ./model-bin
    # --resume_from_checkpoint
    # --gradient_accumulation_steps 6 \
    
    # --model_name_or_path ./model-bin-phuong/avsr_cocktail \
    # --warmup_steps 4000 \
    # --model_name_or_path ./model-bin/mcorec_finetuning_2spk_sep_bs_3_gradacc_6_no_mcorec_lyr1_train_all/checkpoint-110000 \
    # /net/fractal/work2/roland/research/mcorec_baseline/model-bin/mcorec_finetuning_2spk_sep_bs_3_gradacc_6_no_mcorec_lyr1_freeze_sep/checkpoint-128000
    

# torchrun --nproc_per_node 1 script/train_sep_ft_mcorec.py \
#     --streaming_dataset \
#     --include_mcorec \
#     --batch_size 3 \
#     --max_steps 400000 \
#     --gradient_accumulation_steps 4 \
#     --save_steps 2000 \
#     --eval_steps 2000 \
#     --warmup_steps 0 \
#     --learning_rate 1e-4 \
#     --model_name_or_path ./model-bin/mcorec_finetuning_2spk_sep_bs_3_gradacc_6_no_mcorec_lyr1_freeze_sep/checkpoint-128000 \
#     --checkpoint_name mcorec_finetuning_2spk_sep_bs_3_gradacc_6_lyr1_freeze_sep_128000 \
#     --output_dir ./model-bin