# !/bin/bash

# Set environment variables for distributed training
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0

# freeze backbone script
torchrun --nproc_per_node 1 script/train_sep_nitsu.py \
    --streaming_dataset \
    --batch_size 2 \
    --max_steps 400000 \
    --gradient_accumulation_steps 6 \
    --save_steps 2000 \
    --eval_steps 2000 \
    --learning_rate 1e-3 \
    --model_name_or_path ./model-bin-phuong/avsr_cocktail \
    --checkpoint_name mcorec_finetuning_2spk_sep_input \
    --output_dir ./model-bin \
    --data_root /gs/bs/tga-shinoda/phuong/datasets_storage/chime9_dataset/AVYT \
    --mcorec_data_root /gs/bs/tga-shinoda/phuong/datasets_storage/chime9_dataset/MCoRec/processed \
    2>&1 | tee train_sep_nitsu.log
    #--resume_from_checkpoint
    # --model_name_or_path ./model-bin-phuong/avsr_cocktail \
    # --warmup_steps 4000 \
   
