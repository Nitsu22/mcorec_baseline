#!/bin/bash

# NOTE: inference_sep_nitsu_step2.py currently loads sep_nitsu_step2 model.
#       To evaluate sep_all checkpoints, update/create an inference script for sep_all.

ckpt_no=100000
eval_name=sep_all_sisnr_stage3_ckpt${ckpt_no}

CUDA_VISIBLE_DEVICES=0 python script/inference_sep_nitsu_step2.py --model_type avsr_cocktail --session_dir "data-bin/dev/*" \
  --checkpoint_path ./model-bin/sep_all_sisnr_stage3/checkpoint-${ckpt_no} \
  --beam_size 3 --max_length 15 \
  --output_dir_name ${eval_name}

python script/evaluate.py --session_dir "data-bin/dev/*" --output_dir_name ${eval_name} > eval_${eval_name}
