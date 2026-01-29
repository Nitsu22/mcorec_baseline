#!/bin/bash

ckpt_no=134000
eval_name=sep_input_step2_lr4_ckpt${ckpt_no}

CUDA_VISIBLE_DEVICES=0 python script/inference_sep_nitsu_step2.py --model_type avsr_cocktail --session_dir "data-bin/dev/*" \
  --checkpoint_path ./model-bin/mcorec_finetuning_2spk_sep_input_step2_lr4/checkpoint-${ckpt_no} \
  --beam_size 3 --max_length 15 \
  --output_dir_name ${eval_name}

python script/evaluate.py --session_dir "data-bin/dev/*" --output_dir_name ${eval_name} > eval_${eval_name}
