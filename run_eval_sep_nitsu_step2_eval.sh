#!/bin/bash

ckpt_no=134000
eval_name=sep_input_step2_lr4_ckpt${ckpt_no}

python script/evaluate.py --session_dir "data-bin/dev/*" --output_dir_name ${eval_name} > eval_${eval_name}
