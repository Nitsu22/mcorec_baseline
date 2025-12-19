#!/bin/bash

python script/inference.py --model_type avsr_cocktail --session_dir "data-bin/dev/*" \
    --checkpoint_path ./model-bin/avsr_cocktail \
    --beam_size 3 --max_length 15 \
    --output_dir_name output_avsr_cocktail

python script/evaluate.py --session_dir "data-bin/dev/*" --output_dir_name output_avsr_cocktail
