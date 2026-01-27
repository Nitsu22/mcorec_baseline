#!/bin/bash
# =============================================================================
# ksrun による 2 ノード並列学習 (funnyv GPU0 + tooru GPU1)
# =============================================================================
#
# 実行手順:
#   1. ssh でログインノード等に接続
#   2. module load ksrun
#   3. cd /net/midgar/work/nitsu/work/chime9/mcorec_baseline
#   4. ksrun run_train_sep_nitsu_ksrun.sh --host funnyv:0 --host tooru:1
#
# 補足:
#   - CUDA_VISIBLE_DEVICES は ksrun が --host で指定した GPU ID に従って
#     自動設定するため、本スクリプトでは設定しません。
#   - 2 GPU 利用のため有効バッチサイズは約 2 倍になります。必要に応じて
#     --batch_size や --gradient_accumulation_steps を調整してください。
#   - conda 環境名 "mcorec" は environment.yml に合わせています。違う場合は要変更。
#
# =============================================================================

module load anaconda
conda activate mcorec

cd "$(dirname "$0")"

# 分散学習用の環境変数
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=1

# マルチノード: マスターは funnyv、全ノードで同じポートを使用
export MASTER_ADDR=funnyv
export MASTER_PORT=29500

# ホスト名から NODE_RANK を決定 (funnyv=0=master, tooru=1=slave)
H=$(hostname -s)
if [[ "$H" == "funnyv" ]]; then
  export NODE_RANK=0
elif [[ "$H" == "tooru" ]]; then
  export NODE_RANK=1
else
  echo "ERROR: 未対応のホストです: $H. --host funnyv:0 --host tooru:1 で実行しているか確認してください."
  exit 1
fi

# 2 ノード × 各 1 GPU で torchrun
torchrun --nnodes=2 --nproc_per_node=1 --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
  script/train_sep_nitsu.py \
  --streaming_dataset \
  --batch_size 2 \
  --max_steps 400000 \
  --gradient_accumulation_steps 3 \
  --save_steps 2000 \
  --eval_steps 2000 \
  --learning_rate 1e-3 \
  --model_name_or_path ./model-bin-phuong/avsr_cocktail \
  --checkpoint_name mcorec_finetuning_2spk_sep_input \
  --output_dir ./model-bin \
  2>&1 | tee "train_sep_nitsu_${H}.log"
