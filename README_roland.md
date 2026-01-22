# 3-Step Learning Process (Roland版移植)

このドキュメントでは、roland版から移植した3ステップの学習プロセスについて説明します。

## 概要

3ステップの学習プロセスは以下の通りです：

1. **Step 1 (AV-TSE only)**: TF-Locoformerセパレータのみを学習（AVHubertエンコーダ、デコーダ、CTCは凍結）
2. **Step 2 (AV-ASR + AV-TSE Multi-task)**: 全パラメータを学習（ASR損失 + 分離損失）
3. **Step 3 (MCoRec AV-ASR only)**: MCoRecデータセットでAV-ASRのみをファインチューニング（分離損失は使用しない）

## 学習コマンド

### Step 1: AV-TSE only

```bash
bash run_train_sep.sh
```

**主要パラメータ:**
- スクリプト: `script/train_sep.py`
- モデル: `avhubert_avsr_model_sep.py` (分離損失のみ)
- データセット: `avhubert_dataset_sep.py`
- バッチサイズ: 3
- 勾配累積: 6
- 学習率: 1e-3
- 最大ステップ数: 400000
- 出力: `./model-bin/mcorec_finetuning_2spk_sep_bs_3_gradacc_6_no_mcorec_lyr1/`

**凍結パラメータ:**
- AVHubertエンコーダ
- デコーダ
- CTC

**学習パラメータ:**
- TF-Locoformerセパレータ

### Step 2: AV-ASR + AV-TSE Multi-task

```bash
bash run_train_sep3.sh
```

**主要パラメータ:**
- スクリプト: `script/train_sep3.py`
- モデル: `avhubert_avsr_model_sep2.py` (ASR損失 + 分離損失)
- データセット: `avhubert_dataset_sep.py`
- バッチサイズ: 3
- 勾配累積: 6
- 学習率: 1e-4
- ウォームアップステップ: 4000
- 最大ステップ数: 400000
- 入力チェックポイント: Step 1の出力（例: `./model-bin/mcorec_finetuning_2spk_sep_bs_3_gradacc_6_no_mcorec_lyr1/checkpoint-102000`）
- 出力: `./model-bin/mcorec_finetuning_2spk_sep_bs_3_gradacc_6_no_mcorec_lyr1_train_all/`

**学習パラメータ:**
- 全パラメータ（エンコーダ、デコーダ、CTC、セパレータ）

### Step 3: MCoRec AV-ASR only

```bash
bash run_train_sep_mcorec_ft6.sh
```

**主要パラメータ:**
- スクリプト: `script/train_sep_ft_mcorec.py`
- モデル: `avhubert_avsr_model_sep3.py` (ASR損失のみ、分離損失はコメントアウト)
- データセット: `avhubert_dataset.py` (baseline版の既存ファイル)
- バッチサイズ: 3
- 勾配累積: 8
- 学習率: 4e-5
- 最大ステップ数: 100000
- MCoRecデータセット: 含む (`--include_mcorec`)
- 入力チェックポイント: Step 2の出力（例: `./model-bin/mcorec_finetuning_2spk_sep_bs_3_gradacc_6_no_mcorec_lyr1_train_all/checkpoint-232000`）
- 出力: `./model-bin/mcorec_finetuning_2spk_sep_bs_3_gradacc_16_lyr1_train_all_232000/`

**学習パラメータ:**
- 全パラメータ（セパレータは凍結）

## ファイル構成

### 新規作成ファイル

- `src/dataset/avhubert_dataset_sep.py` - Step 1, 2用データセット（分離ラベル対応）
- `src/separator/tf-locoformer.py` - TF-Locoformerセパレータ
- `src/avhubert_avsr/avhubert_avsr_model_sep.py` - Step 1用モデル
- `src/avhubert_avsr/avhubert_avsr_model_sep2.py` - Step 2用モデル
- `src/avhubert_avsr/avhubert_avsr_model_sep3.py` - Step 3用モデル
- `src/nets/backend/e2e_asr_avhubert_sep.py` - Step 1用バックエンド（分離損失のみ）
- `src/nets/backend/e2e_asr_avhubert_sep2.py` - Step 2用バックエンド（ASR + 分離損失）
- `src/nets/backend/e2e_asr_avhubert_sep3.py` - Step 3用バックエンド（ASR損失のみ）
- `script/train_sep.py` - Step 1学習スクリプト
- `script/train_sep3.py` - Step 2学習スクリプト
- `script/train_sep_ft_mcorec.py` - Step 3学習スクリプト
- `run_train_sep.sh` - Step 1実行スクリプト
- `run_train_sep3.sh` - Step 2実行スクリプト
- `run_train_sep_mcorec_ft6.sh` - Step 3実行スクリプト

### 既存ファイル（そのまま使用）

- `src/dataset/avhubert_dataset.py` - Step 3用データセット
- `src/avhubert_avsr/configuration_avhubert_avsr.py` - モデル設定

## 注意事項

- データパスはroland版の絶対パス（`/net/bull/work1/chime-9/...`）を使用
- Step 2とStep 3は、前のステップのチェックポイントを`--model_name_or_path`で指定する必要があります
- チェックポイント名は実行時に適宜変更してください
