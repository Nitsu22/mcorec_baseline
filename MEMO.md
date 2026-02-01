# Training Steps (Sep-All + SI-SNR)

## Environment
```bash
source ~/.bashrc
conda activate mcorec_sep
```

## Stage 1 (AV-TSE / SEANet only)
```bash
bash run_train_sep_nitsu_all_stage1.sh
```

## Stage 2 (Joint multitask: ASR + SEANet)
```bash
bash run_train_sep_nitsu_all_stage2.sh
```

## Stage 3 (ASR-only fine-tuning, SEANet frozen)
```bash
bash run_train_sep_nitsu_all_stage3.sh
```

Notes:
- Each run script is configured for 4 GPUs with effective batch size 24.
- Stage2 expects the Stage1 checkpoint at:
  ./model-bin/sep_all_sisnr_stage1/checkpoint-100000
- Stage3 expects the Stage2 checkpoint at:
  ./model-bin/sep_all_sisnr_stage2/checkpoint-250000
