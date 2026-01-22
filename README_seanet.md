# SEANet + AV-Hubert (MuSE Visual Frontend) Integration

This document describes how to run AV-Hubert fine-tuning with SEANet as an online front-end, using MuSE VisualFrontend features cached on disk.

## Overview

When `--use_seanet` is enabled:
1) audio waveform (after AddNoise/AddMultiSpk) + MuSE visual features -> SEANet
2) SEANet output waveform -> differentiable log-fbank + stack -> AV-Hubert
3) loss is AV-Hubert only (no SEANet loss)

SEANet can be applied to all datasets or only MCoRec, controlled by `--seanet_scope`.

## Precompute MuSE Features (required)

This is mandatory because training expects cached MuSE features and will stop if a cache entry is missing.

Example (all datasets, include MCoRec):
```bash
python script/precompute_muse_features.py \
  --cache_dir data-bin/cache \
  --muse_cache_dir data-bin/cache/muse_lip \
  --include_mcorec \
  --muse_scope all \
  --visual_frontend_ckpt ../MuSE/pretrain_networks/visual_frontend.pt \
  --device cuda
```

Example (MCoRec only):
```bash
python script/precompute_muse_features.py \
  --cache_dir data-bin/cache \
  --muse_cache_dir data-bin/cache/muse_lip \
  --include_mcorec \
  --muse_scope mcorec \
  --visual_frontend_ckpt ../MuSE/pretrain_networks/visual_frontend.pt \
  --device cuda
```

Notes:
- `--streaming_dataset` is not supported for SEANet. Use `streaming=False`.
- Cache entries are stored under `data-bin/cache/muse_lip/<sha1[:2]>/<sha1>.npy`.
- An index is appended to `data-bin/cache/muse_lip/index.jsonl`.

## Training

Example (SEANet on all datasets):
```bash
torchrun --nproc_per_node 2 script/train.py \
  --include_mcorec \
  --use_seanet \
  --seanet_scope all \
  --seanet_checkpoint /net/midgar/work/nitsu/work/chime9/SEANet/exps/seanet/model/model_0147.model \
  --muse_cache_dir data-bin/cache/muse_lip
```

Example (SEANet on MCoRec only):
```bash
torchrun --nproc_per_node 2 script/train.py \
  --include_mcorec \
  --use_seanet \
  --seanet_scope mcorec \
  --seanet_checkpoint /net/midgar/work/nitsu/work/chime9/SEANet/exps/seanet/model/model_0147.model \
  --muse_cache_dir data-bin/cache/muse_lip
```

Freeze options:
- `--freeze_seanet` to keep SEANet frozen
- `--freeze_avhubert` to keep AV-Hubert frozen

## Failure Behavior

The pipeline stops with an error if:
- MuSE cache entry is missing
- MuSE feature length does not match video length
- Audio length does not match expected `video_len * 640`
- `--use_seanet` is combined with `--streaming_dataset`

## Key Files

- `src/muse/visual_frontend_cache.py`: MuSE cache + VisualFrontend extractor
- `src/seanet_wrapper.py`: SEANet loader for `model_0147.model`
- `src/avhubert_avsr/avhubert_avsr_seanet.py`: AV-Hubert wrapper that applies SEANet
- `src/dataset/avhubert_dataset.py`: SEANet data collator and wave transform
- `script/precompute_muse_features.py`: MuSE cache generation
- `script/train.py`: SEANet flags and wiring

## Requirements

Install from `requirements_all.txt` to get a single pinned environment for MuSE (cold), SEANet, and AV-Hubert.
