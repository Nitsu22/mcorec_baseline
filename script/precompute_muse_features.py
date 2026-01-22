import argparse
import json
import os
import sys
import time

import datasets
import torch

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))

from src.muse.visual_frontend_cache import MuSEFeatureCache, VisualFrontendExtractor, _as_str


def _iter_splits(ds, dataset_name, split_names):
    for split in split_names:
        if split not in ds:
            continue
        for sample in ds[split]:
            yield dataset_name, split, sample


def load_datasets(cache_dir, include_mcorec, muse_scope):
    if muse_scope == "mcorec":
        if not include_mcorec:
            raise ValueError("--muse_scope mcorec requires --include_mcorec")
        mcorec_dataset = datasets.load_dataset(
            "MCoRecChallenge/MCoRec", streaming=False, cache_dir=cache_dir
        ).remove_columns(["__key__", "__url__"])
        return None, None, None, None, mcorec_dataset

    lrs2 = datasets.load_dataset("nguyenvulebinh/AVYT", "lrs2", streaming=False, cache_dir=cache_dir).remove_columns(
        ["__key__", "__url__"]
    )
    vox2 = datasets.load_dataset("nguyenvulebinh/AVYT", "vox2", streaming=False, cache_dir=cache_dir).remove_columns(
        ["__key__", "__url__"]
    )
    avyt = datasets.load_dataset("nguyenvulebinh/AVYT", "avyt", streaming=False, cache_dir=cache_dir).remove_columns(
        ["__key__", "__url__"]
    )
    avyt_mix = datasets.load_dataset("nguyenvulebinh/AVYT", "avyt-mix", streaming=False, cache_dir=cache_dir).remove_columns(
        ["__key__", "__url__"]
    )
    mcorec_dataset = None
    if include_mcorec:
        mcorec_dataset = datasets.load_dataset("MCoRecChallenge/MCoRec", streaming=False, cache_dir=cache_dir).remove_columns(
            ["__key__", "__url__"]
        )
    return lrs2, vox2, avyt, avyt_mix, mcorec_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, default="data-bin/cache")
    parser.add_argument(
        "--muse_cache_dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "data-bin/cache/muse_lip"),
    )
    parser.add_argument("--include_mcorec", action="store_true", default=False)
    parser.add_argument("--muse_scope", choices=["all", "mcorec"], default="all")
    parser.add_argument(
        "--visual_frontend_ckpt",
        type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "MuSE", "pretrain_networks", "visual_frontend.pt"),
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_samples", type=int, default=0)
    args = parser.parse_args()

    if args.muse_scope == "mcorec" and not args.include_mcorec:
        raise ValueError("--muse_scope mcorec requires --include_mcorec")

    cache_dir = args.cache_dir
    muse_cache = MuSEFeatureCache(args.muse_cache_dir)
    extractor = VisualFrontendExtractor(checkpoint_path=args.visual_frontend_ckpt, device=args.device)

    lrs2, vox2, avyt, avyt_mix, mcorec_dataset = load_datasets(
        cache_dir, include_mcorec=args.include_mcorec, muse_scope=args.muse_scope
    )

    datasets_to_process = []
    if args.muse_scope == "all":
        datasets_to_process.extend(
            [
                ("lrs2", lrs2, ["train", "pretrain", "valid", "test_snr_0_interferer_2"]),
                ("vox2", vox2, ["dev"]),
                ("avyt", avyt, ["talking", "silent"]),
                ("avyt-mix", avyt_mix, ["train", "test"]),
            ]
        )
        if args.include_mcorec and mcorec_dataset is not None:
            datasets_to_process.append(("mcorec", mcorec_dataset, ["train", "valid"]))
    else:
        if mcorec_dataset is None:
            raise ValueError("MCoRec dataset not loaded")
        datasets_to_process.append(("mcorec", mcorec_dataset, ["train", "valid"]))

    os.makedirs(args.muse_cache_dir, exist_ok=True)
    index_path = os.path.join(args.muse_cache_dir, "index.jsonl")
    processed = 0
    started = time.time()

    with open(index_path, "a", encoding="utf-8") as index_file:
        for dataset_name, ds, splits in datasets_to_process:
            for _, split, sample in _iter_splits(ds, dataset_name, splits):
                if args.max_samples and processed >= args.max_samples:
                    break
                video_path = _as_str(sample["video"])
                start_time = sample.get("start_time")
                end_time = sample.get("end_time")

                if muse_cache.exists(video_path, start_time, end_time):
                    processed += 1
                    if processed % 100 == 0:
                        elapsed = time.time() - started
                        print(f"[skip] {processed} samples ({elapsed:.1f}s)")
                    continue

                feats = extractor.extract(video_path, start_time=start_time, end_time=end_time)
                cache_path = muse_cache.save(video_path, start_time, end_time, feats)

                record = {
                    "dataset": dataset_name,
                    "split": split,
                    "video": video_path,
                    "start_time": start_time,
                    "end_time": end_time,
                    "cache_path": cache_path,
                    "num_frames": int(feats.shape[0]),
                }
                index_file.write(json.dumps(record, ensure_ascii=True) + "\n")
                processed += 1
                if processed % 50 == 0:
                    elapsed = time.time() - started
                    print(f"[done] {processed} samples ({elapsed:.1f}s)")

    elapsed = time.time() - started
    print(f"Completed {processed} samples in {elapsed:.1f}s. Cache: {args.muse_cache_dir}")


if __name__ == "__main__":
    main()
