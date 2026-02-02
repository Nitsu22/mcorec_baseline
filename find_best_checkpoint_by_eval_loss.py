#!/usr/bin/env python3
"""
指定ディレクトリ配下の checkpoint-* から、eval_loss が最小のチェックポイントを特定する。

使い方:
  python find_best_checkpoint_by_eval_loss.py <ベースディレクトリ>
  python find_best_checkpoint_by_eval_loss.py  # デフォルトでカレントディレクトリ

例:
  python find_best_checkpoint_by_eval_loss.py model-bin/mcorec_finetuning_2spk_sep_test
"""

import argparse
import json
import re
from pathlib import Path
from typing import Optional


def find_checkpoint_dirs(base_path: Path) -> list[tuple[int, Path]]:
    """checkpoint-N 形式のディレクトリを列挙し、(N, パス) のリストで返す。"""
    pattern = re.compile(r"^checkpoint-(\d+)$")
    found = []
    if not base_path.is_dir():
        return found
    for child in base_path.iterdir():
        if not child.is_dir():
            continue
        m = pattern.match(child.name)
        if m:
            found.append((int(m.group(1)), child))
    return sorted(found, key=lambda x: x[0])


def load_trainer_state(checkpoint_path: Path) -> Optional[dict]:
    """trainer_state.json を読み込む。存在しない・不正な場合は None。"""
    path = checkpoint_path / "trainer_state.json"
    if not path.is_file():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def find_min_eval_loss_in_log_history(log_history: list) -> Optional[tuple[float, int]]:
    """
    log_history から 'eval_loss' を持つエントリを探し、
    (最小 eval_loss, その step) を返す。該当がなければ None。
    """
    best_loss = None
    best_step = None
    for entry in log_history:
        if "eval_loss" not in entry:
            continue
        loss = entry["eval_loss"]
        step = entry.get("step")
        if step is None:
            continue
        if best_loss is None or loss < best_loss:
            best_loss = loss
            best_step = step
    if best_loss is None:
        return None
    return (best_loss, best_step)


def find_best_checkpoint(base_path: Path) -> dict:
    """
    ベースパス配下の checkpoint から eval_loss 最小のものを特定する。

    返り値:
      {
        "best_checkpoint_dir": Path,
        "best_step": int,
        "best_eval_loss": float,
        "all_eval_results": [(step, eval_loss), ...],
        "error": Optional[str],
      }
    """
    base_path = Path(base_path).resolve()
    result = {
        "best_checkpoint_dir": None,
        "best_step": None,
        "best_eval_loss": None,
        "all_eval_results": [],
        "error": None,
    }

    checkpoints = find_checkpoint_dirs(base_path)
    if not checkpoints:
        result["error"] = f"checkpoint-* ディレクトリがありません: {base_path}"
        return result

    # 最もステップが大きい（最新）チェックポイントの trainer_state に全履歴がある前提
    latest_step, latest_path = checkpoints[-1]
    state = load_trainer_state(latest_path)
    if state is None:
        result["error"] = f"trainer_state.json を読み込めません: {latest_path}"
        return result

    log_history = state.get("log_history")
    if not log_history:
        result["error"] = "log_history が空です"
        return result

    min_info = find_min_eval_loss_in_log_history(log_history)
    if min_info is None:
        result["error"] = "log_history に eval_loss を持つエントリがありません"
        return result

    best_eval_loss, best_step = min_info
    best_dir = base_path / f"checkpoint-{best_step}"

    # 全 eval 結果を step 順で保持（参考用）
    all_evals = []
    for entry in log_history:
        if "eval_loss" in entry and "step" in entry:
            all_evals.append((entry["step"], entry["eval_loss"]))
    all_evals.sort(key=lambda x: x[0])

    result["best_checkpoint_dir"] = best_dir
    result["best_step"] = best_step
    result["best_eval_loss"] = best_eval_loss
    result["all_eval_results"] = all_evals

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="eval_loss が最小のチェックポイントを特定する"
    )
    parser.add_argument(
        "base_dir",
        nargs="?",
        default=".",
        help="checkpoint-* が含まれるベースディレクトリ（省略時はカレント）",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="ベストチェックポイントのパスのみ出力",
    )
    args = parser.parse_args()

    base = Path(args.base_dir)
    if not base.exists():
        print(f"エラー: ディレクトリが存在しません: {base}", file=__import__("sys").stderr)
        raise SystemExit(1)

    r = find_best_checkpoint(base)
    if r["error"]:
        print(f"エラー: {r['error']}", file=__import__("sys").stderr)
        raise SystemExit(1)

    if args.quiet:
        print(r["best_checkpoint_dir"])
        return

    print(f"ベースディレクトリ: {base.resolve()}")
    print(f"eval_loss 最小のチェックポイント: {r['best_checkpoint_dir']}")
    print(f"  ステップ: {r['best_step']}")
    print(f"  eval_loss: {r['best_eval_loss']}")
    if r["all_eval_results"]:
        print(f"  (eval 記録数: {len(r['all_eval_results'])} 件)")


if __name__ == "__main__":
    main()
