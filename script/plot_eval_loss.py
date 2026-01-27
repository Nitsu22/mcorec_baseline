#!/usr/bin/env python3
"""
trainer_state.json から eval_loss の推移を抽出し、表とグラフで可視化する。
"""
import json
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=str,
                        help="model-bin 以下のモデルディレクトリ（例: model-bin/mcorec_finetuning_2spk_sep_input）")
    parser.add_argument("--out", "-o", type=str, default=None,
                        help="プロット画像の保存先（指定時のみ画像を保存）")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.is_absolute():
        base = Path(__file__).resolve().parent.parent
        model_dir = base / args.model_dir

    # 最新の checkpoint の trainer_state.json を使用（全履歴が含まれる）
    checkpoints = sorted(
        (model_dir / d.name for d in model_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")),
        key=lambda p: int(p.name.replace("checkpoint-", "")),
        reverse=True,
    )
    if not checkpoints:
        print(f"エラー: {model_dir} に checkpoint がありません")
        return 1

    state_path = checkpoints[0] / "trainer_state.json"
    if not state_path.exists():
        print(f"エラー: {state_path} が存在しません")
        return 1

    with open(state_path, "r", encoding="utf-8") as f:
        state = json.load(f)

    log = state.get("log_history", [])
    eval_entries = [e for e in log if "eval_loss" in e]

    if not eval_entries:
        print("eval_loss のエントリが見つかりませんでした")
        return 1

    # step でソート
    eval_entries.sort(key=lambda e: e.get("step", 0))

    steps = [e["step"] for e in eval_entries]
    eval_losses = [e["eval_loss"] for e in eval_entries]
    epochs = [e.get("epoch", 0) for e in eval_entries]

    # 表出力
    print(f"\n=== eval_loss 推移 (trainer_state: {state_path.name}) ===")
    print(f"  best_metric: {state.get('best_metric')} @ step {state.get('best_global_step')}")
    print(f"  best_model: {state.get('best_model_checkpoint', '')}")
    print()
    print(f"  {'step':>8}  {'epoch':>10}  {'eval_loss':>12}")
    print("  " + "-" * 34)
    for s, ep, l in zip(steps, epochs, eval_losses):
        print(f"  {s:>8}  {ep:>10.4f}  {l:>12.4f}")

    # プロット（matplotlib が利用可能な場合のみ）
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(steps, eval_losses, "o-", markersize=4, color="C0")
        ax.set_xlabel("step")
        ax.set_ylabel("eval_loss")
        ax.set_title(f"eval_loss 推移 — {model_dir.name}")
        ax.grid(True, alpha=0.3)
        if args.out:
            outpath = Path(args.out)
            outpath.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(outpath, dpi=120, bbox_inches="tight")
            print(f"\nプロットを保存: {outpath}")
        else:
            default_out = model_dir / "eval_loss_curve.png"
            fig.savefig(default_out, dpi=120, bbox_inches="tight")
            print(f"\nプロットを保存: {default_out}")
        plt.close(fig)
    except Exception as e:
        print(f"\nプロットの作成をスキップしました: {e}")
    return 0

if __name__ == "__main__":
    exit(main())
