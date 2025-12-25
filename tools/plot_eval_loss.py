#!/usr/bin/env python3
"""
trainer_state.jsonからeval_lossのグラフを出力するスクリプト
"""
import os
import json
import glob
import argparse
import matplotlib.pyplot as plt
from pathlib import Path


def load_eval_losses(checkpoint_dir):
    """
    チェックポイントディレクトリから全てのtrainer_state.jsonを読み込み、
    eval_lossとstepのペアを抽出する
    
    Args:
        checkpoint_dir: チェックポイントディレクトリのパス
        
    Returns:
        list: [(step, eval_loss), ...] のリスト
    """
    json_files = glob.glob(os.path.join(checkpoint_dir, 'checkpoint-*/trainer_state.json'))
    
    if not json_files:
        raise ValueError(f"trainer_state.jsonが見つかりません: {checkpoint_dir}")
    
    # 最新のチェックポイントのtrainer_state.jsonを読み込む
    # (最新のチェックポイントに全履歴が含まれている)
    latest_json = max(json_files, key=lambda x: int(x.split('checkpoint-')[-1].split('/')[0]))
    
    print(f"読み込み中: {latest_json}")
    
    with open(latest_json, 'r') as f:
        data = json.load(f)
    
    eval_data = []
    if 'log_history' in data:
        for log_entry in data['log_history']:
            if 'eval_loss' in log_entry and 'step' in log_entry:
                eval_data.append((log_entry['step'], log_entry['eval_loss']))
    
    # stepでソート
    eval_data.sort(key=lambda x: x[0])
    
    return eval_data


def plot_eval_loss(eval_data, output_path):
    """
    eval_lossのグラフを描画して保存する
    
    Args:
        eval_data: [(step, eval_loss), ...] のリスト
        output_path: 出力ファイルのパス
    """
    if not eval_data:
        raise ValueError("eval_lossデータが見つかりません")
    
    steps = [x[0] for x in eval_data]
    eval_losses = [x[1] for x in eval_data]
    
    plt.figure(figsize=(12, 6))
    plt.plot(steps, eval_losses, marker='o', linestyle='-', markersize=3, linewidth=1.5)
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Evaluation Loss', fontsize=12)
    plt.title('Evaluation Loss over Training Steps', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 統計情報を表示
    min_loss = min(eval_losses)
    min_step = steps[eval_losses.index(min_loss)]
    plt.text(0.02, 0.98, f'Min loss: {min_loss:.4f} at step {min_step}', 
             transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"グラフを保存しました: {output_path}")
    print(f"データポイント数: {len(eval_data)}")
    print(f"Step範囲: {steps[0]} - {steps[-1]}")
    print(f"Loss範囲: {min(eval_losses):.4f} - {max(eval_losses):.4f}")


def main():
    parser = argparse.ArgumentParser(description='trainer_state.jsonからeval_lossのグラフを出力')
    parser.add_argument('--checkpoint_dir', type=str, 
                       default='./model-bin/mcorec_finetuning_face',
                       help='チェックポイントディレクトリのパス (default: ./model-bin/mcorec_finetuning)')
    parser.add_argument('--output', type=str,
                       default='./tools/eval_loss_plot.png',
                       help='出力PNGファイルのパス (default: ./tools/eval_loss_plot.png)')
    
    args = parser.parse_args()
    
    # チェックポイントディレクトリの存在確認
    if not os.path.exists(args.checkpoint_dir):
        raise ValueError(f"チェックポイントディレクトリが見つかりません: {args.checkpoint_dir}")
    
    # 出力ディレクトリの作成
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # eval_lossデータの読み込み
    eval_data = load_eval_losses(args.checkpoint_dir)
    
    # グラフの描画と保存
    plot_eval_loss(eval_data, args.output)


if __name__ == '__main__':
    main()

