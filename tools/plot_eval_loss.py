#!/usr/bin/env python3
"""
trainer_state.jsonからtrain_lossとeval_lossのグラフを出力するスクリプト
"""
import os
import json
import glob
import argparse
import matplotlib.pyplot as plt
from pathlib import Path


def load_losses(checkpoint_dir):
    """
    チェックポイントディレクトリから全てのtrainer_state.jsonを読み込み、
    train_lossとeval_lossとstepのペアを抽出する
    
    Args:
        checkpoint_dir: チェックポイントディレクトリのパス
        
    Returns:
        tuple: (train_data, eval_data) それぞれ [(step, loss), ...] のリスト
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
    
    train_data = []
    eval_data = []
    if 'log_history' in data:
        for log_entry in data['log_history']:
            if 'step' in log_entry:
                step = log_entry['step']
                # train lossは'loss'キーに格納されている
                if 'loss' in log_entry and 'eval_loss' not in log_entry:
                    train_data.append((step, log_entry['loss']))
                # eval lossは'eval_loss'キーに格納されている
                if 'eval_loss' in log_entry:
                    eval_data.append((step, log_entry['eval_loss']))
    
    # stepでソート
    train_data.sort(key=lambda x: x[0])
    eval_data.sort(key=lambda x: x[0])
    
    return train_data, eval_data


def plot_losses(train_data, eval_data, output_path):
    """
    train_lossとeval_lossのグラフを描画して保存する
    
    Args:
        train_data: [(step, train_loss), ...] のリスト
        eval_data: [(step, eval_loss), ...] のリスト
        output_path: 出力ファイルのパス
    """
    if not train_data and not eval_data:
        raise ValueError("lossデータが見つかりません")
    
    plt.figure(figsize=(14, 7))
    
    # train lossのプロット
    if train_data:
        train_steps = [x[0] for x in train_data]
        train_losses = [x[1] for x in train_data]
        plt.plot(train_steps, train_losses, marker='o', linestyle='-', markersize=2, 
                linewidth=1.5, label='Train Loss', color='blue', alpha=0.7)
    
    # eval lossのプロット
    if eval_data:
        eval_steps = [x[0] for x in eval_data]
        eval_losses = [x[1] for x in eval_data]
        plt.plot(eval_steps, eval_losses, marker='s', linestyle='-', markersize=3, 
                linewidth=1.5, label='Eval Loss', color='red', alpha=0.8)
    
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Train Loss vs Evaluation Loss over Training Steps', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 統計情報を表示
    info_text = []
    if train_data:
        train_losses = [x[1] for x in train_data]
        train_steps = [x[0] for x in train_data]
        min_train_loss = min(train_losses)
        min_train_step = train_steps[train_losses.index(min_train_loss)]
        info_text.append(f'Min train loss: {min_train_loss:.4f} at step {min_train_step}')
    
    if eval_data:
        eval_losses = [x[1] for x in eval_data]
        eval_steps = [x[0] for x in eval_data]
        min_eval_loss = min(eval_losses)
        min_eval_step = eval_steps[eval_losses.index(min_eval_loss)]
        info_text.append(f'Min eval loss: {min_eval_loss:.4f} at step {min_eval_step}')
    
    if info_text:
        plt.text(0.02, 0.98, '\n'.join(info_text), 
                 transform=plt.gca().transAxes, fontsize=9,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"グラフを保存しました: {output_path}")
    if train_data:
        train_steps = [x[0] for x in train_data]
        train_losses = [x[1] for x in train_data]
        print(f"Train Loss - データポイント数: {len(train_data)}, Step範囲: {train_steps[0]} - {train_steps[-1]}, Loss範囲: {min(train_losses):.4f} - {max(train_losses):.4f}")
    if eval_data:
        eval_steps = [x[0] for x in eval_data]
        eval_losses = [x[1] for x in eval_data]
        print(f"Eval Loss - データポイント数: {len(eval_data)}, Step範囲: {eval_steps[0]} - {eval_steps[-1]}, Loss範囲: {min(eval_losses):.4f} - {max(eval_losses):.4f}")


def main():
    parser = argparse.ArgumentParser(description='trainer_state.jsonからtrain_lossとeval_lossのグラフを出力')
    parser.add_argument('--checkpoint_dir', type=str, 
                       default='./model-bin/mcorec_finetuning_face',
                       help='チェックポイントディレクトリのパス (default: ./model-bin/mcorec_finetuning_face)')
    parser.add_argument('--output', type=str,
                       default='./tools/loss_plot.png',
                       help='出力PNGファイルのパス (default: ./tools/loss_plot.png)')
    
    args = parser.parse_args()
    
    # チェックポイントディレクトリの存在確認
    if not os.path.exists(args.checkpoint_dir):
        raise ValueError(f"チェックポイントディレクトリが見つかりません: {args.checkpoint_dir}")
    
    # 出力ディレクトリの作成
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # train_lossとeval_lossデータの読み込み
    train_data, eval_data = load_losses(args.checkpoint_dir)
    
    # グラフの描画と保存
    plot_losses(train_data, eval_data, args.output)


if __name__ == '__main__':
    main()

