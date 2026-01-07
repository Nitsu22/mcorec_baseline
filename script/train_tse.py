import os
import sys
os.sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))

import argparse
from src.tse.tools import init_system
from src.tse.trainer import init_trainer
from src.tse.data_loader import init_loader

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

parser = argparse.ArgumentParser(description="Audio-visual target speaker extraction with MixIT.")

parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training and validation (MixIT uses batch_size=1)')
parser.add_argument('--max_epoch', type=int, default=150, help='Maximum number of epochs')
parser.add_argument('--n_cpu', type=int, default=12, help='Number of loader threads')
parser.add_argument('--val_step', type=int, default=3, help='Every [val_step] epochs: Validation, update learning rate and save model')
parser.add_argument('--length', type=float, default=4, help='Training data length')
parser.add_argument('--lr', type=float, default=0.0010, help='Init learning rate')
parser.add_argument("--lr_decay", type=float, default=0.97, help='Learning rate decay every [val_step] epochs')
parser.add_argument("--alpha", type=float, default=1.0, help='Weight for the loss (MixIT uses alpha=1.0)')
parser.add_argument('--init_model', type=str, default="", help='Init model from pretrain')
parser.add_argument('--save_path', type=str, default="", help='Path to save the clean list')
parser.add_argument('--data_list', type=str, default="/net/midgar/work/nitsu/work/chime9/data/datalist_train", help='The path of the training list (should contain central_crops paths)')
parser.add_argument('--visual_path', type=str, default="", help='The path of the lip embs (not used, kept for compatibility)')
parser.add_argument('--audio_path', type=str, default="", help='The path of the clean audio (not used, kept for compatibility)')
parser.add_argument('--musan_path', type=str, default="", help='The path for the musan dataset for augmentation, can ignore if do not use')
parser.add_argument('--backbone', type=str, default="seanet", help='Model backbone (seanet)')
parser.add_argument('--eval', dest='eval', action='store_true', help='Do evaluation only')
    
# AV-HuBERT loss arguments
parser.add_argument('--use_avhubert_loss', action='store_true', default=False, help='Use AV-HuBERT feature loss in addition to SI-SNR loss')
parser.add_argument('--avhubert_model_path', type=str, default=None, help='Path to AV-HuBERT model checkpoint (required if use_avhubert_loss=True)')
parser.add_argument('--avhubert_weight', type=float, default=0.1, help='Weight for AV-HuBERT loss (default: 0.1)')

args = init_system(parser.parse_args())
s = init_trainer(args)
args = init_loader(args)

if args.eval == True:
	s.eval_network('Test', args)
	quit()

while args.epoch < args.max_epoch:
	args = init_loader(args)
	s.train_network(args)
	if args.epoch % args.val_step == 0:
		s.save_parameters(args.model_save_path + "/model_%04d.model"%args.epoch)
		# s.eval_network('Val', args)
	args.epoch += 1

