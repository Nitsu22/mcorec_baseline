import torch, sys, os, time
import torch.nn as nn
from .tools import init_system, cal_SISNR
from .loss import loss_speech
from .model.seanet import seanet
from torch.cuda.amp import autocast, GradScaler
from collections import OrderedDict
import soundfile
import numpy 

def init_trainer(args):
	s = trainer(args)
	args.epoch = 1
	if args.init_model != "":
		print("Model %s loaded from pretrain!"%args.init_model)
		s.load_parameters(args.init_model)
	elif len(args.modelfiles) >= 1:
		print("Model %s loaded from previous state!"%args.modelfiles[-1])
		args.epoch = int(os.path.splitext(os.path.basename(args.modelfiles[-1]))[0][6:]) + 1
		s.load_parameters(args.modelfiles[-1])
	return s

class trainer(nn.Module):
	def __init__(self, args):
		super(trainer, self).__init__()
		if args.backbone == 'seanet':
			self.model       = seanet(256, 40, 64, 128, 100, 6).cuda()
		else:
			raise ValueError(f"Unsupported backbone: {args.backbone}. Only 'seanet' is supported.")
		self.loss_se     = loss_speech().cuda()
		self.optim       = torch.optim.AdamW(self.parameters(), lr = args.lr)
		self.scheduler   = torch.optim.lr_scheduler.StepLR(self.optim, step_size = args.val_step, gamma = args.lr_decay)
		print("Model para number = %.2f"%(sum(param.numel() for param in self.parameters()) / 1e6))
		
	def train_network(self, args):
		# MixIT: batchサイズ1前提で各話者出力を合計
		B, time_start, nloss, nloss1, nloss2, nloss3 = 1, time.time(), 0, 0, 0, 0
		self.train()
		scaler = GradScaler()
		self.scheduler.step(args.epoch - 1)
		lr = self.optim.param_groups[0]['lr']	
		for num, (audio1, mixture1_lip_crops, audio2, mixture2_lip_crops, mixture3_lip_crops) in enumerate(args.trainLoader, start = 1):
			self.zero_grad()
			with autocast():				
				# Ensure batch dimension
				audio1 = audio1.cuda()
				audio2 = audio2.cuda()
				if audio1.dim() == 1:
					audio1 = audio1.unsqueeze(0)
				if audio2.dim() == 1:
					audio2 = audio2.unsqueeze(0)

				mixture_plus = audio1 + audio2  # [1, T]

				mixture1_outputs = []
				for lip in mixture1_lip_crops:
					lip = lip.cuda()
					if lip.dim() == 2:
						lip = lip.unsqueeze(0)  # [1, F, C]
					
					# サイズチェック（フレーム数が0の場合はスキップ）
					if lip.shape[1] == 0:
						print(f'WARNING: Empty lip crop detected in mixture1, shape={lip.shape}, skipping...')
						continue
					
					out_speech, _ = self.model(mixture_plus, lip, M = B)
					mixture1_outputs.append(out_speech[-B:,:])  # [1, T]

				mixture2_outputs = []
				for lip in mixture2_lip_crops:
					lip = lip.cuda()
					if lip.dim() == 2:
						lip = lip.unsqueeze(0)
					
					# サイズチェック（フレーム数が0の場合はスキップ）
					if lip.shape[1] == 0:
						print(f'WARNING: Empty lip crop detected in mixture2, shape={lip.shape}, skipping...')
						continue
					
					out_speech, _ = self.model(mixture_plus, lip, M = B)
					mixture2_outputs.append(out_speech[-B:,:])

				if len(mixture1_outputs) == 0 or len(mixture2_outputs) == 0:
					raise ValueError("Lip crops are empty for mixture1 or mixture2.")

				# メモリ削減のため逐次合計（挙動はstack+sumと同等）
				estim1 = None
				for out in mixture1_outputs:
					estim1 = out if estim1 is None else estim1 + out
				estim2 = None
				for out in mixture2_outputs:
					estim2 = out if estim2 is None else estim2 + out

				loss1 = self.loss_se.forward(estim1, audio1)
				loss2 = self.loss_se.forward(estim2, audio2)
				
				# Calculate loss3 for mixture3_lip_crops
				mixture3_losses = []
				for lip in mixture3_lip_crops:
					lip = lip.cuda()
					if lip.dim() == 2:
						lip = lip.unsqueeze(0)  # [1, F, C]
					
					# サイズチェック（フレーム数が0の場合はスキップ）
					if lip.shape[1] == 0:
						print(f'WARNING: Empty lip crop detected in mixture3, shape={lip.shape}, skipping...')
						continue
					
					out_speech, _ = self.model(mixture_plus, lip, M = B)
					out_speech = out_speech[-B:,:]  # [1, T]
					
					# L2 loss with zero vector (mean squared error)
					l2_loss = torch.mean(out_speech ** 2)
					mixture3_losses.append(l2_loss)
				
				if len(mixture3_losses) == 0:
					loss3 = torch.tensor(0.0, device=audio1.device)
				else:
					# Average of L2 losses
					loss3 = sum(mixture3_losses) / len(mixture3_losses)
				
				loss = (loss1 + loss2) / 2 + loss3

			scaler.scale(loss).backward()
			torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
			scaler.step(self.optim)
			scaler.update()

			nloss += loss.detach().cpu().numpy()
			nloss1 += loss1.detach().cpu().numpy()
			nloss2 += loss2.detach().cpu().numpy()
			nloss3 += loss3.detach().cpu().numpy()
			time_used = time.time() - time_start
			sys.stderr.write("Train: [%2d] %.2f%% (est %.1f mins) Lr: %6f, Loss: %.3f (L1: %.3f, L2: %.3f, L3: %.3f)\r"%\
			(args.epoch, 100 * (num / args.trainLoader.__len__()), time_used * args.trainLoader.__len__() / num / 60, lr, nloss/num, nloss1/num, nloss2/num, nloss3/num))
			sys.stderr.flush()
		sys.stdout.write("\n")

		args.score_file.write("Train: [%2d] %.2f%% (est %.1f mins) Lr: %6f, Loss: %.3f (L1: %.3f, L2: %.3f, L3: %.3f)\r"%\
			(args.epoch, 100 * (num / args.trainLoader.__len__()), time_used * args.trainLoader.__len__() / num / 60, lr, nloss/num, nloss1/num, nloss2/num, nloss3/num))
		args.score_file.flush()
		return

	def eval_network(self, eval_type, args):
		Loader = args.valLoader   if eval_type == 'Val' else args.infLoader
		B      = args.batch_size  if eval_type == 'Val' else 1
		self.eval()
		time_start = time.time()
		for num, (audio, face, others) in enumerate(Loader, start = 1):
			self.zero_grad()
			with torch.no_grad():
				audio, face = audio.cuda(), face.cuda()
				audio_seg = 8
				audio_len = audio.shape[1]
				face_len = face.shape[1]
				output_segments = []
				for seg in range(audio_seg):
					start_audio = seg * (audio_len // audio_seg)
					start_face = seg * (face_len // audio_seg)
					if seg < (audio_seg - 1):
						end_audio = start_audio + (audio_len // audio_seg)
						end_face = start_face + (face_len // audio_seg)
					else:
						end_audio = audio_len
						end_face = face_len
					if args.backbone == 'seanet':
						out_speech, _ = self.model(audio[:, start_audio:end_audio], 
						                           face[:, start_face:end_face, :], B)
						out = out_speech[-B:,:]
					
					output_segments.append(out)
				out_cat = torch.cat(output_segments, dim=1)

			time_used = time.time() - time_start

			audio_path = others['audio_path'][0]
			scale = others['scale'][0]
			out_wav_path = audio_path.replace('.wav', '_clean_ft.wav')
			soundfile.write(out_wav_path, numpy.multiply(out_cat[0].cpu(), scale), 16000)
		return

	def save_parameters(self, path):
		model = OrderedDict(list(self.state_dict().items()))
		torch.save(model, path)

	def load_parameters(self, path):
		selfState = self.state_dict()
		loadedState = torch.load(path)	
		for name, param in loadedState.items():
			origName = name
			if name not in selfState:
				name = 'model.' + name
				if name not in selfState:
					print("%s is not in the model."%origName)
					continue
			if selfState[name].size() != loadedState[origName].size():
				sys.stderr.write("Wrong parameter length: %s, model: %s, loaded: %s"%(origName, selfState[name].size(), loadedState[origName].size()))
				continue
			selfState[name].copy_(param)

