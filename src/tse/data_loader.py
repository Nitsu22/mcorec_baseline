import numpy, os, random, soundfile, torch, json
import re
import itertools
from collections import defaultdict

def init_loader(args):
	args.trainLoader = torch.utils.data.DataLoader(train_loader(set_type = 'train', **vars(args)), batch_size = args.batch_size, shuffle = True, num_workers = args.n_cpu, drop_last = True)
	args.infLoader  = torch.utils.data.DataLoader(inf_loader(args.data_list), batch_size = 1, shuffle = False, num_workers = 0, drop_last = False)
	return args

def load_audio_all(path):
	if not os.path.isfile(path):
		print('NOT FILE', path)
		raise FileNotFoundError(f"Audio file not found: {path}")
	audio, _ = soundfile.read(path)
	return audio

def load_visual_all(path):
	if not os.path.isfile(path):
		print('NOT FILE', path)
		raise FileNotFoundError(f"Visual file not found: {path}")
	face = numpy.load(path)
	return face

class train_loader(object):
	def __init__(self, set_type, data_list, visual_path, audio_path, length, musan_path, **kwargs):
		self.length = length
		
		# Read data_list and filter central_crops only
		lines = open(data_list).read().splitlines()
		central_crops_paths = []
		for line in lines:
			line = line.replace('\n', '').strip()
			if 'central_crops' in line and line.endswith('.wav'):
				central_crops_paths.append(line)
		
		# Group by session
		self.session_dict = defaultdict(list)
		for path in central_crops_paths:
			# Extract session ID from path
			session_match = re.search(r'session_\d+', path)
			if session_match:
				session_id = session_match.group()
				self.session_dict[session_id].append(path)
		
		# Create all possible session pairs (56 * 55 / 2 = 1,540 pairs)
		session_list = list(self.session_dict.keys())
		self.pair_list = []
		for session1, session2 in itertools.combinations(session_list, 2):
			# Get track_00_lip.av.wav paths from each session
			session1_paths = [p for p in self.session_dict[session1] if 'track_00_lip.av.wav' in p]
			session2_paths = [p for p in self.session_dict[session2] if 'track_00_lip.av.wav' in p]
			
			if len(session1_paths) > 0 and len(session2_paths) > 0:
				# Randomly select one path from each session
				path1 = random.choice(session1_paths)
				path2 = random.choice(session2_paths)
				
				# Extract session IDs and metadata paths
				parts1 = path1.split('/')
				parts2 = path2.split('/')
				session_idx1 = next((i for i, p in enumerate(parts1) if p.startswith('session_')), None)
				session_idx2 = next((i for i, p in enumerate(parts2) if p.startswith('session_')), None)
				
				if session_idx1 is None or session_idx2 is None:
					continue
				
				metadata_path1 = '/'.join(parts1[:session_idx1 + 1]) + '/metadata.json'
				metadata_path2 = '/'.join(parts2[:session_idx2 + 1]) + '/metadata.json'
				
				# Load metadata to count speakers
				try:
					with open(metadata_path1, 'r') as file:
						metadata_dict1 = json.load(file)
					with open(metadata_path2, 'r') as file:
						metadata_dict2 = json.load(file)
					
					# Get all speakers from each session
					spk_ids1 = [spk_id for spk_id in metadata_dict1.keys() if spk_id.startswith('spk_')]
					spk_ids2 = [spk_id for spk_id in metadata_dict2.keys() if spk_id.startswith('spk_')]
					
					# Check if total speaker count is <= 8
					total_speakers = len(spk_ids1) + len(spk_ids2)
					if total_speakers <= 8:
						self.pair_list.append((path1, path2))
				except (FileNotFoundError, json.JSONDecodeError):
					# Skip if metadata file is missing or invalid
					continue
		
		print(f"Created {len(self.pair_list)} pairs from {len(session_list)} sessions (all combinations, filtered to <=8 speakers per pair)")

	def __getitem__(self, index):         
		# Get pair of paths
		audio_path1, audio_path2 = self.pair_list[index]
		
		# Extract session IDs and metadata paths
		parts1 = audio_path1.split('/')
		parts2 = audio_path2.split('/')
		session_idx1 = next((i for i, p in enumerate(parts1) if p.startswith('session_')), None)
		session_idx2 = next((i for i, p in enumerate(parts2) if p.startswith('session_')), None)
		
		if session_idx1 is None or session_idx2 is None:
			raise ValueError(f"Could not find session in paths: {audio_path1}, {audio_path2}")
		
		session1 = parts1[session_idx1]
		session2 = parts2[session_idx2]
		metadata_path1 = '/'.join(parts1[:session_idx1 + 1]) + '/metadata.json'
		metadata_path2 = '/'.join(parts2[:session_idx2 + 1]) + '/metadata.json'
		
		with open(metadata_path1, 'r') as file:
			metadata_dict1 = json.load(file)
		with open(metadata_path2, 'r') as file:
			metadata_dict2 = json.load(file)
		
		# Extract track numbers
		track_match1 = re.search(r'track_\d+', audio_path1)
		track_match2 = re.search(r'track_\d+', audio_path2)
		track1 = track_match1.group() if track_match1 else 'track_00'
		track2 = track_match2.group() if track_match2 else 'track_00'
		
		# Get all speakers from each session
		spk_ids1 = [spk_id for spk_id in metadata_dict1.keys() if spk_id.startswith('spk_')]
		spk_ids2 = [spk_id for spk_id in metadata_dict2.keys() if spk_id.startswith('spk_')]
		
		# central_crops track files start from 0.0
		vid_start1 = vid_start2 = 0.0
		
		# Get visual file lengths (in seconds)
		visual_path1 = audio_path1.replace('.wav', '.npy')
		visual_path2 = audio_path2.replace('.wav', '.npy')
		face1_len_sec = len(load_visual_all(visual_path1)) / 25.0 if os.path.isfile(visual_path1) else 28.0
		face2_len_sec = len(load_visual_all(visual_path2)) / 25.0 if os.path.isfile(visual_path2) else 28.0
		
		# Get base paths for speaker lip crops
		base_path1 = '/'.join(audio_path1.split('/')[:audio_path1.split('/').index(session1) + 1])
		base_path2 = '/'.join(audio_path2.split('/')[:audio_path2.split('/').index(session2) + 1])
		
		# Collect all lip crop file lengths for session1
		all_lip_crop_lengths1 = [face1_len_sec]
		for spk_id in spk_ids1:
			lip_path = os.path.join(base_path1, 'speakers', spk_id, 'central_crops', f'{track1}_lip.av.npy')
			if os.path.isfile(lip_path):
				all_lip_crop_lengths1.append(len(load_visual_all(lip_path)) / 25.0)
		
		all_length1 = max(self.length, min(min(all_lip_crop_lengths1), 28.0))
		
		# Collect all lip crop file lengths for session2
		all_lip_crop_lengths2 = [face2_len_sec]
		for spk_id in spk_ids2:
			lip_path = os.path.join(base_path2, 'speakers', spk_id, 'central_crops', f'{track2}_lip.av.npy')
			if os.path.isfile(lip_path):
				all_lip_crop_lengths2.append(len(load_visual_all(lip_path)) / 25.0)
		
		all_length2 = max(self.length, min(min(all_lip_crop_lengths2), 28.0))
		
		# Use independent random timestamps for each session
		start_timestamp1 = int(random.random() * (all_length1 - self.length)) + 1
		start_timestamp2 = int(random.random() * (all_length2 - self.length)) + 1
		start_face1 = int(start_timestamp1 * 25)
		start_face2 = int(start_timestamp2 * 25)
		start_audio1 = start_face1 * 640
		start_audio2 = start_face2 * 640
		
		# Load mixture1 and mixture2
		audio1 = load_audio_all(path = audio_path1)
		audio2 = load_audio_all(path = audio_path2)
		audio1_len = len(audio1)
		audio2_len = len(audio2)
		required_audio_len = int(self.length * 16000)
		
		# Range check and segment extraction for mixture1
		if start_audio1 >= audio1_len:
			start_audio1 = max(0, audio1_len - required_audio_len)
			start_face1 = int(start_audio1 / 640)
		
		audio1_segment = audio1[start_audio1:start_audio1 + required_audio_len]
		if len(audio1_segment) < required_audio_len:
			audio1_segment = numpy.concatenate([audio1_segment, numpy.zeros(required_audio_len - len(audio1_segment), dtype=audio1_segment.dtype)])
		
		# Range check and segment extraction for mixture2
		if start_audio2 >= audio2_len:
			start_audio2 = max(0, audio2_len - required_audio_len)
			start_face2 = int(start_audio2 / 640)
		
		audio2_segment = audio2[start_audio2:start_audio2 + required_audio_len]
		if len(audio2_segment) < required_audio_len:
			audio2_segment = numpy.concatenate([audio2_segment, numpy.zeros(required_audio_len - len(audio2_segment), dtype=audio2_segment.dtype)])
		
		if len(audio1_segment) == 0 or len(audio2_segment) == 0:
			raise ValueError(f"Empty audio segment: {audio_path1 if len(audio1_segment) == 0 else audio_path2}")
		
		# Normalize audio
		audio1_max = numpy.max(numpy.abs(audio1_segment))
		audio2_max = numpy.max(numpy.abs(audio2_segment))
		if audio1_max > 0:
			audio1_segment = numpy.divide(audio1_segment, audio1_max)
		if audio2_max > 0:
			audio2_segment = numpy.divide(audio2_segment, audio2_max)
		
		# Get lip crops for all speakers
		required_face_len = int(self.length * 25)
		mixture1_lip_crops = []
		for spk_id in spk_ids1:
			lip_path = os.path.join(base_path1, 'speakers', spk_id, 'central_crops', f'{track1}_lip.av.npy')
			if not os.path.isfile(lip_path):
				continue
			
			face = load_visual_all(lip_path)
			face_len = len(face)
			if start_face1 >= face_len:
				continue
			
			face_segment = face[start_face1:min(start_face1 + required_face_len, face_len)]
			if len(face_segment) == 0:
				continue
			
			if len(face_segment) < required_face_len:
				face_segment = numpy.concatenate([face_segment, numpy.zeros((required_face_len - len(face_segment), face_segment.shape[1]), dtype=face_segment.dtype)], axis=0)
			
			mixture1_lip_crops.append(torch.FloatTensor(face_segment))
		
		mixture2_lip_crops = []
		for spk_id in spk_ids2:
			lip_path = os.path.join(base_path2, 'speakers', spk_id, 'central_crops', f'{track2}_lip.av.npy')
			if not os.path.isfile(lip_path):
				continue
			
			face = load_visual_all(lip_path)
			face_len = len(face)
			if start_face2 >= face_len:
				continue
			
			face_segment = face[start_face2:min(start_face2 + required_face_len, face_len)]
			if len(face_segment) == 0:
				continue
			
			if len(face_segment) < required_face_len:
				face_segment = numpy.concatenate([face_segment, numpy.zeros((required_face_len - len(face_segment), face_segment.shape[1]), dtype=face_segment.dtype)], axis=0)
			
			mixture2_lip_crops.append(torch.FloatTensor(face_segment))
		
		# Select random session3 (different from session1 and session2)
		session_list = list(self.session_dict.keys())
		available_sessions = [s for s in session_list if s != session1 and s != session2]
		if len(available_sessions) == 0:
			raise ValueError(f"No available sessions for session3 (session1={session1}, session2={session2})")
		
		session3 = random.choice(available_sessions)
		
		# Get track_00_lip.av.wav paths from session3
		session3_paths = [p for p in self.session_dict[session3] if 'track_00_lip.av.wav' in p]
		if len(session3_paths) == 0:
			raise ValueError(f"No track_00_lip.av.wav paths found in session3: {session3}")
		
		# Randomly select one path from session3
		audio_path3 = random.choice(session3_paths)
		
		# Extract session3 metadata path
		parts3 = audio_path3.split('/')
		session_idx3 = next((i for i, p in enumerate(parts3) if p.startswith('session_')), None)
		if session_idx3 is None:
			raise ValueError(f"Could not find session in path: {audio_path3}")
		
		metadata_path3 = '/'.join(parts3[:session_idx3 + 1]) + '/metadata.json'
		
		with open(metadata_path3, 'r') as file:
			metadata_dict3 = json.load(file)
		
		# Extract track number for session3 (should be track_00)
		track_match3 = re.search(r'track_\d+', audio_path3)
		track3 = track_match3.group() if track_match3 else 'track_00'
		
		# Get all speakers from session3
		spk_ids3 = [spk_id for spk_id in metadata_dict3.keys() if spk_id.startswith('spk_')]
		
		# Get visual file length for session3
		visual_path3 = audio_path3.replace('.wav', '.npy')
		face3_len_sec = len(load_visual_all(visual_path3)) / 25.0 if os.path.isfile(visual_path3) else 28.0
		
		# Get base path for session3 speaker lip crops
		base_path3 = '/'.join(parts3[:parts3.index(session3) + 1])
		
		# Collect all lip crop file lengths for session3
		all_lip_crop_lengths3 = [face3_len_sec]
		for spk_id in spk_ids3:
			lip_path = os.path.join(base_path3, 'speakers', spk_id, 'central_crops', f'{track3}_lip.av.npy')
			if os.path.isfile(lip_path):
				all_lip_crop_lengths3.append(len(load_visual_all(lip_path)) / 25.0)
		
		all_length3 = max(self.length, min(min(all_lip_crop_lengths3), 28.0))
		
		# Use independent random timestamp for session3
		start_timestamp3 = int(random.random() * (all_length3 - self.length)) + 1
		start_face3 = int(start_timestamp3 * 25)
		
		# Get lip crops for speakers in session3 (max 2 speakers)
		# First, filter speakers with existing lip crop files
		available_spk_ids3 = []
		for spk_id in spk_ids3:
			lip_path = os.path.join(base_path3, 'speakers', spk_id, 'central_crops', f'{track3}_lip.av.npy')
			if os.path.isfile(lip_path):
				available_spk_ids3.append(spk_id)
		
		# Randomly select up to 2 speakers
		if len(available_spk_ids3) > 2:
			selected_spk_ids3 = random.sample(available_spk_ids3, 2)
		else:
			selected_spk_ids3 = available_spk_ids3
		
		# Get lip crops for selected speakers
		mixture3_lip_crops = []
		for spk_id in selected_spk_ids3:
			lip_path = os.path.join(base_path3, 'speakers', spk_id, 'central_crops', f'{track3}_lip.av.npy')
			if not os.path.isfile(lip_path):
				continue
			
			face = load_visual_all(lip_path)
			face_len = len(face)
			if start_face3 >= face_len:
				continue
			
			face_segment = face[start_face3:min(start_face3 + required_face_len, face_len)]
			if len(face_segment) == 0:
				continue
			
			if len(face_segment) < required_face_len:
				face_segment = numpy.concatenate([face_segment, numpy.zeros((required_face_len - len(face_segment), face_segment.shape[1]), dtype=face_segment.dtype)], axis=0)
			
			mixture3_lip_crops.append(torch.FloatTensor(face_segment))
		
		return torch.FloatTensor(audio1_segment), mixture1_lip_crops, torch.FloatTensor(audio2_segment), mixture2_lip_crops, mixture3_lip_crops

	def __len__(self):
		return len(self.pair_list)

class inf_loader(object):
	def __init__(self, data_list):
		self.data_list = []
		lines = open(data_list).read().splitlines()
		for line in lines:
			self.data_list.append(line.replace('\n', ''))
		
	def __getitem__(self, index):        
		audio_path = self.data_list[index]
		visual_path = audio_path.replace('.wav', '.npy')
		audio = load_audio_all(path = audio_path)
		face = load_visual_all(path = visual_path)
		scale = numpy.max(numpy.abs(audio))
		if scale > 0:
			audio = numpy.divide(audio, scale)
		
		others = {
			'audio_path': audio_path,
			'scale': scale,
		}
		return torch.FloatTensor(audio), torch.FloatTensor(face), others

	def __len__(self):
		return len(self.data_list)

