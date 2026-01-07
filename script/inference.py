import os
import sys
import argparse
import json
import math
import glob
import numpy as np
from abc import ABC, abstractmethod
from tqdm import tqdm
import torch
import torchvision
import torchaudio
from collections import OrderedDict

# Add src to path
os.sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))

from src.tokenizer.spm_tokenizer import TextTransform
from src.talking_detector.segmentation import segment_by_asd
from src.cluster.conv_spks import (
    get_speaker_activity_segments, 
    calculate_conversation_scores, 
    cluster_speakers, 
    get_clustering_f1_score
)


class BaseInferenceModel(ABC):
    """Abstract base class for all inference models"""
    
    def __init__(self, checkpoint_path=None, cache_dir=None, beam_size=3):
        self.model = None
        self.text_transform = None
        self.av_data_collator = None
        self.beam_search = None
        self.tokenizer = None
        self.checkpoint_path = checkpoint_path
        self.cache_dir = cache_dir or "./model-bin"
        self.beam_size = beam_size
        
    @abstractmethod
    def load_model(self):
        """Load the specific model architecture"""
        pass
    
    @abstractmethod
    def inference(self, videos, audios, **kwargs):
        """Perform inference on audio-visual data"""
        pass
    
    def get_tokenizer_paths(self):
        """Get paths for tokenizer files"""
        base_dir = os.path.dirname(os.path.dirname(__file__))
        sp_model_path = os.path.join(base_dir, "src/tokenizer/spm/unigram/unigram5000.model")
        dict_path = os.path.join(base_dir, "src/tokenizer/spm/unigram/unigram5000_units.txt")
        return sp_model_path, dict_path


class AVSRCocktailModel(BaseInferenceModel):
    """AVSR Cocktail model implementation"""
    
    def load_model(self):
        from src.dataset.avhubert_dataset import AudioTransform, VideoTransform, DataCollator
        from src.avhubert_avsr.avhubert_avsr_model import AVHubertAVSR, get_beam_search_decoder
        from src.avhubert_avsr.configuration_avhubert_avsr import AVHubertAVSRConfig
        
        # Load text transform
        sp_model_path, dict_path = self.get_tokenizer_paths()
        self.text_transform = TextTransform(
            sp_model_path=sp_model_path,
            dict_path=dict_path,
        )
        
        # Load data collator
        audio_transform = AudioTransform(subset="test")
        video_transform = VideoTransform(subset="test")
        
        self.av_data_collator = DataCollator(
            text_transform=self.text_transform,
            audio_transform=audio_transform,
            video_transform=video_transform,
        )
        
        # Load model
        model_path = self.checkpoint_path or "./model-bin/avsr_cocktail"
        print(f"Loading model from {model_path}")
        avsr_model = AVHubertAVSR.from_pretrained(model_path)
        avsr_model.eval().cuda()
        self.model = avsr_model.avsr
        self.beam_search = get_beam_search_decoder(self.model, self.text_transform.token_list, beam_size=self.beam_size)
    
    def inference(self, videos, audios, **kwargs):
        avhubert_features = self.model.encoder(
            input_features=audios, 
            video=videos,
        )
        audiovisual_feat = avhubert_features.last_hidden_state
        audiovisual_feat = audiovisual_feat.squeeze(0)
        
        nbest_hyps = self.beam_search(audiovisual_feat)
        nbest_hyps = [h.asdict() for h in nbest_hyps[:min(len(nbest_hyps), 1)]]
        predicted_token_id = torch.tensor(list(map(int, nbest_hyps[0]["yseq"][1:])))
        predicted = self.text_transform.post_process(predicted_token_id).replace("<eos>", "")
        return predicted


class AutoAVSRModel(BaseInferenceModel):
    """Auto AVSR model implementation"""
    
    def load_model(self):
        from src.dataset.av_dataset import AudioTransform, VideoTransform, DataCollator
        from src.auto_avsr.configuration_avsr import AutoAVSRConfig
        from src.auto_avsr.avsr_model import AutoAVSR, get_beam_search_decoder
        
        # Load text transform
        sp_model_path, dict_path = self.get_tokenizer_paths()
        self.text_transform = TextTransform(
            sp_model_path=sp_model_path,
            dict_path=dict_path,
        )
        
        # Load data collator
        audio_transform = AudioTransform(subset="test")
        video_transform = VideoTransform(subset="test")
        
        self.av_data_collator = DataCollator(
            text_transform=self.text_transform,
            audio_transform=audio_transform,
            video_transform=video_transform,
        )
        
        # Load model    
        avsr_config = AutoAVSRConfig()
        avsr_model = AutoAVSR(avsr_config)    
        ckpt_path = self.checkpoint_path or "./model-bin/auto_avsr/avsr_trlrwlrs2lrs3vox2avsp_base.pth"
        print(f"Loading model from {ckpt_path}")
        pretrained_weights = torch.load(ckpt_path, weights_only=True)
        avsr_model.avsr.load_state_dict(pretrained_weights)
        avsr_model.eval().cuda()
        self.model = avsr_model.avsr
        self.beam_search = get_beam_search_decoder(self.model, self.text_transform.token_list, beam_size=self.beam_size)
    
    def inference(self, videos, audios, **kwargs):
        video_feat, _ = self.model.encoder(videos, None)
        audio_feat, _ = self.model.aux_encoder(audios, None)
        audiovisual_feat = self.model.fusion(torch.cat((video_feat, audio_feat), dim=-1))
        audiovisual_feat = audiovisual_feat.squeeze(0)

        nbest_hyps = self.beam_search(audiovisual_feat)
        nbest_hyps = [h.asdict() for h in nbest_hyps[:min(len(nbest_hyps), 1)]]
        predicted_token_id = torch.tensor(list(map(int, nbest_hyps[0]["yseq"][1:])))
        predicted = self.text_transform.post_process(predicted_token_id).replace("<eos>", "")
        return predicted


class MuAViCModel(BaseInferenceModel):
    """MuAViC model implementation"""
    
    def load_model(self):
        from src.dataset.avhubert_dataset import AudioTransform, VideoTransform, DataCollator
        from src.avhubert_muavic.avhubert2text import AV2TextForConditionalGeneration
        from transformers import Speech2TextTokenizer
        
        # Load text transform
        sp_model_path, dict_path = self.get_tokenizer_paths()
        self.text_transform = TextTransform(
            sp_model_path=sp_model_path,
            dict_path=dict_path,
        )
        
        # Load data collator
        audio_transform = AudioTransform(subset="test")
        video_transform = VideoTransform(subset="test")
        
        self.av_data_collator = DataCollator(
            text_transform=self.text_transform,
            audio_transform=audio_transform,
            video_transform=video_transform,
        )
        
        # Load model
        model_name = self.checkpoint_path or 'nguyenvulebinh/AV-HuBERT-MuAViC-en'
        print(f"Loading model from {model_name}")
        self.model = AV2TextForConditionalGeneration.from_pretrained(
            model_name, 
            cache_dir=self.cache_dir
        )
        self.tokenizer = Speech2TextTokenizer.from_pretrained(
            model_name, 
            cache_dir=self.cache_dir
        )
        self.model = self.model.cuda().eval()
    
    def inference(self, videos, audios, **kwargs):
        attention_mask = torch.BoolTensor(audios.size(0), audios.size(-1)).fill_(False).cuda()
        output = self.model.generate(
            audios,
            attention_mask=attention_mask,
            video=videos,
        )
        output = self.tokenizer.batch_decode(output, skip_special_tokens=True)[0].upper()
        return output


class InferenceEngine:
    """Main inference engine that handles model selection and processing"""
    
    def __init__(self, model_type: str, checkpoint_path=None, cache_dir=None, beam_size=3, max_length=15, 
                 tse_checkpoint_path=None, use_tse=True):
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        self.cache_dir = cache_dir
        self.beam_size = beam_size
        self.max_length = max_length
        self.tse_checkpoint_path = tse_checkpoint_path
        self.use_tse = use_tse
        self.tse_model = None
        self.model_impl = self._get_model_implementation()
        
    def _get_model_implementation(self) -> BaseInferenceModel:
        """Factory method to get the appropriate model implementation"""
        if self.model_type == "avsr_cocktail":
            return AVSRCocktailModel(self.checkpoint_path, self.cache_dir, self.beam_size)
        elif self.model_type == "auto_avsr":
            return AutoAVSRModel(self.checkpoint_path, self.cache_dir, self.beam_size)
        elif self.model_type == "muavic_en":
            return MuAViCModel(self.checkpoint_path, self.cache_dir, self.beam_size)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def load_model(self):
        """Load the selected model"""
        print(f"Loading {self.model_type} model...")
        self.model_impl.load_model()
        print(f"{self.model_type} model loaded successfully!")
        
        # Load AV-TSE model if enabled
        if self.use_tse:
            self.load_tse_model()
    
    def load_tse_model(self):
        """Load AV-TSE (Target Speaker Extraction) model"""
        from src.tse.model.seanet import seanet
        
        if self.tse_checkpoint_path is None:
            self.tse_checkpoint_path = "/net/midgar/work/nitsu/work/chime9/SEANet/configs/exps/seanet_chime9_mixit_all_half_ft_plusnull_max8_2/model/model_0147.model"
        
        print(f"Loading AV-TSE model from {self.tse_checkpoint_path}")
        self.tse_model = seanet(256, 40, 64, 128, 100, 6).cuda()
        
        # Load checkpoint
        selfState = self.tse_model.state_dict()
        loadedState = torch.load(self.tse_checkpoint_path, map_location='cuda')
        for name, param in loadedState.items():
            origName = name
            if name not in selfState:
                print(f"{origName} is not in the model, skipping...")
                continue
            if selfState[name].size() != loadedState[origName].size():
                print(f"Wrong parameter length: {origName}, model: {selfState[name].size()}, loaded: {loadedState[origName].size()}, skipping...")
                continue
            selfState[name].copy_(param)
        
        self.tse_model.eval()
        print("AV-TSE model loaded successfully!")
    
    def load_lip_embedding(self, npy_path, start_time, end_time):
        """
        Load lip embedding from .npy file and extract segment
        
        Args:
            npy_path: Path to .npy file containing lip embeddings
            start_time: Start time in seconds
            end_time: End time in seconds
        
        Returns:
            lip_embedding: torch.Tensor of shape [1, F, C] where F is number of frames, C is feature dim
        """
        if not os.path.exists(npy_path):
            raise FileNotFoundError(f"Lip embedding file not found: {npy_path}")
        
        # Load full lip embedding
        lip_embedding = np.load(npy_path)  # Shape: [F, C]
        
        # Calculate frame indices (25fps)
        fps = 25.0
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        
        # Extract segment
        lip_segment = lip_embedding[start_frame:end_frame]  # Shape: [F_seg, C]
        
        # Convert to torch tensor and add batch dimension
        lip_tensor = torch.FloatTensor(lip_segment)  # Shape: [F_seg, C]
        lip_tensor = lip_tensor.unsqueeze(0)  # Shape: [1, F_seg, C]
        
        return lip_tensor
    
    def load_central_audio(self, central_video_path, start_time, end_time):
        """
        Load audio segment from central_video.mp4
        
        Args:
            central_video_path: Path to central_video.mp4
            start_time: Start time in seconds
            end_time: End time in seconds
        
        Returns:
            audio: torch.Tensor of shape [1, T] where T is number of samples
        """
        from src.dataset.av_dataset import load_audio
        
        audio = load_audio(central_video_path, start_time=start_time, end_time=end_time)  # Shape: [T, 1]
        audio = audio.transpose(0, 1)  # Shape: [1, T]
        
        return audio
    
    def separate_audio_with_tse(self, mixture_audio, lip_embedding):
        """
        Separate audio using AV-TSE model
        
        Args:
            mixture_audio: torch.Tensor of shape [1, T] (mixed audio)
            lip_embedding: torch.Tensor of shape [1, F, C] (lip embedding)
        
        Returns:
            separated_audio: torch.Tensor of shape [1, T] (separated audio)
        """
        if self.tse_model is None:
            raise RuntimeError("AV-TSE model not loaded. Call load_tse_model() first.")
        
        self.tse_model.eval()
        with torch.no_grad():
            mixture_audio = mixture_audio.cuda()
            lip_embedding = lip_embedding.cuda()
            
            # SEANet推論方式に準拠: 8セグメントに分割して処理
            B = 1
            audio_seg = 8
            audio_len = mixture_audio.shape[1]
            face_len = lip_embedding.shape[1]
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
                
                out_speech, _ = self.tse_model(
                    mixture_audio[:, start_audio:end_audio], 
                    lip_embedding[:, start_face:end_face, :], 
                    B
                )
                out = out_speech[-B:, :]  # Shape: [1, T_seg]
                output_segments.append(out)
            
            # Concatenate segments
            separated_audio = torch.cat(output_segments, dim=1)  # Shape: [1, T]
        
        return separated_audio
    
    def chunk_video(self, video_path, asd_path=None, max_length=15):
        """Split video into chunks for inference"""
        if asd_path is not None:
            with open(asd_path, "r") as f:
                asd = json.load(f)

            # Convert frame numbers to integers and sort them
            frames = sorted([int(f) for f in asd.keys()])
            # Find the minimum frame number to normalize frame indices
            min_frame = min(frames)

            segments_by_frames = segment_by_asd(asd, {
                "max_chunk_size": max_length,  # in seconds
            })
            # Normalize frame indices, for inference, don't care about the actual frame indices
            segments = [((seg[0] - min_frame) / 25, (seg[-1] - min_frame) / 25) for seg in segments_by_frames]

        else:
            # Get video duration
            audio, rate = torchaudio.load(video_path)
            video_duration = audio.shape[1] / rate
            # num chunks
            num_chunks = math.ceil(video_duration / max_length)
            chunk_size = math.ceil(video_duration / num_chunks)
            segments = []
            # Convert to integer steps for range
            steps = int(video_duration * 100)  # Convert to centiseconds for precision
            step_size = int(chunk_size * 100)
            for i in range(0, steps, step_size):
                start_time = i / 100
                end_time = min((i + step_size) / 100, video_duration)
                segments.append((start_time, end_time))
            
        return segments
    
    def format_vtt_timestamp(self, timestamp):
        """Format timestamp for VTT output"""
        hours = int(timestamp // 3600)
        minutes = int((timestamp % 3600) // 60)
        seconds = int(timestamp % 60)
        milliseconds = int((timestamp - int(timestamp)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
    
    def infer_video(self, video_path, asd_path=None, offset=0., desc=None, 
                    central_video_path=None, lip_embedding_path=None):
        """
        Perform inference on a video file
        
        Args:
            video_path: Path to lip video file (used for video features)
            asd_path: Path to ASD JSON file (optional)
            offset: Time offset for output timestamps
            desc: Description for progress bar
            central_video_path: Path to central_video.mp4 (for mixed audio, required if use_tse=True)
            lip_embedding_path: Path to lip embedding .npy file (required if use_tse=True)
        """
        segments = self.chunk_video(video_path, asd_path, max_length=self.max_length)
        segment_output = []
        
        for seg in tqdm(segments, desc="Processing segments" if desc is None else desc, total=len(segments)):
            # Prepare sample for video (always use lip video for visual features)
            sample = {
                "video": video_path,
                "start_time": seg[0],
                "end_time": seg[1],
            }
            sample_features = self.model_impl.av_data_collator([sample])
            videos = sample_features["videos"].cuda()
            video_lengths = sample_features["video_lengths"].cuda()
            
            # Handle audio: use AV-TSE separated audio if enabled, otherwise use original
            if self.use_tse and self.tse_model is not None:
                if central_video_path is None or lip_embedding_path is None:
                    raise ValueError("central_video_path and lip_embedding_path are required when use_tse=True")
                
                # Convert track-relative time to session-absolute time for central_video.mp4
                # seg[0] and seg[1] are relative to track start, so add offset (track_start_time)
                session_start_time = seg[0] + offset
                session_end_time = seg[1] + offset
                
                # Load mixed audio from central_video.mp4 (using session-absolute time)
                mixture_audio = self.load_central_audio(central_video_path, session_start_time, session_end_time)
                
                # Load lip embedding segment (track-relative time)
                lip_embedding = self.load_lip_embedding(lip_embedding_path, seg[0], seg[1])
                
                # Separate audio using AV-TSE
                separated_audio = self.separate_audio_with_tse(mixture_audio, lip_embedding)
                
                # Convert separated audio to format expected by AV-ASR
                # separated_audio shape: [1, T]
                # AV-ASR expects: [T, 1] for processing, then collated to [B, T, 1]
                separated_audio_t = separated_audio.squeeze(0).transpose(0, 1)  # [1, T] -> [T, 1]
                
                # Ensure length matches video
                from src.dataset.av_dataset import cut_or_pad
                video_frames = videos.shape[1]
                rate_ratio = 640  # 1 frame = 640 samples at 25fps, 16000Hz
                expected_audio_len = video_frames * rate_ratio
                separated_audio_t = cut_or_pad(separated_audio_t, expected_audio_len, dim=0)
                
                # Apply audio transform (same as DataCollator does)
                from src.dataset.av_dataset import AudioTransform
                audio_transform = AudioTransform(subset="test")
                separated_audio_t = audio_transform(separated_audio_t)  # Shape: [T, 1]
                
                # Prepare for AV-ASR: add batch dimension and move to GPU
                audios = separated_audio_t.unsqueeze(0).cuda()  # [1, T, 1]
                audio_lengths = torch.tensor([separated_audio_t.shape[0]], dtype=torch.long).cuda()
            else:
                # Use original audio (no AV-TSE)
                audios = sample_features["audios"].cuda()
                audio_lengths = sample_features["audio_lengths"].cuda()
            
            try:
                output = self.model_impl.inference(videos, audios)
            except Exception as e:
                print(f"Error during inference for segment {sample}")
                raise e
            
            segment_output.append(output)

            # GPU Memory Cleanup
            del audios, videos, audio_lengths, video_lengths, sample_features
            if self.use_tse and self.tse_model is not None:
                del mixture_audio, lip_embedding, separated_audio, separated_audio_t
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        return [
            {
                "start_time": seg[0] + offset,
                "end_time": seg[1] + offset,
                "text": output
            } for seg, output in zip(segments, segment_output)
        ]
    
    def mcorec_session_infer(self, session_dir, output_dir):
        """Process a complete MCoReC session"""
        # Load session metadata
        with open(os.path.join(session_dir, "metadata.json"), "r") as f:
            metadata = json.load(f)
            
        # Process speaker clustering
        speaker_segments = {}
        for speaker_name, speaker_data in metadata.items():
            list_tracks_asd = []
            for track in speaker_data['central']['crops']:
                list_tracks_asd.append(os.path.join(session_dir, track['asd']))
            uem_start = speaker_data['central']['uem']['start']
            uem_end = speaker_data['central']['uem']['end']
            speaker_activity_segments = get_speaker_activity_segments(list_tracks_asd, uem_start, uem_end)
            speaker_segments[speaker_name] = speaker_activity_segments
        
        scores = calculate_conversation_scores(speaker_segments)
        clusters = cluster_speakers(scores, list(speaker_segments.keys()))   
        output_clusters_file = os.path.join(output_dir, "speaker_to_cluster.json")
        with open(output_clusters_file, "w") as f:
            json.dump(clusters, f, indent=4)    
        
        # Get central_video.mp4 path
        central_video_path = os.path.join(session_dir, "central_video.mp4")
        
        # Process speaker transcripts
        for speaker_name, speaker_data in tqdm(metadata.items(), desc="Processing speakers", total=len(metadata)):
            print()
            speaker_track_hypotheses = []
            for idx, track in enumerate(speaker_data['central']['crops']):
                video_path = os.path.join(session_dir, track['lip'])
                asd_path = os.path.join(session_dir, track['asd']) if 'asd' in track else None
                
                # Get lip embedding path (.npy file)
                # track['lip'] is like "speakers/spk_0/central_crops/track_00_lip.av.mp4"
                # lip embedding is at "speakers/spk_0/central_crops/track_00_lip.av.npy"
                lip_embedding_path = None
                if self.use_tse:
                    lip_path = track['lip']
                    if lip_path.endswith('.mp4'):
                        lip_embedding_path = lip_path.replace('.mp4', '.npy')
                    else:
                        lip_embedding_path = lip_path + '.npy'
                    lip_embedding_path = os.path.join(session_dir, lip_embedding_path)
                
                with open(os.path.join(session_dir, track['crop_metadata']), "r") as f:
                    crop_metadata = json.load(f)
                track_start_time = crop_metadata['start_time']
                hypotheses = self.infer_video(
                    video_path, 
                    asd_path, 
                    offset=track_start_time, 
                    desc=f"Processing speaker {speaker_name} track {idx+1} of {len(speaker_data['central']['crops'])}",
                    central_video_path=central_video_path if self.use_tse else None,
                    lip_embedding_path=lip_embedding_path if self.use_tse else None
                )
                speaker_track_hypotheses.extend(hypotheses)

                # GPU Memory Cleanup after each track
                torch.cuda.empty_cache()

            output_file = os.path.join(output_dir, f"{speaker_name}.vtt")
            with open(output_file, "w") as f:
                f.write("WEBVTT\n\n")
                for hyp in speaker_track_hypotheses:
                    text = hyp["text"].strip().replace("<unk>", "").strip()
                    start_time = self.format_vtt_timestamp(hyp["start_time"])
                    end_time = self.format_vtt_timestamp(hyp["end_time"])
                    if len(text) == 0:
                        continue
                    f.write(f"{start_time} --> {end_time}\n{text}\n\n")


def main():
    parser = argparse.ArgumentParser(description="Unified inference script for multiple AVSR models")
    
    # Model selection argument
    parser.add_argument(
        '--model_type', 
        type=str, 
        required=True,
        choices=['avsr_cocktail', 'auto_avsr', 'muavic_en'],
        help='Type of model to use for inference'
    )
    
    # Input/output arguments
    parser.add_argument(
        '--session_dir', 
        type=str, 
        required=True, 
        help='Path to folder containing session data (supports glob patterns with *)'
    )
    
    # Model checkpoint arguments
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default=None,
        help='Path to model checkpoint or pretrained model name'
    )
    
    parser.add_argument(
        '--cache_dir',
        type=str,
        default='./model-bin',
        help='Directory to cache downloaded models (default: ./model-bin)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--max_length',
        type=int,
        default=15,
        help='Maximum length of video segments in seconds (default: 15)'
    )
    
    parser.add_argument(
        '--beam_size',
        type=int,
        default=3,
        help='Beam size for beam search decoding (default: 3)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--output_dir_name',
        type=str,
        default='output',
        help='Name of the output directory within each session (default: output)'
    )
    
    # AV-TSE arguments
    parser.add_argument(
        '--tse_checkpoint_path',
        type=str,
        default=None,
        help='Path to AV-TSE model checkpoint (default: uses default SEANet checkpoint)'
    )
    
    parser.add_argument(
        '--use_tse',
        action='store_true',
        default=True,
        help='Use AV-TSE for audio separation before AV-ASR (default: True)'
    )
    
    parser.add_argument(
        '--no_tse',
        dest='use_tse',
        action='store_false',
        help='Disable AV-TSE (use original audio directly)'
    )
    
    args = parser.parse_args()
    
    # Initialize inference engine
    engine = InferenceEngine(
        args.model_type, 
        args.checkpoint_path, 
        args.cache_dir, 
        args.beam_size, 
        args.max_length,
        tse_checkpoint_path=args.tse_checkpoint_path,
        use_tse=args.use_tse
    )
    engine.load_model()
    
    # Process session directories
    if args.session_dir.strip().endswith("*"):
        all_session_dirs = glob.glob(args.session_dir)
    else:
        all_session_dirs = [args.session_dir]
    
    print(f"Inferring {len(all_session_dirs)} sessions using {args.model_type} model")
    
    for session_dir in all_session_dirs:
        output_dir = os.path.join(session_dir, args.output_dir_name)
        os.makedirs(output_dir, exist_ok=True)
        
        session_name = session_dir.split('/')[-1]
        print(f"Processing session {session_name}")
        
        if args.verbose:
            print(f"  Model: {args.model_type}")
            print(f"  Input: {session_dir}")
            print(f"  Output: {output_dir}")
        
        engine.mcorec_session_infer(session_dir, output_dir)
        
        if args.verbose:
            print(f"  Completed session {session_name}")


if __name__ == "__main__":
    main()
