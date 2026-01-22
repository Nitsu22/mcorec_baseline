from typing import Optional

import torch

from src.avhubert_avsr.avhubert_avsr_model import AVHubertAVSR
from src.dataset.avhubert_dataset import TorchFBanksAndStack


def _pad_features(samples, pad_val=0.0):
    lengths = [len(s) for s in samples]
    if not lengths:
        raise ValueError("Cannot pad empty sample list")
    max_size = max(lengths)
    if max_size == 0:
        raise ValueError("Cannot pad zero-length features")
    feat_dim = samples[0].shape[1]
    batch = samples[0].new_full((len(samples), max_size, feat_dim), pad_val)
    for i, sample in enumerate(samples):
        if len(sample) == max_size:
            batch[i] = sample
        else:
            pad = sample.new_full((max_size - len(sample), feat_dim), pad_val)
            batch[i] = torch.cat([sample, pad], dim=0)
    return batch, lengths


class AVHubertAVSRSEANet(AVHubertAVSR):
    def __init__(self, config):
        super().__init__(config)
        self.seanet = None
        self.seanet_fbank = None
        self.rate_ratio = 640

    def configure_seanet(
        self,
        seanet_model: torch.nn.Module,
        fbank_extractor: TorchFBanksAndStack,
        rate_ratio: int = 640,
        freeze_seanet: bool = False,
        freeze_avhubert: bool = False,
    ):
        self.seanet = seanet_model
        self.seanet_fbank = fbank_extractor
        self.rate_ratio = rate_ratio

        if freeze_seanet:
            for param in self.seanet.parameters():
                param.requires_grad = False
        if freeze_avhubert:
            for param in self.avsr.parameters():
                param.requires_grad = False

    def forward(
        self,
        videos,
        audios,
        labels,
        video_lengths,
        audio_lengths,
        label_lengths,
        audio_wavs: Optional[torch.Tensor] = None,
        audio_wav_lengths: Optional[torch.Tensor] = None,
        muse_features: Optional[torch.Tensor] = None,
        muse_feature_lengths: Optional[torch.Tensor] = None,
        use_seanet_mask: Optional[torch.Tensor] = None,
    ):
        if self.seanet is None or self.seanet_fbank is None:
            return super().forward(videos, audios, labels, video_lengths, audio_lengths, label_lengths)

        if (
            audio_wavs is None
            or audio_wav_lengths is None
            or muse_features is None
            or muse_feature_lengths is None
            or use_seanet_mask is None
        ):
            return super().forward(videos, audios, labels, video_lengths, audio_lengths, label_lengths)

        if not torch.any(use_seanet_mask):
            return super().forward(videos, audios, labels, video_lengths, audio_lengths, label_lengths)

        audio_feat_list = []
        audio_feat_lengths = []
        batch_size = videos.size(0)

        for i in range(batch_size):
            if bool(use_seanet_mask[i].item()):
                wav_len = int(audio_wav_lengths[i].item())
                muse_len = int(muse_feature_lengths[i].item())
                vid_len = int(video_lengths[i].item())
                if wav_len <= 0 or muse_len <= 0:
                    raise ValueError(f"Invalid MuSE/SEANet input lengths at index {i}: wav={wav_len}, muse={muse_len}")
                if muse_len != vid_len:
                    raise ValueError(f"MuSE/video length mismatch at index {i}: muse={muse_len}, video={vid_len}")
                expected_wav_len = muse_len * self.rate_ratio
                if wav_len != expected_wav_len:
                    raise ValueError(
                        f"Audio/MuSE length mismatch at index {i}: wav={wav_len}, expected={expected_wav_len}"
                    )

                wav = audio_wavs[i, 0, :wav_len]
                muse = muse_features[i, :muse_len, :]

                out_speech, _ = self.seanet(wav.unsqueeze(0), muse.unsqueeze(0), M=1)
                out = out_speech[-1].unsqueeze(-1)
                feat = self.seanet_fbank(out)
                if feat.shape[0] != muse_len:
                    raise ValueError(
                        f"SEANet fbank length mismatch at index {i}: feat={feat.shape[0]}, expected={muse_len}"
                    )
                audio_feat_list.append(feat)
                audio_feat_lengths.append(feat.shape[0])
            else:
                base_len = int(audio_lengths[i].item())
                if base_len <= 0:
                    raise ValueError(f"Invalid audio feature length at index {i}: {base_len}")
                feat = audios[i, :, :base_len].transpose(0, 1)
                audio_feat_list.append(feat)
                audio_feat_lengths.append(base_len)

        audio_feat_batch, _ = _pad_features(audio_feat_list)
        audio_feat_batch = audio_feat_batch.permute(0, 2, 1)
        audio_lengths = torch.tensor(audio_feat_lengths, device=audios.device)

        return super().forward(videos, audio_feat_batch, labels, video_lengths, audio_lengths, label_lengths)
