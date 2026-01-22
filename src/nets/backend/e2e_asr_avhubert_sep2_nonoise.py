# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Transformer speech recognition model (pytorch)."""

import logging
import numpy
import torch

from src.nets.backend.ctc import CTC
from src.nets.backend.nets_utils import (
    make_non_pad_mask,
    th_accuracy,
)
from src.nets.backend.transformer.add_sos_eos import add_sos_eos
from src.nets.backend.transformer.decoder import Decoder
# from src.nets.backend.transformer.encoder import Encoder
from src.nets.backend.backbones.avhubert import AVHubertModel
from src.nets.backend.transformer.label_smoothing_loss import LabelSmoothingLoss
from src.nets.backend.transformer.mask import target_mask
from src.nets.backend.nets_utils import MLPHead

from src.separator.seanet import seanet_separator
# from src.separator.loss_seanet import loss_speech


class E2E(torch.nn.Module):
    def __init__(self, args, ignore_id=-1):
        torch.nn.Module.__init__(self)

        # self.encoder = Encoder(
        #     attention_dim=args.adim,
        #     attention_heads=args.aheads,
        #     linear_units=args.eunits,
        #     num_blocks=args.elayers,
        #     input_layer=args.transformer_input_layer,
        #     dropout_rate=args.dropout_rate,
        #     positional_dropout_rate=args.dropout_rate,
        #     attention_dropout_rate=args.transformer_attn_dropout_rate,
        #     encoder_attn_layer_type=args.transformer_encoder_attn_layer_type,
        #     macaron_style=args.macaron_style,
        #     use_cnn_module=args.use_cnn_module,
        #     cnn_module_kernel=args.cnn_module_kernel,
        #     zero_triu=getattr(args, "zero_triu", False),
        #     a_upsample_ratio=args.a_upsample_ratio,
        #     relu_type=getattr(args, "relu_type", "swish"),
        # )

        # self.aux_encoder = Encoder(
        #     attention_dim=args.aux_adim,
        #     attention_heads=args.aux_aheads,
        #     linear_units=args.aux_eunits,
        #     num_blocks=args.aux_elayers,
        #     input_layer=args.aux_transformer_input_layer,
        #     dropout_rate=args.aux_dropout_rate,
        #     positional_dropout_rate=args.aux_dropout_rate,
        #     attention_dropout_rate=args.aux_transformer_attn_dropout_rate,
        #     encoder_attn_layer_type=args.aux_transformer_encoder_attn_layer_type,
        #     macaron_style=args.aux_macaron_style,
        #     use_cnn_module=args.aux_use_cnn_module,
        #     cnn_module_kernel=args.aux_cnn_module_kernel,
        #     zero_triu=getattr(args, "aux_zero_triu", False),
        #     a_upsample_ratio=args.aux_a_upsample_ratio,
        #     relu_type=getattr(args, "aux_relu_type", "swish"),
        # )
        
        self.encoder = AVHubertModel(args)
        # print(self.encoder)
        # exit()

        # self.transformer_input_layer = args.transformer_input_layer
        # self.a_upsample_ratio = args.a_upsample_ratio

        # self.fusion = MLPHead(
        #     idim=args.adim + args.aux_adim,
        #     hdim=args.fusion_hdim,
        #     odim=args.adim,
        #     norm=args.fusion_norm,
        # )

        self.proj_decoder = None
        if args.adim != args.ddim:
            self.proj_decoder = torch.nn.Linear(args.adim, args.ddim)

        if args.mtlalpha < 1:
            self.decoder = Decoder(
                odim=args.odim,
                attention_dim=args.ddim,
                attention_heads=args.dheads,
                linear_units=args.dunits,
                num_blocks=args.dlayers,
                dropout_rate=args.dropout_rate,
                positional_dropout_rate=args.dropout_rate,
                self_attention_dropout_rate=args.transformer_attn_dropout_rate,
                src_attention_dropout_rate=args.transformer_attn_dropout_rate,
            )
        else:
            self.decoder = None
        self.blank = 0
        self.sos = args.odim - 1
        self.eos = args.odim - 1
        self.odim = args.odim
        self.ignore_id = ignore_id

        # self.lsm_weight = a
        self.criterion = LabelSmoothingLoss(
            self.odim,
            self.ignore_id,
            args.lsm_weight,
            args.transformer_length_normalized_loss,
        )

        self.adim = args.adim
        self.mtlalpha = args.mtlalpha
        if args.mtlalpha > 0.0:
            self.ctc = CTC(
                args.odim, args.adim, args.dropout_rate, ctc_type=args.ctc_type, reduce=True
            )
        else:
            self.ctc = None

        # TODO: define separator layers
        #=========================================================
        self.separator = seanet_separator() # Default config N = 1024, L = 40, B = 1024, H = 128, K = 100, R = 6
        self.separator = self.separator.float()
        # self.loss_se = loss_speech().cuda()
        self.mse_loss_feat = torch.nn.MSELoss()
        #=========================================================


    def forward(self, video, audio, video_lengths, audio_lengths, label, label_audios, label_noises, label_audio_lengths, label_noise_lengths):
        video_padding_mask = make_non_pad_mask(video_lengths).to(video.device)
        # attention_mask = 
        # avhubert_features = self.encoder(
        #     input_features = audio, 
        #     video = video,
        #     attention_mask = video_padding_mask
        # )
        avhubert_features = self.encoder(
            input_features = audio, 
            video = video,
            attention_mask = video_padding_mask,
        )
        
        # avhubert_features = self.encoder(
        #     input_features = audio, 
        #     video = video,
        #     attention_mask = None
        # )
            # attention_mask: Optional[torch.Tensor] = None,

        # audio_lengths = torch.div(audio_lengths, 640, rounding_mode="trunc")
        # audio_padding_mask = make_non_pad_mask(audio_lengths).to(video.device).unsqueeze(-2)

        # audio_feat, _ = self.aux_encoder(audio, audio_padding_mask)

        # x = self.fusion(torch.cat((video_feat, audio_feat), dim=-1))
        x = avhubert_features.last_hidden_state
        x_hidden_states = avhubert_features.hidden_states

        # TODO: input x_hidden_states [2 or 3] to separator layers
        #=========================================================
        # print('X_HIDDEN_STATES_LEN', len(x_hidden_states))
        # print('X_HIDDEN_STATES_DIM', x_hidden_states[2].shape)
        # OUTPUTS:
        # X_HIDDEN_STATES_LEN 25
        # X_HIDDEN_STATES_DIM torch.Size([16, 400, 1024]) -> Bs x T x D
        hidden_layer_idx = 1
        B, T, D = x_hidden_states[hidden_layer_idx].shape
        # print('B, T, D', B, T, D)
        # print(x_hidden_states[hidden_layer_idx])

        out_s, out_n = self.separator(x_hidden_states[hidden_layer_idx], M=B)
        # print('out_s', out_s)
        # print('out_s.shape', out_s.shape)
        # print('label_audios.shape', label_audios.repeat(5,1).shape)
        # print('lengths', video_lengths, audio_lengths)
        # TODO: DIFFERENT SHAPE OF SEP OUTPUT AND LABEL AUDIO --> How to solve? 
        # out_s.shape torch.Size([18, 400, 1024])
        # label_audios.shape torch.Size([3, 104, 400])
        # 104 --> configuration_avhubert_avsr.py : audio_feat_dim=104


        loss_s_main = self.mse_loss_feat(out_s[-B:,:,:], label_audios)
        loss_n_main = self.mse_loss_feat(out_n[-B:,:,:], label_noises)	
        loss_n_rest = self.mse_loss_feat(out_n[:-B,:,:], label_noises.repeat(2, 1, 1))
        loss_s_rest = self.mse_loss_feat(out_s[:-B,:,:], label_audios.repeat(2, 1, 1))
        loss_sep = loss_s_main + (loss_n_main + loss_n_rest + loss_s_rest) * 0.1

        # loss_s_main = self.loss_se.forward(out_s[-B:,:], speech)
        # loss_n_main = self.loss_se.forward(out_n[-B:,:], noise)	
        # loss_n_rest = self.loss_se.forward(out_n[:-B,:], noise.repeat(5, 1))
        # loss_s_rest = self.loss_se.forward(out_s[:-B,:], speech.repeat(5, 1))
        # loss = loss_s_main + (loss_n_main + loss_n_rest + loss_s_rest) * 0.1
        # loss_sep = self.neg_sisdr(est_speech, target_speech)

        #=========================================================

        # ctc loss --> label is text
        loss_ctc, ys_hat = self.ctc(x, video_lengths, label)

        if self.proj_decoder:
            x = self.proj_decoder(x)

        # decoder loss
        ys_in_pad, ys_out_pad = add_sos_eos(label, self.sos, self.eos, self.ignore_id)
        ys_mask = target_mask(ys_in_pad, self.ignore_id)
        pred_pad, _ = self.decoder(ys_in_pad, ys_mask, x, video_padding_mask.unsqueeze(-2))
        loss_att = self.criterion(pred_pad, ys_out_pad)

        loss = self.mtlalpha * loss_ctc + (1 - self.mtlalpha) * loss_att + 0.1 * loss_sep
        # loss = loss_sep

        acc = th_accuracy(
            pred_pad.view(-1, self.odim), ys_out_pad, ignore_label=self.ignore_id
        )

        # return loss, loss_ctc, loss_att, acc
        return loss, loss_ctc, loss_att, loss_sep, acc
