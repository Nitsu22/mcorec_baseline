import math
from typing import Optional, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F


def _check_finite(name, tensor):
    finite = torch.isfinite(tensor)
    if not finite.all():
        nonfinite = (~finite).sum().item()
        total = tensor.numel()
        raise RuntimeError(
            f"{name} has non-finite values: {nonfinite}/{total} "
            f"(dtype={tensor.dtype}, shape={tuple(tensor.shape)})"
        )


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: int = 10000):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RotaryEmbedding dim must be even, got {dim}")
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def _build_cache(self, seq_len: int, device, dtype):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]
        self._cos_cached = cos.to(dtype=dtype)
        self._sin_cached = sin.to(dtype=dtype)
        self._seq_len_cached = seq_len

    def rotate_queries_or_keys(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[-2]
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached < seq_len
            or self._cos_cached.device != x.device
            or self._cos_cached.dtype != x.dtype
        ):
            self._build_cache(seq_len, x.device, x.dtype)
        cos = self._cos_cached[..., :seq_len, :]
        sin = self._sin_cached[..., :seq_len, :]
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        x_rot = torch.stack((-x2, x1), dim=-1).reshape_as(x)
        return x * cos + x_rot * sin


class tf_locoformer_separator(nn.Module):
    """TF-Locoformer separator (Small Config) with Direct Mapping.
    
    Based on TF-LOCOFORMER (S) configuration:
    - D (emb_dim) = 96
    - B (n_layers) = 4
    - C (ffn_hidden_dim) = 256
    - K (conv1d_kernel) = 4
    - H (n_heads) = 4
    - G (num_groups) = 4
    """

    def __init__(
        self,
        input_dim: int = 1024,
        output_dim: int = 104,
        num_spk: int = 1,
        compress_dim: int = 256,
        # --- Small Model Config ---
        n_layers: int = 1,          # B=4
        emb_dim: int = 96,          # D=96
        ffn_hidden_dim: Union[int, List[int]] = (256, 256), # C=256
        conv1d_kernel: int = 4,     # K=4
        n_heads: int = 4,           # H=4
        num_groups: int = 4,        # G=4
        # --------------------------
        norm_type: str = "rmsgroupnorm",
        tf_order: str = "ft",
        flash_attention: bool = False,
        attention_dim: int = 96, # Typically matches emb_dim
        pos_enc: str = "rope",
        ffn_type: Union[str, List[str]] = ("swiglu_conv1d", "swiglu_conv1d"),
        conv1d_shift: int = 1,
        dropout: float = 0.0,
        eps: float = 1.0e-5,
        R: int = 3,
    ):
        super().__init__()
        if num_spk != 1:
            raise ValueError("This separator supports num_spk=1 only.")
        if compress_dim <= 0 or compress_dim > input_dim:
            raise ValueError(f"compress_dim must be in [1, {input_dim}], got {compress_dim}")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.R = R
        self.compress_dim = compress_dim

        self.in_proj = nn.Linear(input_dim, compress_dim, bias=False)
        self.out_proj = nn.Linear(compress_dim, input_dim, bias=False)

        # Input Convolution (Encoder equivalent)
        t_ksize = 3
        ks, padding = (t_ksize, 3), (t_ksize // 2, 1)
        self.conv = nn.Sequential(
            nn.Conv2d(1, emb_dim, ks, padding=padding),
            nn.GroupNorm(1, emb_dim, eps=eps), # Acts as LayerNorm (gLN)
        )

        if attention_dim % n_heads != 0:
            raise ValueError(f"attention_dim must be divisible by n_heads")
        
        if pos_enc == "nope":
            rope_freq = rope_time = None
        elif pos_enc == "rope":
            rope_dim = attention_dim // n_heads
            rope_freq = RotaryEmbedding(rope_dim)
            rope_time = RotaryEmbedding(rope_dim)
        else:
            raise ValueError(f"Unsupported positional encoding: {pos_enc}")

        self.blocks = nn.ModuleList([])
        for _ in range(n_layers):
            self.blocks.append(
                TFLocoformerBlock(
                    rope_freq,
                    rope_time,
                    emb_dim=emb_dim,
                    norm_type=norm_type,
                    num_groups=num_groups,
                    tf_order=tf_order,
                    n_heads=n_heads,
                    flash_attention=flash_attention,
                    attention_dim=attention_dim,
                    ffn_type=ffn_type,
                    ffn_hidden_dim=ffn_hidden_dim,
                    conv1d_kernel=conv1d_kernel,
                    conv1d_shift=conv1d_shift,
                    dropout=dropout,
                    eps=eps,
                )
            )

        # Output Projection for Mapping (Deconv2D equivalent)
        # Replacing mask_head (Sigmoid) with direct regression
        # Using ConvTranspose2d to match the inverse of the input Conv2d logic roughly, 
        # or simply Conv2d(1x1) to project back to 1 channel.
        # Since we want to map back to the input feature space:
        self.output_head = nn.Conv2d(emb_dim, 1, kernel_size=1)
        
        self.basis_linear = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, mixture_feat: torch.Tensor, M: int = 1):
        # mixture_feat: [B, T, F]
        if mixture_feat.ndim == 3:
            mix = mixture_feat
            batch = mix.unsqueeze(1)  # [B, 1, T, F]
        elif mixture_feat.ndim == 4:
            if mixture_feat.shape[1] != 1:
                raise ValueError("Expected channel dimension=1 for 4D input.")
            batch = mixture_feat
            mix = mixture_feat.squeeze(1)
        else:
            raise ValueError("mixture_feat must be 3D [B, T, F] or 4D [B, 1, T, F].")

        if mix.shape[-1] != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {mix.shape[-1]}")

        _check_finite("mix", mix)

        with torch.cuda.amp.autocast(enabled=True):
            mix_comp = self.in_proj(mix)  # [B, T, compress_dim]
            _check_finite("mix_comp", mix_comp)
            batch = mix_comp.unsqueeze(1)  # [B, 1, T, compress_dim]
            batch = batch.to(torch.float32)
            z = self.conv(batch)  # [B, emb_dim, T, F]
            _check_finite("conv_out", z)

        for idx, block in enumerate(self.blocks):
            z = block(z)
            _check_finite(f"block_{idx}_out", z)

        # --- Mapping Output ---
        # Direct estimation instead of masking
        est_feat_map = self.output_head(z)  # [B, 1, T, F]
        _check_finite("output_head_out", est_feat_map)
        est_feat_comp = est_feat_map.squeeze(1)  # [B, T, compress_dim]
        _check_finite("est_feat_comp", est_feat_comp)
        est_feat_1024 = self.out_proj(est_feat_comp)  # [B, T, input_dim]
        _check_finite("out_proj_out", est_feat_1024)
        
        # Note: If your target features are strictly positive, you might want F.relu() here.
        # For general mapping (e.g., Log-Mel or complex real/imag parts), linear output is correct.

        est_speech = self.basis_linear(est_feat_1024)  # [B, T, out_dim]
        _check_finite("basis_linear_out", est_speech)

        if self.R > 1:
            est_speech = est_speech.repeat(self.R, 1, 1)

        est_speech = est_speech.transpose(1, 2)  # [R*B, out_dim, T]
        est_noise = est_speech.new_zeros(est_speech.shape)
        
        return est_speech, est_noise


class TFLocoformerBlock(nn.Module):
    def __init__(
        self,
        rope_freq,
        rope_time,
        emb_dim=96,
        norm_type="rmsgroupnorm",
        num_groups=4, # G=4
        tf_order="ft",
        n_heads=4, # H=4
        flash_attention=False,
        attention_dim=96,
        ffn_type=("swiglu_conv1d", "swiglu_conv1d"),
        ffn_hidden_dim=(256, 256),
        conv1d_kernel=4,
        conv1d_shift=1,
        dropout=0.0,
        eps=1.0e-5,
    ):
        super().__init__()

        if tf_order not in ["tf", "ft"]:
            raise ValueError(tf_order)
        self.tf_order = tf_order

        self.freq_path = LocoformerBlock(
            rope_freq,
            emb_dim=emb_dim,
            norm_type=norm_type,
            num_groups=num_groups,
            n_heads=n_heads,
            flash_attention=flash_attention,
            attention_dim=attention_dim,
            ffn_type=ffn_type,
            ffn_hidden_dim=ffn_hidden_dim,
            conv1d_kernel=conv1d_kernel,
            conv1d_shift=conv1d_shift,
            dropout=dropout,
            eps=eps,
        )
        self.frame_path = LocoformerBlock(
            rope_time,
            emb_dim=emb_dim,
            norm_type=norm_type,
            num_groups=num_groups,
            n_heads=n_heads,
            flash_attention=flash_attention,
            attention_dim=attention_dim,
            ffn_type=ffn_type,
            ffn_hidden_dim=ffn_hidden_dim,
            conv1d_kernel=conv1d_kernel,
            conv1d_shift=conv1d_shift,
            dropout=dropout,
            eps=eps,
        )

    def forward(self, input):
        _check_finite("TFLocoformerBlock.input", input)
        if self.tf_order == "ft":
            return self.freq_frame_process(input)
        return self.frame_freq_process(input)

    def freq_frame_process(self, input):
        output = input.movedim(1, -1)
        _check_finite("TFLocoformerBlock.freq_frame.input_moved", output)
        output = self.freq_path(output)
        _check_finite("TFLocoformerBlock.freq_frame.after_freq_path", output)

        output = output.transpose(1, 2)
        _check_finite("TFLocoformerBlock.freq_frame.after_transpose", output)
        output = self.frame_path(output)
        _check_finite("TFLocoformerBlock.freq_frame.after_frame_path", output)
        return output.transpose(-1, 1)

    def frame_freq_process(self, input):
        output = input.transpose(1, -1)
        _check_finite("TFLocoformerBlock.frame_freq.input_transpose", output)
        output = self.frame_path(output)
        _check_finite("TFLocoformerBlock.frame_freq.after_frame_path", output)

        output = output.transpose(1, 2)
        _check_finite("TFLocoformerBlock.frame_freq.after_transpose", output)
        output = self.freq_path(output)
        _check_finite("TFLocoformerBlock.frame_freq.after_freq_path", output)
        return output.movedim(-1, 1)


class LocoformerBlock(nn.Module):
    def __init__(
        self,
        rope,
        emb_dim=96,
        norm_type="rmsgroupnorm",
        num_groups=4,
        n_heads=4,
        flash_attention=False,
        attention_dim=96,
        ffn_type=("swiglu_conv1d", "swiglu_conv1d"),
        ffn_hidden_dim=(256, 256),
        conv1d_kernel=4,
        conv1d_shift=1,
        dropout=0.0,
        eps=1.0e-5,
    ):
        super().__init__()

        FFN = {
            "conv1d": ConvDeconv1d,
            "swiglu_conv1d": SwiGLUConvDeconv1d,
        }
        Norm = {
            "layernorm": nn.LayerNorm,
            "rmsgroupnorm": RMSGroupNorm,
        }
        
        ffn_type_list = ffn_type if isinstance(ffn_type, (list, tuple)) else [ffn_type]
        ffn_hidden_list = ffn_hidden_dim if isinstance(ffn_hidden_dim, (list, tuple)) else [ffn_hidden_dim]

        self.macaron_style = len(ffn_type_list) == 2
        
        self.ffn_norm = nn.ModuleList([])
        self.ffn = nn.ModuleList([])

        for f_type, f_dim in zip(ffn_type_list, ffn_hidden_list):
             if f_type not in FFN:
                raise ValueError(f_type)
             if norm_type == "rmsgroupnorm":
                 self.ffn_norm.append(Norm[norm_type](num_groups, emb_dim, eps=eps))
             else:
                 self.ffn_norm.append(Norm[norm_type](emb_dim, eps=eps))
             self.ffn.append(
                 FFN[f_type](
                     emb_dim,
                     f_dim,
                     conv1d_kernel,
                     conv1d_shift,
                     dropout=dropout,
                 )
             )

        if norm_type == "rmsgroupnorm":
            self.attn_norm = Norm[norm_type](num_groups, emb_dim, eps=eps)
        else:
            self.attn_norm = Norm[norm_type](emb_dim, eps=eps)
        self.attn = MultiHeadSelfAttention(
            emb_dim,
            attention_dim=attention_dim,
            n_heads=n_heads,
            rope=rope,
            dropout=dropout,
            flash_attention=flash_attention,
        )

    def forward(self, x):
        _check_finite("LocoformerBlock.input", x)
        B, T, Freq, C = x.shape
        # Scale residual connections by 0.5 if macaron style (as per paper Eq. 2 & 4)
        scale = 0.5 if self.macaron_style else 1.0

        if self.macaron_style:
            input_ = x
            output = self.ffn_norm[0](x)
            _check_finite("LocoformerBlock.ffn0_norm_out", output)
            output = self.ffn[0](output)
            _check_finite("LocoformerBlock.ffn0_out", output)
            output = output * scale + input_ 
            _check_finite("LocoformerBlock.ffn0_res_out", output)
        else:
            output = x

        input_ = output
        output = self.attn_norm(output)
        _check_finite("LocoformerBlock.attn_norm_out", output)
        output = output.view([B * T, Freq, C])
        _check_finite("LocoformerBlock.attn_in", output)
        output = self.attn(output)
        _check_finite("LocoformerBlock.attn_out", output)
        output = output.view([B, T, Freq, C]) + input_
        _check_finite("LocoformerBlock.attn_res_out", output)

        if self.macaron_style:
            input_ = output
            output = self.ffn_norm[1](output)
            _check_finite("LocoformerBlock.ffn1_norm_out", output)
            output = self.ffn[1](output)
            _check_finite("LocoformerBlock.ffn1_out", output)
            output = output * scale + input_ 
            _check_finite("LocoformerBlock.ffn1_res_out", output)
        elif len(self.ffn) == 1:
            input_ = output
            output = self.ffn_norm[0](output)
            _check_finite("LocoformerBlock.ffn_norm_out", output)
            output = self.ffn[0](output)
            _check_finite("LocoformerBlock.ffn_out", output)
            output = output + input_
            _check_finite("LocoformerBlock.ffn_res_out", output)

        return output

class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        emb_dim,
        attention_dim,
        n_heads=8,
        dropout=0.0,
        rope=None,
        flash_attention=False,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.dropout = dropout
        self.rope = rope
        self.qkv = nn.Linear(emb_dim, attention_dim * 3, bias=False)
        self.aggregate_heads = nn.Sequential(nn.Linear(attention_dim, emb_dim, bias=False), nn.Dropout(dropout))

        if flash_attention:
            self.flash_attention_config = dict(enable_flash=True, enable_math=False, enable_mem_efficient=False)
        else:
            self.flash_attention_config = dict(enable_flash=False, enable_math=True, enable_mem_efficient=True)

    def forward(self, input):
        query, key, value = self.get_qkv(input)
        if self.rope is not None:
            query, key = self.apply_rope(query, key)

        if hasattr(F, "scaled_dot_product_attention"):
            if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "sdp_kernel"):
                with torch.backends.cuda.sdp_kernel(**self.flash_attention_config):
                    output = F.scaled_dot_product_attention(
                        query=query,
                        key=key,
                        value=value,
                        attn_mask=None,
                        dropout_p=self.dropout if self.training else 0.0,
                    )
            else:
                output = F.scaled_dot_product_attention(
                    query=query,
                    key=key,
                    value=value,
                    attn_mask=None,
                    dropout_p=self.dropout if self.training else 0.0,
                )
        else:
            output = self._manual_attention(query, key, value)

        output = output.transpose(1, 2)
        output = output.reshape(output.shape[:2] + (-1,))
        return self.aggregate_heads(output)

    def get_qkv(self, input):
        n_batch, seq_len = input.shape[:2]
        _check_finite("MultiHeadSelfAttention.input", input)
        x = self.qkv(input).reshape(n_batch, seq_len, 3, self.n_heads, -1)
        _check_finite("MultiHeadSelfAttention.qkv_out", x)
        x = x.movedim(-2, 1)
        query, key, value = x[..., 0, :], x[..., 1, :], x[..., 2, :]
        _check_finite("MultiHeadSelfAttention.query", query)
        _check_finite("MultiHeadSelfAttention.key", key)
        _check_finite("MultiHeadSelfAttention.value", value)
        return query, key, value

    @torch.cuda.amp.autocast(enabled=True)
    def apply_rope(self, query, key):
        query = self.rope.rotate_queries_or_keys(query)
        key = self.rope.rotate_queries_or_keys(key)
        return query, key

    def _manual_attention(self, query, key, value):
        dk = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(dk)
        attn = torch.softmax(scores, dim=-1)
        if self.training and self.dropout > 0:
            attn = F.dropout(attn, p=self.dropout)
        return torch.matmul(attn, value)


class ConvDeconv1d(nn.Module):
    def __init__(self, dim, dim_inner, conv1d_kernel, conv1d_shift, dropout=0.0, **kwargs):
        super().__init__()
        self.diff_ks = conv1d_kernel - conv1d_shift
        self.net = nn.Sequential(
            nn.Conv1d(dim, dim_inner, conv1d_kernel, stride=conv1d_shift),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.ConvTranspose1d(dim_inner, dim, conv1d_kernel, stride=conv1d_shift),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        b, s1, s2, h = x.shape
        x = x.view(b * s1, s2, h)
        x = x.transpose(-1, -2)
        x = self.net(x).transpose(-1, -2)
        x = x[..., self.diff_ks // 2 : self.diff_ks // 2 + s2, :]
        return x.view(b, s1, s2, h)


class SwiGLUConvDeconv1d(nn.Module):
    def __init__(self, dim, dim_inner, conv1d_kernel, conv1d_shift, dropout=0.0, **kwargs):
        super().__init__()
        self.conv1d = nn.Conv1d(dim, dim_inner * 2, conv1d_kernel, stride=conv1d_shift)
        self.swish = nn.SiLU()
        self.deconv1d = nn.ConvTranspose1d(dim_inner, dim, conv1d_kernel, stride=conv1d_shift)
        self.dropout = nn.Dropout(dropout)
        self.dim_inner = dim_inner
        self.diff_ks = conv1d_kernel - conv1d_shift
        self.conv1d_kernel = conv1d_kernel
        self.conv1d_shift = conv1d_shift

    def forward(self, x):
        b, s1, s2, h = x.shape
        x = x.contiguous().view(b * s1, s2, h)
        x = x.transpose(-1, -2)

        seq_len = (
            math.ceil((s2 + 2 * self.diff_ks - self.conv1d_kernel) / self.conv1d_shift) * self.conv1d_shift
            + self.conv1d_kernel
        )
        x = F.pad(x, (self.diff_ks, seq_len - s2 - self.diff_ks))

        x = self.conv1d(x)
        _check_finite("SwiGLUConvDeconv1d.conv1d_out", x)
        gate = self.swish(x[..., self.dim_inner :, :])
        _check_finite("SwiGLUConvDeconv1d.gate", gate)
        x = x[..., : self.dim_inner, :] * gate
        _check_finite("SwiGLUConvDeconv1d.gated", x)
        x = self.dropout(x)
        x = self.deconv1d(x).transpose(-1, -2)
        _check_finite("SwiGLUConvDeconv1d.deconv1d_out", x)

        x = x[..., self.diff_ks : self.diff_ks + s2, :]
        x = self.dropout(x)
        _check_finite("SwiGLUConvDeconv1d.output", x)
        return x.view(b, s1, s2, h)


class RMSGroupNorm(nn.Module):
    def __init__(self, num_groups, dim, eps=1e-8, bias=False):
        super().__init__()
        if dim % num_groups != 0:
            raise ValueError(dim, num_groups)
        self.num_groups = num_groups
        self.dim_per_group = dim // self.num_groups

        self.gamma = nn.Parameter(torch.Tensor(dim).to(torch.float32))
        nn.init.ones_(self.gamma)

        self.bias = bias
        if self.bias:
            self.beta = nn.Parameter(torch.Tensor(dim).to(torch.float32))
            nn.init.zeros_(self.beta)
        self.eps = eps
        self.num_groups = num_groups

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, input):
        _check_finite("RMSGroupNorm.input", input)
        others = input.shape[:-1]
        input = input.view(others + (self.num_groups, self.dim_per_group))
        _check_finite("RMSGroupNorm.input_reshaped", input)
        norm_ = input.norm(2, dim=-1, keepdim=True)
        _check_finite("RMSGroupNorm.norm", norm_)
        rms = norm_ * self.dim_per_group ** (-1.0 / 2)
        _check_finite("RMSGroupNorm.rms", rms)
        output = input / (rms + self.eps)
        _check_finite("RMSGroupNorm.output_normed", output)
        output = output.view(others + (-1,))
        _check_finite("RMSGroupNorm.output_reshaped", output)
        _check_finite("RMSGroupNorm.gamma", self.gamma)
        output = output * self.gamma
        _check_finite("RMSGroupNorm.output_scaled", output)
        if self.bias:
            output = output + self.beta
            _check_finite("RMSGroupNorm.output_biased", output)
        return output

# (RotaryEmbedding, MultiHeadSelfAttention, ConvDeconv1d, SwiGLUConvDeconv1d, RMSGroupNorm are unchanged)
TFLocoformerSeparator = tf_locoformer_separator
