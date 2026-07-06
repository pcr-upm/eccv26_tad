import math
from collections import OrderedDict
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmengine.model import BaseModule, ModuleList
from mmengine.registry import MODELS


class LayerNorm(nn.LayerNorm):
    """LayerNorm that internally casts to float32 (fp16/bf16 autocast safe)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        return super().forward(x.float()).to(orig_dtype)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class TemporalFreqAdapter(BaseModule):
    """Frame2Freq's adapter: a temporal Conv3D branch + a multi-resolution FFT branch,
    fused by concatenation, applied to the patch tokens of a per-frame ViT block.

    https://github.com/th-nesh/Frame2Freq/blob/main/models_adapter.py
    """

    def __init__(
        self,
        in_channels: int,
        adapter_channels: int,
        freq_windows: Optional[List[int]] = None,
        kernel_size: Tuple[int, int, int] = (3, 1, 1),
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_channels, adapter_channels)
        self.fc2 = nn.Linear(adapter_channels, in_channels)
        self.freq_windows = freq_windows
        self.Ca = adapter_channels

        Ct = adapter_channels // 2
        Cf = adapter_channels - Ct
        self.Ct, self.Cf = Ct, Cf

        self.temporal_conv = nn.Conv3d(
            Ct, Ct, kernel_size=kernel_size, padding=tuple(k // 2 for k in kernel_size), groups=Ct
        )
        self.freq_conv = nn.Conv3d(
            2 * Cf, 2 * Cf, kernel_size=kernel_size, padding=tuple(k // 2 for k in kernel_size), groups=2 * Cf
        )

        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.constant_(self.fc2.bias, 0.0)
        nn.init.constant_(self.temporal_conv.bias, 0.0)
        nn.init.constant_(self.freq_conv.bias, 0.0)

    def _windows_for(self, T: int) -> List[int]:
        if self.freq_windows is not None:
            windows = self.freq_windows
        else:
            windows = [w for w in (T, T // 2, T // 4) if w >= 1]
        for w in windows:
            assert T % w == 0, f"T={T} not divisible by freq window={w}"
        return windows

    def forward(self, x: torch.Tensor, T: int) -> torch.Tensor:
        """x: (B*T, L, C) with a leading CLS token, T: number of frames."""
        BT, L, C = x.shape
        B = BT // T
        Ct, Cf, Ca = self.Ct, self.Cf, self.Ca

        H = W = round(math.sqrt(L - 1))
        assert L - 1 == H * W

        tokens = self.fc1(x[:, 1:, :])  # (BT, H*W, Ca)
        tokens = tokens.view(B, T, H, W, Ca)
        xt, xf = tokens[..., :Ct], tokens[..., Ct:]
        orig_dtype = tokens.dtype

        # temporal branch: depthwise Conv3D over (T, H, W).
        # conv_depthwise3d has no fp16/bf16 CUDA kernel, so run it in fp32 with
        # autocast disabled (autocast would otherwise re-cast the fp32 input
        # back to the autocast dtype before calling into the conv kernel).
        xt = xt.permute(0, 4, 1, 2, 3).float()  # (B, Ct, T, H, W)
        with torch.autocast(device_type=xt.device.type, enabled=False):
            xt = self.temporal_conv(xt)
        xt = xt.permute(0, 2, 3, 4, 1).to(orig_dtype)  # (B, T, H, W, Ct)

        # frequency branch: multi-resolution FFT along T, depthwise Conv3D, inverse FFT
        windows = self._windows_for(T)
        out_f = torch.zeros_like(xf, dtype=torch.float32)
        for w in windows:
            chunks = T // w
            xf_chunk = xf.view(B, chunks, w, H, W, Cf).float()

            X = torch.fft.rfft(xf_chunk, dim=2)
            X = torch.view_as_real(X).contiguous()
            B_, chunks_, Freq, H_, W_, d, _ = X.shape
            X = X.view(B_ * chunks_, 2 * d, Freq, H_, W_)

            with torch.autocast(device_type=X.device.type, enabled=False):
                Y = self.freq_conv(X)
            Y = Y.view(B, chunks, Freq, H, W, d, 2)
            Y = torch.complex(Y[..., 0], Y[..., 1])

            y = torch.fft.irfft(Y, n=w, dim=2)  # (B, chunks, w, H, W, Cf)
            out_f += y.contiguous().view(B, T, H, W, Cf)

        out_f = (out_f / len(windows)).to(orig_dtype)

        out = torch.cat([xt, out_f], dim=-1)  # (B, T, H, W, Ca)
        out = out.reshape(BT, H * W, Ca)
        out = self.fc2(out)

        # pad the CLS token position with zeros and add the residual
        out = F.pad(out, (0, 0, 1, 0))
        return x + out


class ResidualAttentionBlock(BaseModule):
    """CLIP transformer block with per-frame self-attention, optionally wrapped with
    TemporalFreqAdapter before the attention and/or the MLP sub-layers."""

    def __init__(
        self,
        d_model: int,
        n_head: int,
        adapter_width: int,
        adapter_kernel_size: Tuple[int, int, int],
        freq_windows: Optional[List[int]],
        adapter_pre_attn: bool,
        adapter_pre_mlp: bool,
    ) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = LayerNorm(d_model)

        adapter_kwargs = dict(
            in_channels=d_model,
            adapter_channels=adapter_width,
            freq_windows=freq_windows,
            kernel_size=adapter_kernel_size,
        )
        self.adapter_pre_attn = TemporalFreqAdapter(**adapter_kwargs) if adapter_pre_attn else None
        self.adapter_pre_mlp = TemporalFreqAdapter(**adapter_kwargs) if adapter_pre_mlp else None

    def attention(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        H = self.attn.num_heads
        head_dim = C // H

        qkv = F.linear(x, weight=self.attn.in_proj_weight, bias=self.attn.in_proj_bias)
        qkv = qkv.view(B, L, 3, H, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, L, C)
        return self.attn.out_proj(out)

    def forward(self, x: torch.Tensor, num_frames: int) -> torch.Tensor:
        if self.adapter_pre_attn is not None:
            x = self.adapter_pre_attn(x, num_frames)
        x = x + self.attention(self.ln_1(x))
        if self.adapter_pre_mlp is not None:
            x = self.adapter_pre_mlp(x, num_frames)
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(BaseModule):
    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        adapter_width: int,
        adapter_layers: int,
        adapter_kernel_size: Tuple[int, int, int],
        freq_windows: Optional[List[int]],
        adapter_pre_attn: bool,
        adapter_pre_mlp: bool,
        with_cp: bool = False,
    ) -> None:
        super().__init__()
        self.with_cp = with_cp
        self.resblocks = ModuleList(
            [
                ResidualAttentionBlock(
                    d_model=width,
                    n_head=heads,
                    adapter_width=adapter_width,
                    adapter_kernel_size=adapter_kernel_size,
                    freq_windows=freq_windows,
                    adapter_pre_attn=adapter_pre_attn and i >= layers - adapter_layers,
                    adapter_pre_mlp=adapter_pre_mlp and i >= layers - adapter_layers,
                )
                for i in range(layers)
            ]
        )

    def forward(self, x: torch.Tensor, num_frames: int) -> torch.Tensor:
        for block in self.resblocks:
            if self.with_cp and x.requires_grad:
                x = cp.checkpoint(block, x, num_frames, use_reentrant=False)
            else:
                x = block(x, num_frames)
        return x


@MODELS.register_module()
class VisionTransformerCLIPFreqAdapter(BaseModule):
    """CLIP ViT backbone (2D patch embed + per-frame attention) with Frame2Freq-style
    temporal/frequency adapters for end-to-end temporal action detection.

    https://github.com/th-nesh/Frame2Freq/blob/main/models_adapter.py

    The pretrained CLIP visual encoder weights are frozen (kept at lr=0 via the
    optimizer config); only the ``TemporalFreqAdapter`` modules are trained.

    Input:  (B, 3, T, H, W)
    Output: (B, embed_dims, T + 1), where index 0 is the CLS-token feature
            (averaged over T frames) and indices 1..T are the per-frame
            patch-token features (averaged over the spatial grid).
    """

    def __init__(
        self,
        input_resolution: int = 224,
        patch_size: int = 16,
        embed_dims: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        num_frames: int = 16,
        adapter_width: int = 384,
        adapter_layers: Optional[int] = None,
        adapter_kernel_size: Tuple[int, int, int] = (3, 1, 1),
        freq_windows: Optional[List[int]] = None,
        adapter_pre_attn: bool = False,
        adapter_pre_mlp: bool = True,
        with_cp: bool = False,
        init_cfg=None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        self.input_resolution = input_resolution
        self.patch_size = patch_size
        self.embed_dims = embed_dims
        self.num_frames = num_frames
        self.grid_size = input_resolution // patch_size

        self.conv1 = nn.Conv2d(3, embed_dims, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = embed_dims**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(embed_dims))
        self.positional_embedding = nn.Parameter(scale * torch.randn(self.grid_size**2 + 1, embed_dims))
        self.ln_pre = LayerNorm(embed_dims)

        self.transformer = Transformer(
            width=embed_dims,
            layers=depth,
            heads=num_heads,
            adapter_width=adapter_width,
            adapter_layers=adapter_layers if adapter_layers is not None else depth,
            adapter_kernel_size=adapter_kernel_size,
            freq_windows=freq_windows,
            adapter_pre_attn=adapter_pre_attn,
            adapter_pre_mlp=adapter_pre_mlp,
            with_cp=with_cp,
        )

        self.ln_post = LayerNorm(embed_dims)

        num_vit_param = sum(p.numel() for n, p in self.named_parameters() if "adapter" not in n)
        num_adapter_param = sum(p.numel() for n, p in self.named_parameters() if "adapter" in n)
        print(
            "CLIP ViT's params: {}, Adapter's params: {}, ratio: {:.1f}%".format(
                num_vit_param, num_adapter_param, num_adapter_param / num_vit_param * 100
            )
        )

    def _interpolate_pos_embed(self, gh: int, gw: int) -> torch.Tensor:
        if gh * gw + 1 == self.positional_embedding.shape[0]:
            return self.positional_embedding

        cls_pos = self.positional_embedding[:1]
        patch_pos = self.positional_embedding[1:]
        patch_pos = patch_pos.reshape(1, self.grid_size, self.grid_size, -1).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(patch_pos.float(), size=(gh, gw), mode="bicubic", align_corners=False)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(gh * gw, -1).to(self.positional_embedding.dtype)
        return torch.cat([cls_pos, patch_pos], dim=0)

    def forward(self, x: torch.Tensor):
        B, _, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, 3, H, W)

        x = self.conv1(x)  # (B*T, D, gh, gw)
        gh, gw = x.shape[-2], x.shape[-1]
        x = x.flatten(2).transpose(1, 2)  # (B*T, gh*gw, D)

        cls_tokens = self.class_embedding.to(x.dtype).view(1, 1, -1).expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B*T, gh*gw+1, D)

        pos_embed = self._interpolate_pos_embed(gh, gw)
        x = x + pos_embed.to(x.dtype)
        x = self.ln_pre(x)

        x = self.transformer(x, T)
        x = self.ln_post(x)

        x = x.view(B, T, gh * gw + 1, self.embed_dims)
        global_feat = x[:, :, 0, :].mean(dim=1)  # (B, D), CLS token averaged over T
        patch_feat = x[:, :, 1:, :].mean(dim=2)  # (B, T, D), per-frame patch-token mean

        out = torch.cat([global_feat.unsqueeze(1), patch_feat], dim=1)  # (B, T+1, D)
        out = out.permute(0, 2, 1).contiguous()  # (B, D, T+1)
        return out, None
