import math
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks import DropPath
from mmcv.cnn.bricks.transformer import FFN, PatchEmbed
from torch import Tensor
from mmengine.model import BaseModule, ModuleList
from mmengine.model.weight_init import constant_init, kaiming_init, trunc_normal_init
from mmengine.registry import MODELS
from torchvision.ops import DeformConv2d

from mmaction.models.backbones.vit_mae import get_sinusoid_encoding
from mmaction.utils import ConfigType, OptConfigType


class HeatmapHead(BaseModule):
    """
    A simple heatmap head consisting of deconvolutional layers for upsampling.
    """

    def __init__(
        self,
        in_channels=768,
        in_size=14,
        out_channels=32,  # e.g., number of landmarks
        deconv_out_channels=(256, 256),
        deconv_kernel_sizes=(4, 4),
        init_cfg: OptConfigType = None,
    ):
        super(HeatmapHead, self).__init__(init_cfg=init_cfg)

        deconv_layers = []
        # Optional upsampling if the input size is not the expected one
        if in_size != 14:
            deconv_layers.append(nn.AdaptiveAvgPool2d((14, 14)))

        current_channels = in_channels
        for i in range(len(deconv_out_channels)):
            deconv_layers.extend(
                [
                    nn.ConvTranspose2d(
                        current_channels,
                        deconv_out_channels[i],
                        kernel_size=deconv_kernel_sizes[i],
                        stride=2,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(deconv_out_channels[i]),
                    nn.ReLU(inplace=True),
                ]
            )
            current_channels = deconv_out_channels[i]

        self.deconv_layers = nn.Sequential(*deconv_layers)
        self.final_layer = nn.Conv2d(
            current_channels, out_channels, kernel_size=1, stride=1
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.deconv_layers(x)
        x = self.final_layer(x)
        return x


class EfficientAdapter(BaseModule):
    """
    An efficient Adapter module.
    Supports 'sparse_conv', '2d_conv', 'temporal_dwconv', 'deformable_conv_t', and 'freq_conv'.
    """

    def __init__(
        self,
        embed_dims: int,
        mlp_ratio: float = 0.25,
        conv_type: str = "sparse_conv",
        kernel_size: int = 3,
        temporal_kernel_size: int = 3,
        dilation: int = 1,
        deformable_groups: int = 1,
        use_attn: Union[bool, int] = False,
        n_key_tokens: int = 1,
        heatmap_grid_size: Optional[tuple] = None,
        freq_windows: Optional[List[int]] = None,
        freq_kernel_size: tuple = (3, 3, 3),
        **kwargs,
    ) -> None:
        super().__init__()
        self.conv_type = conv_type
        self.use_attn = use_attn
        self.n_key_tokens = n_key_tokens
        self.heatmap_grid_size = heatmap_grid_size

        hidden_dims = int(embed_dims * mlp_ratio)
        # Round to nearest power of 2
        hidden_dims = 2 ** round(math.log2(hidden_dims))
        self.hidden_dims = int(hidden_dims)

        # Simplified non-gated projection
        self.down_proj = nn.Linear(embed_dims, hidden_dims)
        trunc_normal_init(self.down_proj, std=0.02, bias=0)

        self.act = nn.GELU()

        self.up_proj = nn.Linear(hidden_dims, embed_dims)
        constant_init(self.up_proj, 0)

        self.gamma = nn.Parameter(torch.ones(1))

        # Gate for heatmap information injection
        if self.n_key_tokens > 1 and self.use_attn == 0:
            self.heatmap_gate = nn.Parameter(torch.zeros(1))

        if self.use_attn:
            self.attn_norm = nn.LayerNorm(hidden_dims)
            self.attn_gamma = nn.Parameter(torch.ones(1))

            if self.use_attn >= 4:
                self.num_heads = max(1, hidden_dims // 64)

                self.qkv = nn.Linear(hidden_dims, hidden_dims * 3, bias=False)
                trunc_normal_init(self.qkv, std=0.02)

                self.attn_proj = nn.Linear(hidden_dims, hidden_dims)
                trunc_normal_init(self.attn_proj, std=0.02)

                # Add MLP to match AdaTAD++ TransEnc structure
                self.attn_mlp = nn.Sequential(
                    nn.LayerNorm(hidden_dims),
                    nn.Linear(hidden_dims, hidden_dims),
                    nn.GELU(),
                    nn.Linear(hidden_dims, hidden_dims),  # Allow projection back
                )
                # Initialized to near-identity/zero impact initially
                trunc_normal_init(self.attn_mlp[1], std=0.02)
                constant_init(self.attn_mlp[3], 0)
            else:
                # Use a slightly smaller dim for attention efficiency
                attn_dim = hidden_dims // 2
                self.num_heads = 4  # Explicitly define heads

                self.down_attn = nn.Linear(hidden_dims, attn_dim)
                trunc_normal_init(self.down_attn, std=0.02)

                self.qkv = nn.Linear(attn_dim, attn_dim * 3)
                self.q_norm = nn.LayerNorm(attn_dim // self.num_heads)
                self.k_norm = nn.LayerNorm(attn_dim // self.num_heads)
                trunc_normal_init(self.qkv, std=0.02)

                self.attn_proj = nn.Linear(attn_dim, hidden_dims)
                trunc_normal_init(self.attn_proj, std=0.02)

        if self.conv_type == "sparse_conv":
            print("Using Sparse Convolution")
            from ..bricks import SparseConv2d

            self.sparse_conv = SparseConv2d(hidden_dims, hidden_dims)
        elif self.conv_type == "2d_conv":
            print("Using 2D Convolution")
            self.conv = nn.Conv2d(
                hidden_dims,
                hidden_dims,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )
            kaiming_init(self.conv, mode="fan_in", nonlinearity="relu")
        elif self.conv_type == "temporal_dwconv":
            print("Using 1D Temporal Depthwise Separable Convolution")
            self.temporal_dwconv = nn.Conv1d(
                hidden_dims,
                hidden_dims,
                kernel_size=temporal_kernel_size,
                stride=1,
                padding=(temporal_kernel_size // 2) * dilation,
                dilation=dilation,
                groups=hidden_dims,
            )
            self.temporal_dwconv.weight.data.normal_(
                mean=0.0, std=math.sqrt(2.0 / temporal_kernel_size)
            )
            self.temporal_dwconv.bias.data.zero_()
            self.temporal_pwconv = nn.Conv1d(hidden_dims, hidden_dims, 1)
            self.temporal_pwconv.weight.data.normal_(
                mean=0.0, std=math.sqrt(2.0 / hidden_dims)
            )
            self.temporal_pwconv.bias.data.zero_()
        elif self.conv_type == "deformable_conv_t":
            print(
                f"Using TEMPORAL Deformable Convolution ((2+1)D) with {deformable_groups} groups"
            )
            assert (
                hidden_dims % deformable_groups == 0
            ), "hidden_dims must be divisible by deformable_groups."

            # 1. Spatial Deformable Convolution (same as before)
            self.offset_conv = nn.Conv2d(
                hidden_dims,
                deformable_groups * 2 * kernel_size * kernel_size,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )
            constant_init(self.offset_conv, 0)

            self.mask_conv = nn.Conv2d(
                hidden_dims,
                deformable_groups * 1 * kernel_size * kernel_size,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )
            constant_init(self.mask_conv, 0)

            self.deform_conv = DeformConv2d(
                hidden_dims,
                hidden_dims,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=False,
                groups=deformable_groups,
            )
            kaiming_init(self.deform_conv, mode="fan_in", nonlinearity="relu")

            # 2. Temporal 1D Convolution (applied after the spatial one)
            self.temporal_conv = nn.Conv1d(
                hidden_dims,
                hidden_dims,
                kernel_size=temporal_kernel_size,
                padding=(temporal_kernel_size // 2) * dilation,
                dilation=dilation,
                groups=hidden_dims,  # Using a depthwise convolution for efficiency
            )
            self.temporal_conv.weight.data.normal_(
                mean=0.0, std=math.sqrt(2.0 / self.temporal_conv.kernel_size[0])
            )
            self.temporal_conv.bias.data.zero_()
            self.temporal_pw_conv = nn.Conv1d(hidden_dims, hidden_dims, 1)  # Pointwise
            self.temporal_pw_conv.weight.data.normal_(
                mean=0.0, std=math.sqrt(2.0 / self.hidden_dims)
            )
            self.temporal_pw_conv.bias.data.zero_()
        elif self.conv_type == "freq_conv":
            print("Using Temporal-Frequency Convolution (Frame2Freq-style)")
            self.freq_windows = freq_windows
            # Split hidden channels into a temporal branch and a frequency branch
            Ct = hidden_dims // 2
            Cf = hidden_dims - Ct
            self.freq_Ct, self.freq_Cf = Ct, Cf

            # Temporal branch: depthwise Conv3D over (T, H, W)
            self.temporal_conv = nn.Conv3d(
                Ct,
                Ct,
                kernel_size=freq_kernel_size,
                padding=tuple(k // 2 for k in freq_kernel_size),
                groups=Ct,
            )
            # Frequency branch: depthwise Conv3D over the (real, imag) parts of the
            # multi-resolution rFFT (along the temporal axis) of each window
            self.freq_conv = nn.Conv3d(
                2 * Cf,
                2 * Cf,
                kernel_size=freq_kernel_size,
                padding=tuple(k // 2 for k in freq_kernel_size),
                groups=2 * Cf,
            )
            nn.init.constant_(self.temporal_conv.bias, 0.0)
            nn.init.constant_(self.freq_conv.bias, 0.0)
        else:
            raise ValueError(
                f"Unknown conv_type: '{self.conv_type}' in EfficientAdapter. Supported: "
                "'sparse_conv', '2d_conv', 'temporal_dwconv', 'deformable_conv_t', 'freq_conv'"
            )

    def _freq_windows_for(self, T: int) -> List[int]:
        """Multi-resolution FFT window sizes (in tokens) for a temporal length T."""
        if self.freq_windows is not None:
            windows = list(self.freq_windows)
        else:
            windows = [w for w in (T, T // 2, T // 4) if w >= 1]
        for w in windows:
            assert T % w == 0, f"freq_conv: T={T} not divisible by window={w}"
        return windows

    def forward(
        self,
        x: Tensor,
        h: int,
        w: int,
        indices: Tensor,
        total_num_patches: int,
        neighbor_indices: Optional[Tensor] = None,
    ) -> Tensor:
        inputs = x
        B, N, C = x.shape
        # Ensure indices are on the same device as inputs
        if indices is not None and indices.device != x.device:
            indices = indices.to(x.device)
        if neighbor_indices is not None and neighbor_indices.device != x.device:
            neighbor_indices = neighbor_indices.to(x.device)

        x_proj = self.down_proj(x)
        x_proj = self.act(x_proj)
        num_special_tokens = N - indices.shape[1]
        special_tokens_proj = x_proj[:, :num_special_tokens, :]
        patch_tokens_proj = x_proj[:, num_special_tokens:, :]

        # Inject heatmap information if available
        if (
            self.n_key_tokens > 1
            and self.heatmap_grid_size is not None
            and self.use_attn == 0
        ):
            # Extract heatmap tokens (assuming CLS is at 0, heatmap at 1:)
            heatmap_tokens = special_tokens_proj[:, 1:, :]
            H_hm, W_hm = self.heatmap_grid_size

            # Calculate spatial indices
            spatial_indices = indices % (h * w)  # [B, N_kept]

            # Check if resize is needed
            if (H_hm != h) or (W_hm != w):
                # Nearest Neighbor Index Mapping
                # Map (h, w) indices to (H_hm, W_hm) indices
                y = spatial_indices // w
                x_coord = spatial_indices % w

                # Scale to [0, H_hm-1], [0, W_hm-1]
                y_hm = (y * H_hm) // h
                x_hm = (x_coord * W_hm) // w

                spatial_indices_hm = y_hm * W_hm + x_hm

                # Gather from original heatmap tokens
                sampled_feats = torch.gather(
                    heatmap_tokens,
                    1,
                    spatial_indices_hm.unsqueeze(-1).expand(-1, -1, self.hidden_dims),
                )
            else:
                # Direct gather
                sampled_feats = torch.gather(
                    heatmap_tokens,
                    1,
                    spatial_indices.unsqueeze(-1).expand(-1, -1, self.hidden_dims),
                )

            # Add to patch tokens
            patch_tokens_proj = patch_tokens_proj + sampled_feats * self.heatmap_gate

        if patch_tokens_proj.shape[1] == 0:
            processed_tokens = self.up_proj(self.act(x_proj))
            return processed_tokens * self.gamma + inputs

        if self.conv_type == "sparse_conv":
            processed_patches = self.sparse_conv(
                patch_tokens_proj,
                neighbor_indices,
            )

        else:  # 2d_conv, temporal_dwconv, deformable_conv_t, or freq_conv
            num_temporal_steps_full = total_num_patches // (h * w)
            no_pruning = indices.shape[1] == total_num_patches

            if no_pruning:
                # No token selection done - use patch_tokens_proj directly
                conv_input_tokens = patch_tokens_proj
            else:
                # Scatter to full grid for convolution
                full_grid = torch.zeros(
                    B,
                    total_num_patches,
                    self.hidden_dims,
                    device=patch_tokens_proj.device,
                    dtype=patch_tokens_proj.dtype,
                )
                indices_expanded = indices.unsqueeze(-1).expand(
                    -1, -1, self.hidden_dims
                )
                conv_input_tokens = full_grid.scatter_(
                    1, indices_expanded, patch_tokens_proj
                )

            if self.conv_type == "temporal_dwconv":
                # Reshape to (B, T, H, W, C)
                conv_input = conv_input_tokens.reshape(
                    B, num_temporal_steps_full, h, w, self.hidden_dims
                )
                # Permute and flatten to (B*H*W, C, T) for Conv1d
                conv_input = conv_input.permute(0, 2, 3, 4, 1).flatten(0, 2)
                conv_output = self.temporal_pwconv(self.temporal_dwconv(conv_input))
                # Reverse to (B, T*H*W, C)
                processed_full_sequence = (
                    conv_output.unflatten(0, (B, h, w))
                    .permute(0, 4, 1, 2, 3)
                    .reshape(B, -1, self.hidden_dims)
                )
            elif self.conv_type == "2d_conv":
                # Reshape to (B*T, C, H, W) for 2D Conv
                conv_input = conv_input_tokens.reshape(
                    B * num_temporal_steps_full, h, w, self.hidden_dims
                )
                conv_input = conv_input.permute(0, 3, 1, 2).contiguous()

                conv_output = self.conv(conv_input)

                processed_full_sequence = conv_output.permute(0, 2, 3, 1).reshape(
                    B, -1, self.hidden_dims
                )
            elif self.conv_type == "deformable_conv_t":
                # 1. Perform Spatial Deformable Convolution on each frame
                # Reshape to (B*T, C, H, W)
                conv_input_2d = conv_input_tokens.reshape(
                    B * num_temporal_steps_full, h, w, self.hidden_dims
                )
                conv_input_2d = conv_input_2d.permute(0, 3, 1, 2).contiguous()

                offset = self.offset_conv(conv_input_2d)
                modulation_mask = torch.sigmoid(self.mask_conv(conv_input_2d))
                spatial_out = self.deform_conv(
                    conv_input_2d, offset, mask=modulation_mask
                )

                # 2. Prepare for Temporal Convolution
                # Reshape from (B*T, C, H, W) back to (B, T, H, W, C)
                temporal_input = spatial_out.permute(0, 2, 3, 1).view(
                    B, num_temporal_steps_full, h, w, self.hidden_dims
                )
                # Permute and flatten to (B*H*W, C, T) for Conv1d
                temporal_input = temporal_input.permute(0, 2, 3, 4, 1).flatten(0, 2)

                # 3. Perform Temporal Convolution
                temporal_out = self.temporal_pw_conv(self.temporal_conv(temporal_input))

                # 4. Reshape back to the original sequence format
                # Unflatten from (B*H*W, C, T) back to (B, H, W, C, T)
                processed_full_sequence = temporal_out.unflatten(0, (B, h, w))
                # Permute to (B, T, H, W, C) then flatten to (B, T*H*W, C)
                processed_full_sequence = processed_full_sequence.permute(
                    0, 4, 1, 2, 3
                ).reshape(B, -1, self.hidden_dims)
            elif self.conv_type == "freq_conv":
                # Reshape to (B, T, H, W, C) and split into temporal/frequency channels
                x5 = conv_input_tokens.reshape(
                    B, num_temporal_steps_full, h, w, self.hidden_dims
                )
                Ct, Cf = self.freq_Ct, self.freq_Cf
                xt, xf = x5[..., :Ct], x5[..., Ct:]
                orig_dtype = x5.dtype

                # Depthwise Conv3d has no BFloat16/Half CUDA kernel, so run both
                # conv branches in float32 with autocast disabled, then cast back.
                with torch.autocast(device_type=xt.device.type, enabled=False):
                    # Temporal branch: depthwise Conv3D over (T, H, W)
                    xt = xt.permute(0, 4, 1, 2, 3).float()  # (B, Ct, T, H, W)
                    xt = self.temporal_conv(xt)
                    xt = xt.permute(0, 2, 3, 4, 1).to(orig_dtype)  # (B, T, H, W, Ct)

                    # Frequency branch: multi-resolution rFFT along T + depthwise Conv3D
                    T = num_temporal_steps_full
                    windows = self._freq_windows_for(T)
                    xf = xf.float()
                    out_f = torch.zeros_like(xf)
                    for win in windows:
                        chunks = T // win
                        xf_chunk = xf.reshape(B, chunks, win, h, w, Cf)
                        X = torch.fft.rfft(xf_chunk, dim=2)
                        X = torch.view_as_real(X).contiguous()
                        B_, chunks_, Freq, H_, W_, d, _ = X.shape
                        X = X.reshape(B_ * chunks_, 2 * d, Freq, H_, W_)
                        Y = self.freq_conv(X)
                        Y = Y.reshape(B, chunks, Freq, h, w, d, 2)
                        Y = torch.complex(Y[..., 0], Y[..., 1])
                        y = torch.fft.irfft(Y, n=win, dim=2)
                        out_f = out_f + y.reshape(B, T, h, w, Cf)
                    out_f = (out_f / len(windows)).to(orig_dtype)

                out = torch.cat([xt, out_f], dim=-1)  # (B, T, H, W, hidden_dims)
                processed_full_sequence = out.reshape(B, -1, self.hidden_dims)

            if no_pruning:
                # No gather needed - output is already in correct order
                processed_patches = processed_full_sequence
            else:
                processed_patches = torch.gather(
                    processed_full_sequence, 1, indices_expanded
                )

        x_combined = torch.cat([special_tokens_proj, processed_patches], dim=1)

        if self.use_attn:
            shortcut = x_combined

            if self.use_attn >= 4:

                # --- Temporal Global Attention Upgrade (AdaTAD++) ---
                # 1. Aggregate patches to frame-level tokens
                num_patches_per_frame = h * w
                # Shape: [B, N_kept]
                frame_indices = indices // num_patches_per_frame
                num_frames = total_num_patches // num_patches_per_frame

                # Initialize [B, T, C]
                B_current, N_kept, C_hid = processed_patches.shape
                frame_feats = torch.zeros(
                    B_current,
                    num_frames,
                    C_hid,
                    device=processed_patches.device,
                    dtype=processed_patches.dtype,
                )
                frame_counts = torch.zeros(
                    B_current,
                    num_frames,
                    1,
                    device=processed_patches.device,
                    dtype=processed_patches.dtype,
                )

                # Scatter Add
                idx_expanded = frame_indices.unsqueeze(-1).expand(-1, -1, C_hid)
                frame_feats = frame_feats.scatter_add(
                    1, idx_expanded, processed_patches
                )

                # Count
                ones = torch.ones_like(
                    frame_indices, dtype=processed_patches.dtype
                ).unsqueeze(-1)
                frame_counts = frame_counts.scatter_add(
                    1, frame_indices.unsqueeze(-1), ones
                )

                # Mean Pooling (safe division)
                frame_feats = frame_feats / (frame_counts + 1e-6)

                # 2. Append CLS token (and others) for joint attention
                # special_tokens_proj shape: [B, N_spec, C]
                # attn_input: [B, N_spec + T, C]
                attn_input = torch.cat([special_tokens_proj, frame_feats], dim=1)

                # 3. Apply Temporal Transformer Block (Norm -> Attn -> Norm -> MLP)
                x_attn = self.attn_norm(attn_input)

                # QKV Projection
                qkv = self.qkv(x_attn)
                # Shape: [B, N_tokens, 3, Heads, Dim] => [3, B, Heads, N_tokens, Dim]
                qkv = qkv.reshape(
                    B_current, -1, 3, self.num_heads, C_hid // self.num_heads
                ).permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]

                # Scaled Dot Product Attention
                x_attn_out = F.scaled_dot_product_attention(q, k, v)

                # Reshape back: [B, N_tokens, C]
                x_attn_out = x_attn_out.transpose(1, 2).reshape(B_current, -1, C_hid)

                # Projection
                x_attn_out = self.attn_proj(x_attn_out)

                # Add Residual to Input (Note: Input to attention was aggregated,
                # but we want to add result to original fine-grained features)
                # Structure: x = x + Attn(Norm(x)) -> x = x + MLP(Norm(x))

                # Since we aggregated, we can't do simple residual on the input sequence directly here.
                # Instead, we apply the MLP on the ATTENDED sequence, then Broadcast back.

                # Apply MLP
                # First add residual to the aggregated sequence itself?
                # AdaTAD++ Eq 8: Ftemp = TransEnc2(Aggr(F)).
                # Then F = F + Broadcast(Ftemp).
                # So the Attn + MLP block happens on the coarse level.

                x_coarse_res = x_attn_out  # Result of attention

                x_coarse_res = x_coarse_res + self.attn_mlp(x_coarse_res)

                # 4. Separate Special and Frame tokens
                num_spec = special_tokens_proj.shape[1]
                special_tokens_out = x_coarse_res[:, :num_spec, :]
                frame_feats_out = x_coarse_res[:, num_spec:, :]

                # 5. Broadcast frame features back to patches
                # Gather using frame_indices: [B, N_kept, C]
                idx_gather = frame_indices.unsqueeze(-1).expand(-1, -1, C_hid)
                patch_tokens_out = torch.gather(frame_feats_out, 1, idx_gather)

                # 6. Recombine and Add to original shortcut
                # x_combined = shortcut + broadcast(attn_result)
                x_attn_broadcast = torch.cat(
                    [special_tokens_out, patch_tokens_out], dim=1
                )
                x_combined = shortcut + x_attn_broadcast

            else:
                x_attn = self.attn_norm(x_combined)
                x_attn = self.down_attn(x_attn)
                B_attn, N_attn, C_attn = x_attn.shape

                # Use self.num_heads instead of hardcoded 4
                head_dim = C_attn // self.num_heads
                qkv = self.qkv(x_attn)
                q, k, v = qkv.reshape(
                    B_attn, N_attn, 3, self.num_heads, head_dim
                ).permute(2, 0, 3, 1, 4)

                # Apply stable QK Norm
                q = self.q_norm(q)
                k = self.k_norm(k)

                if self.use_attn == 2:
                    # Option 2: Key tokens -> Visual tokens
                    num_keys = self.n_key_tokens
                    q_vis = q[:, :, num_keys:, :]
                    k_key = k[:, :, :num_keys, :]
                    v_key = v[:, :, :num_keys, :]

                    # Visual tokens attend to key tokens
                    x_attn_vis = F.scaled_dot_product_attention(q_vis, k_key, v_key)

                    # Key tokens attend to nothing (no update)
                    x_attn_key = torch.zeros_like(q[:, :, :num_keys, :])

                    x_attn = torch.cat([x_attn_key, x_attn_vis], dim=2)
                elif self.use_attn == 3:
                    # Option 3: Visual tokens -> Key tokens
                    num_keys = self.n_key_tokens
                    q_key = q[:, :, :num_keys, :]
                    k_vis = k[:, :, num_keys:, :]
                    v_vis = v[:, :, num_keys:, :]

                    # Key tokens attend to visual tokens
                    x_attn_key = F.scaled_dot_product_attention(q_key, k_vis, v_vis)

                    # Visual tokens attend to nothing (no update)
                    x_attn_vis = torch.zeros_like(q[:, :, num_keys:, :])

                    x_attn = torch.cat([x_attn_key, x_attn_vis], dim=2)
                else:
                    x_attn = F.scaled_dot_product_attention(q, k, v)

                x_attn = x_attn.transpose(1, 2).reshape(B_attn, N_attn, C_attn)
                x_attn = self.attn_proj(x_attn)

                # Use learned gamma scaling
                x_combined = shortcut + x_attn * self.attn_gamma

        x = self.up_proj(x_combined)
        return x * self.gamma + inputs


class KTPAttention(BaseModule):
    def __init__(
        self,
        embed_dims: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop_rate: float = 0.0,
        drop_rate: float = 0.0,
        init_cfg: OptConfigType = None,
        keep_rate: float = 1.0,
        enhanced_weight_class: int = 1,
        n_key_tokens: int = 1,
        selector: str = "attention",
        **kwargs,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        head_embed_dims = embed_dims // num_heads
        self.scale = qk_scale or head_embed_dims**-0.5

        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(embed_dims))
            self.v_bias = nn.Parameter(torch.zeros(embed_dims))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(drop_rate)

        self.keep_rate = keep_rate
        assert 0 < keep_rate <= 1, f"keep_rate must > 0 and <= 1, got {keep_rate}"
        assert selector in (
            "attention",
            "norm",
        ), f"selector must be 'attention' or 'norm', got {selector}"
        self.selector = selector
        self.enhanced_weight_class = enhanced_weight_class
        self.n_key_tokens = n_key_tokens
        # Learnable weights for each key token's contribution to importance scoring
        if n_key_tokens > 1:
            # Weights for [CLS] + heatmap tokens, initialized so CLS has weight 1 and heatmaps start smaller
            self.key_token_weights = nn.Parameter(torch.ones(n_key_tokens))
            # Initialize heatmap weights to a smaller value initially
            with torch.no_grad():
                self.key_token_weights[1:] = 0.5  # Heatmap tokens start with 0.5 weight

    def forward_part1(self, x: torch.Tensor):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat(
                (
                    self.q_bias,
                    torch.zeros_like(self.v_bias, requires_grad=False),
                    self.v_bias,
                )
            )
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.keep_rate >= 1:
            # use flash attention
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
            x = x.transpose(1, 2).reshape(B, N, -1)
            attn = None
        elif self.selector == "norm":
            # L2-norm of (layer-normalised) input features as importance scores.
            # Capture before x is overwritten by the attention output.
            norm_scores = x.norm(dim=-1)  # [B, N]
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
            x = x.transpose(1, 2).reshape(B, N, -1)
            attn = norm_scores  # [B, N] — consumed directly in forward()
        else:
            # use flash attention for output
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
            x = x.transpose(1, 2).reshape(B, N, -1)

            # calculate attention map for pruning
            q_subset = q[:, :, : self.n_key_tokens]

            attn = (q_subset * self.scale) @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

    def forward(
        self,
        x: torch.Tensor,
        last_idx: Optional[torch.Tensor] = None,
        n_extra_key: int = 0,
        return_scores: bool = False,
    ):
        B, N, C = x.shape
        x, attn = self.forward_part1(x)

        if self.keep_rate >= 1:
            return x, last_idx, None

        # num_s_tokens accounts for CLS (+ any EViT fused tokens)
        num_s_tokens = self.n_key_tokens + n_extra_key
        num_keep_tokens = math.ceil(self.keep_rate * (N - num_s_tokens))

        if num_keep_tokens <= 0:
            return x, torch.empty(B, 0, dtype=last_idx.dtype, device=x.device), None

        # Compute importance scores
        if self.selector == "norm":
            # attn is already [B, N] L2-norm scores; use directly
            attn_topk = attn
        elif self.n_key_tokens > 1:
            weights = self.key_token_weights * self.enhanced_weight_class
            weights = weights.view(1, 1, -1, 1)
            attn_topk = (attn * weights).sum(dim=-2).mean(dim=1)
        else:
            attn_topk = attn.sum(dim=-2).mean(dim=1)
            if self.enhanced_weight_class != 1:
                cls_boost = attn[:, :, 0].mean(dim=1) * (self.enhanced_weight_class - 1)
                attn_topk = attn_topk + cls_boost

        # Slice to cover only the current patch tokens (skip special + extra key tokens)
        attn_topk = attn_topk[:, num_s_tokens:]
        num_keep_tokens = min(num_keep_tokens, attn_topk.shape[1])
        _, idx_evad = torch.topk(attn_topk, num_keep_tokens, dim=1, largest=True)

        idx = idx_evad.sort(dim=1)[0]
        # attn_topk holds scores for all patch tokens (local indices); return when asked
        return x, idx, (attn_topk if return_scores else None)


class Block(BaseModule):
    """
    The basic block in the Vision Transformer with token pruning and merging.
    """

    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        act_cfg: ConfigType = dict(type="GELU"),
        norm_cfg: ConfigType = dict(type="LN", eps=1e-6),
        init_cfg: OptConfigType = None,
        with_cp: bool = False,
        keep_rate: float = 1.0,
        n_key_tokens: int = 1,
        use_adapter: bool = False,
        adapter_mlp_ratio: float = 0.25,
        adapter_conv_type: str = "sparse_conv",
        deform_conv_groups: int = 1,
        adapter_use_attn: Union[bool, int] = False,
        **kwargs,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.with_cp = with_cp
        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.n_key_tokens = n_key_tokens
        self.keep_rate = keep_rate
        self.num_heads = num_heads
        self.use_adapter = use_adapter
        self.adapter_use_attn = adapter_use_attn
        self.adapter_conv_type = adapter_conv_type
        self._adapter_time_ms = 0.0  # Accumulated adapter time in ms
        self._block_time_ms = 0.0  # Accumulated block time in ms
        # Only sparse_conv needs neighbor_indices
        self.needs_neighbor_indices = use_adapter and adapter_conv_type == "sparse_conv"

        self.attn = KTPAttention(
            embed_dims,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            drop_rate=drop_rate,
            keep_rate=keep_rate,
            n_key_tokens=n_key_tokens,
            **kwargs,
        )

        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        )
        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]
        mlp_hidden_dim = int(embed_dims * mlp_ratio)
        self.mlp = FFN(
            embed_dims=embed_dims,
            feedforward_channels=mlp_hidden_dim,
            act_cfg=act_cfg,
            ffn_drop=drop_rate,
            add_identity=False,
        )
        if self.use_adapter:
            self.adapter = EfficientAdapter(
                embed_dims=embed_dims,
                mlp_ratio=adapter_mlp_ratio,
                conv_type=adapter_conv_type,
                deformable_groups=deform_conv_groups,
                use_attn=self.adapter_use_attn,
                n_key_tokens=n_key_tokens,
                **kwargs,
            )

    def forward(
        self,
        x: torch.Tensor,
        last_idx: torch.Tensor,
        h: int,
        w: int,
        total_num_patches: int,
        neighbor_indices: Optional[torch.Tensor] = None,
        measure_adapter_time: bool = False,
        measure_block_time: bool = False,
        global_mapping_buffer: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:

        def _inner_forward(x, last_idx, neighbor_indices):
            # Enable timer only for the first batch to avoid spam
            enable_timer = False  # Set to True to debug

            with CudaTimer("Block Total", enabled=enable_timer):
                B, N, C = x.shape
                x_norm = self.norm1(x)

                with CudaTimer("Attention", enabled=enable_timer):
                    attn_x, idx, _ = self.attn(x_norm, last_idx)

                x = x + self.drop_path(attn_x)

                num_s_tokens = self.n_key_tokens

                if self.keep_rate < 1:
                    with CudaTimer("Pruning/Merging Logic", enabled=enable_timer):
                        x_key = x[:, :num_s_tokens]
                        x_nonkey = x[:, num_s_tokens:]

                        idx_expanded = idx.unsqueeze(-1).expand(-1, -1, C)
                        x_nonkey_keep = torch.gather(
                            x_nonkey, dim=1, index=idx_expanded
                        )

                        idx_from_topk = idx
                        updated_global_idx = torch.gather(last_idx, dim=1, index=idx)

                        x = torch.cat([x_key, x_nonkey_keep], dim=1)
                        idx = updated_global_idx

                        # Only update neighbor_indices if needed (sparse_conv uses it)
                        if neighbor_indices is not None:
                            with CudaTimer("Neighbor Update", enabled=enable_timer):
                                B, N_prev, _ = x_nonkey.shape
                                N_new = x_nonkey_keep.shape[1]

                                # Gather kept neighbor indices
                                idx_expand_9 = idx_from_topk.unsqueeze(-1).expand(
                                    -1, -1, 9
                                )
                                neighbor_indices_kept = torch.gather(
                                    neighbor_indices, 1, idx_expand_9
                                )

                                # Reuse pre-allocated buffer and reset to -1
                                global_mapping = global_mapping_buffer[: B * N_prev]
                                global_mapping.fill_(-1)

                                # Compute offsets for batch indexing
                                offsets_old = (
                                    torch.arange(B, device=x.device, dtype=torch.int32)
                                    * N_prev
                                ).view(B, 1)
                                old_global_kept = (
                                    idx_from_topk.to(torch.int32) + offsets_old
                                )

                                offsets_new = (
                                    torch.arange(B, device=x.device, dtype=torch.int32)
                                    * N_new
                                ).view(B, 1)
                                new_local_kept = (
                                    torch.arange(
                                        idx_from_topk.shape[1],
                                        device=x.device,
                                        dtype=torch.int32,
                                    )
                                    .unsqueeze(0)
                                    .expand(B, -1)
                                )
                                new_global_kept = new_local_kept + offsets_new

                                global_mapping[old_global_kept.reshape(-1).long()] = (
                                    new_global_kept.reshape(-1)
                                )

                                # Update neighbor_indices in-place without clone
                                # Map OLD GLOBAL indices to NEW GLOBAL indices
                                valid_mask = neighbor_indices_kept != -1

                                # Use masked indexing to avoid clone
                                flat_neighbors = neighbor_indices_kept.reshape(-1)
                                flat_valid = valid_mask.reshape(-1)

                                # Only lookup valid indices
                                new_flat = torch.where(
                                    flat_valid,
                                    global_mapping[
                                        torch.clamp(flat_neighbors.long(), min=0)
                                    ],
                                    torch.tensor(
                                        -1, device=x.device, dtype=torch.int32
                                    ),
                                )
                                neighbor_indices = new_flat.view(B, N_new, 9)

                x_post_mlp = self.mlp(self.norm2(x))
                x = x + self.drop_path(x_post_mlp)
                if self.use_adapter:
                    if measure_adapter_time:
                        start_event = torch.cuda.Event(enable_timing=True)
                        end_event = torch.cuda.Event(enable_timing=True)
                        start_event.record()
                        x = self.adapter(
                            x, h, w, idx, total_num_patches, neighbor_indices
                        )
                        end_event.record()
                        torch.cuda.synchronize()
                        self._adapter_time_ms += start_event.elapsed_time(end_event)
                    else:
                        with CudaTimer("Adapter", enabled=enable_timer):
                            x = self.adapter(
                                x, h, w, idx, total_num_patches, neighbor_indices
                            )
                return x, idx, neighbor_indices

        if measure_block_time:
            block_start = torch.cuda.Event(enable_timing=True)
            block_end = torch.cuda.Event(enable_timing=True)
            block_start.record()

        if self.with_cp and x.requires_grad:
            x, idx, neighbor_indices = cp.checkpoint(
                _inner_forward, x, last_idx, neighbor_indices, use_reentrant=False
            )
        else:
            x, idx, neighbor_indices = _inner_forward(x, last_idx, neighbor_indices)

        if measure_block_time:
            block_end.record()
            torch.cuda.synchronize()
            self._block_time_ms += block_start.elapsed_time(block_end)

        return x, idx, neighbor_indices


class CudaTimer:
    def __init__(self, name, enabled=True):
        self.name = name
        self.enabled = enabled
        if enabled:
            self.start = torch.cuda.Event(enable_timing=True)
            self.end = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        if self.enabled:
            self.start.record()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enabled:
            self.end.record()
            torch.cuda.synchronize()
            print(f"[Timer] {self.name}: {self.start.elapsed_time(self.end):.3f} ms")


class EViTBlock(Block):
    """
    EViT-style block that fuses dropped tokens into a single representative token
    instead of discarding them (Liang et al., ICLR 2022).

    The fused token is appended right after the CLS token so that it:
      - participates in subsequent self-attention,
      - is treated as a special token by EfficientAdapter (skipped by SparseConv2D).

    ``forward()`` accepts an optional incoming ``fused_token`` (from previous pruning
    rounds) and returns an updated one.  Non-pruning blocks simply pass the fused
    token through unchanged.
    """

    def forward(
        self,
        x: torch.Tensor,
        last_idx: torch.Tensor,
        h: int,
        w: int,
        total_num_patches: int,
        neighbor_indices: Optional[torch.Tensor] = None,
        measure_adapter_time: bool = False,
        measure_block_time: bool = False,
        global_mapping_buffer: Optional[torch.Tensor] = None,
        fused_token: Optional[torch.Tensor] = None,
    ) -> tuple:
        # 1 if a fused token is currently in the sequence, else 0
        n_extra_key = 1 if fused_token is not None else 0

        def _inner_forward(x, last_idx, neighbor_indices, fused_token):
            enable_timer = False

            with CudaTimer("EViT Block Total", enabled=enable_timer):
                B, N, C = x.shape
                x_norm = self.norm1(x)

                with CudaTimer("Attention", enabled=enable_timer):
                    # Pass n_extra_key so CLS-attention scoring skips the fused token
                    # return_scores=True only needed at pruning layers
                    attn_x, idx, attn_topk = self.attn(
                        x_norm,
                        last_idx,
                        n_extra_key=n_extra_key,
                        return_scores=(self.keep_rate < 1),
                    )

                x = x + self.drop_path(attn_x)

                # n_key_tokens_eff: CLS tokens (original) + optional fused token
                num_s_tokens_orig = self.n_key_tokens  # CLS only
                num_s_tokens_eff = num_s_tokens_orig + n_extra_key  # + fused

                if self.keep_rate < 1:
                    with CudaTimer("EViT Pruning", enabled=enable_timer):
                        x_key = x[:, :num_s_tokens_orig]  # [B, 1, C]  CLS
                        x_nonkey = x[:, num_s_tokens_eff:]  # [B, N_patch, C]

                        # idx is local within x_nonkey (0..N_patch-1)
                        idx_expanded = idx.unsqueeze(-1).expand(-1, -1, C)
                        x_nonkey_keep = torch.gather(
                            x_nonkey, dim=1, index=idx_expanded
                        )

                        # --- EViT fusion of dropped tokens ---
                        N_patch = x_nonkey.shape[1]
                        N_keep = idx.shape[1]
                        N_drop = N_patch - N_keep

                        if N_drop > 0:
                            # Build boolean keep mask [B, N_patch]
                            keep_mask = torch.zeros(
                                B, N_patch, device=x.device, dtype=torch.bool
                            )
                            keep_mask.scatter_(1, idx, True)

                            # Scores for dropped tokens (attn_topk is [B, N_patch])
                            scores_drop = attn_topk[~keep_mask].view(B, N_drop)
                            x_drop = x_nonkey[~keep_mask].view(B, N_drop, C)

                            # Softmax-weighted average (EViT Eq. 2)
                            weights = F.softmax(scores_drop, dim=-1).unsqueeze(-1)
                            new_fused = (x_drop * weights).sum(
                                dim=1, keepdim=True
                            )  # [B,1,C]

                            # Blend with carry-over fused token (if any)
                            if fused_token is not None:
                                fused_token = 0.5 * fused_token + 0.5 * new_fused
                            else:
                                fused_token = new_fused
                        # (if N_drop == 0 we keep fused_token as-is)

                        # Rebuild sequence: [CLS, fused (if any), kept_patches]
                        if fused_token is not None:
                            x = torch.cat([x_key, fused_token, x_nonkey_keep], dim=1)
                        else:
                            x = torch.cat([x_key, x_nonkey_keep], dim=1)

                        idx_from_topk = idx
                        updated_global_idx = torch.gather(last_idx, dim=1, index=idx)
                        idx = updated_global_idx

                        # Update neighbor_indices for sparse_conv (same logic as Block)
                        if neighbor_indices is not None:
                            with CudaTimer("Neighbor Update", enabled=enable_timer):
                                N_prev = x_nonkey.shape[1]
                                N_new = x_nonkey_keep.shape[1]

                                idx_expand_9 = idx_from_topk.unsqueeze(-1).expand(
                                    -1, -1, 9
                                )
                                neighbor_indices_kept = torch.gather(
                                    neighbor_indices, 1, idx_expand_9
                                )

                                global_mapping = global_mapping_buffer[: B * N_prev]
                                global_mapping.fill_(-1)

                                offsets_old = (
                                    torch.arange(B, device=x.device, dtype=torch.int32)
                                    * N_prev
                                ).view(B, 1)
                                old_global_kept = (
                                    idx_from_topk.to(torch.int32) + offsets_old
                                )

                                offsets_new = (
                                    torch.arange(B, device=x.device, dtype=torch.int32)
                                    * N_new
                                ).view(B, 1)
                                new_local_kept = (
                                    torch.arange(
                                        idx_from_topk.shape[1],
                                        device=x.device,
                                        dtype=torch.int32,
                                    )
                                    .unsqueeze(0)
                                    .expand(B, -1)
                                )
                                new_global_kept = new_local_kept + offsets_new

                                global_mapping[old_global_kept.reshape(-1).long()] = (
                                    new_global_kept.reshape(-1)
                                )

                                valid_mask = neighbor_indices_kept != -1
                                flat_neighbors = neighbor_indices_kept.reshape(-1)
                                flat_valid = valid_mask.reshape(-1)

                                new_flat = torch.where(
                                    flat_valid,
                                    global_mapping[
                                        torch.clamp(flat_neighbors.long(), min=0)
                                    ],
                                    torch.tensor(
                                        -1, device=x.device, dtype=torch.int32
                                    ),
                                )
                                neighbor_indices = new_flat.view(B, N_new, 9)

                else:
                    # Non-pruning block: sequence already has fused token in correct position
                    # idx stays as last_idx (passed through by KTPAttention when keep_rate>=1)
                    idx = last_idx

                x_post_mlp = self.mlp(self.norm2(x))
                x = x + self.drop_path(x_post_mlp)

                if self.use_adapter:
                    if measure_adapter_time:
                        start_event = torch.cuda.Event(enable_timing=True)
                        end_event = torch.cuda.Event(enable_timing=True)
                        start_event.record()
                        x = self.adapter(
                            x, h, w, idx, total_num_patches, neighbor_indices
                        )
                        end_event.record()
                        torch.cuda.synchronize()
                        self._adapter_time_ms += start_event.elapsed_time(end_event)
                    else:
                        x = self.adapter(
                            x, h, w, idx, total_num_patches, neighbor_indices
                        )

            return x, idx, neighbor_indices, fused_token

        if measure_block_time:
            block_start = torch.cuda.Event(enable_timing=True)
            block_end = torch.cuda.Event(enable_timing=True)
            block_start.record()

        if self.with_cp and x.requires_grad:
            x, idx, neighbor_indices, fused_token = cp.checkpoint(
                _inner_forward,
                x,
                last_idx,
                neighbor_indices,
                fused_token,
                use_reentrant=False,
            )
        else:
            x, idx, neighbor_indices, fused_token = _inner_forward(
                x, last_idx, neighbor_indices, fused_token
            )

        if measure_block_time:
            block_end.record()
            torch.cuda.synchronize()
            self._block_time_ms += block_start.elapsed_time(block_end)

        return x, idx, neighbor_indices, fused_token


@MODELS.register_module()
class VisionTransformerSparseAdapterEViT(BaseModule):
    """
    ``VisionTransformerSparseAdapterPOGUISE`` with EViT-style fused-token selection.

    Identical hyper-parameters to the parent class; the only behavioural change is
    that each pruning block fuses dropped tokens into a single representative token
    (EViT, Liang et al., ICLR 2022) instead of discarding them.  The fused token is
    threaded through all subsequent blocks as an extra special token so it continues
    to attend to and be attended by the kept patches.

    Use this class as a drop-in replacement to ablate selector strategy while keeping
    the SparseConv2D adapter unchanged.
    """

    def __init__(self, **kwargs) -> None:
        # Delegate to Block-based __init__ by temporarily patching the block class
        # We re-use the full init logic of VisionTransformerSparseAdapterPOGUISE
        # via composition.
        super().__init__(
            init_cfg=kwargs.pop(
                "init_cfg",
                [
                    dict(type="TruncNormal", layer="Linear", std=0.02, bias=0.0),
                    dict(type="Constant", layer="LayerNorm", val=1.0, bias=0.0),
                ],
            )
        )
        # Instantiate the standard backbone internally and steal its weights
        self._backbone = VisionTransformerSparseAdapterPOGUISE(**kwargs)

        # Replace every Block with an EViTBlock (same weights, same config)
        new_blocks = ModuleList()
        for blk in self._backbone.blocks:
            evit_blk = EViTBlock.__new__(EViTBlock)
            evit_blk.__dict__.update(blk.__dict__)
            # Copy nn.Module state (parameters, buffers, sub-modules)
            evit_blk._parameters = blk._parameters
            evit_blk._buffers = blk._buffers
            evit_blk._modules = blk._modules
            new_blocks.append(evit_blk)
        self._backbone.blocks = new_blocks

    def forward(self, x: torch.Tensor):
        bb = self._backbone
        if bb.use_adapter:
            bb._freeze_layers()

        b, _, _, h, w = x.shape
        h_patch = h // bb.patch_size
        w_patch = w // bb.patch_size
        x, _ = bb.patch_embed(x)

        current_total_patches = x.shape[1]

        if (h_patch, w_patch) != bb.grid_size:
            pos_embed = bb.pos_embed.reshape(-1, *bb.grid_size, bb.embed_dims)
            pos_embed = pos_embed.permute(0, 3, 1, 2)
            pos_embed = F.interpolate(
                pos_embed, size=(h_patch, w_patch), mode="bicubic", align_corners=False
            )
            pos_embed = pos_embed.permute(0, 2, 3, 1).flatten(1, 2)
            pos_embed = pos_embed.reshape(1, -1, bb.embed_dims)
        else:
            pos_embed = bb.pos_embed

        x = x + pos_embed
        x = bb.pos_drop(x)
        cls_tokens = bb.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        if bb.n_landmarks > 0:
            heatmap_tokens = bb.heatmap_tokens.expand(b, -1, -1)
            x = torch.cat((x[:, :1, :], heatmap_tokens, x[:, 1:, :]), dim=1)

        idx = (
            torch.arange(0, current_total_patches, device=x.device)
            .unsqueeze(0)
            .repeat(b, 1)
        )

        if bb.needs_neighbor_indices:
            T_patches = current_total_patches // (h_patch * w_patch)
            neighbor_indices = bb.get_initial_neighbor_indices(
                h_patch, w_patch, T_patches, x.device, b
            )
            global_mapping_buffer = torch.empty(
                b * current_total_patches, dtype=torch.int32, device=x.device
            )
        else:
            neighbor_indices = None
            global_mapping_buffer = None

        fused_token: Optional[torch.Tensor] = None
        for blk in bb.blocks:
            x, idx, neighbor_indices, fused_token = blk(
                x,
                idx,
                h_patch,
                w_patch,
                current_total_patches,
                neighbor_indices,
                global_mapping_buffer=global_mapping_buffer,
                fused_token=fused_token,
            )

        x_heatmap = None
        if bb.n_landmarks > 0:
            start_idx = 1
            end_idx = 1 + bb.n_heatmap_tokens
            heatmap_tokens_out = x[:, start_idx:end_idx, :]
            heatmap_feats = heatmap_tokens_out.reshape(
                b, *bb.hw_out_conv, bb.embed_dims
            )
            heatmap_feats = heatmap_feats.permute(0, 3, 1, 2).contiguous()
            x_heatmap = bb.heatmap_head(heatmap_feats)

        x = bb.norm(x)

        # Number of special tokens: CLS (+ heatmap tokens) + fused token (if any)
        num_special_tokens = (
            1 + bb.n_heatmap_tokens + (1 if fused_token is not None else 0)
        )

        final_patch_tokens = x[:, num_special_tokens:, :]
        final_patch_indices = idx

        T_out = bb.num_frames // bb.tubelet_size
        patches_per_frame = h_patch * w_patch
        token_frame_indices = final_patch_indices // patches_per_frame

        feature_sum = torch.zeros(
            b,
            T_out,
            bb.embed_dims,
            device=final_patch_tokens.device,
            dtype=final_patch_tokens.dtype,
        )
        token_count = torch.zeros(
            b,
            T_out,
            1,
            device=final_patch_tokens.device,
            dtype=final_patch_tokens.dtype,
        )

        if final_patch_tokens.shape[1] > 0:
            idx_expanded = token_frame_indices.unsqueeze(-1).expand(
                -1, -1, bb.embed_dims
            )
            feature_sum.scatter_add_(1, idx_expanded, final_patch_tokens)
            idx_mask = token_frame_indices.unsqueeze(-1)
            token_count.scatter_add_(
                1, idx_mask, torch.ones_like(idx_mask, dtype=final_patch_tokens.dtype)
            )

        epsilon = 1e-6
        pooled_features = feature_sum / (token_count + epsilon)
        pooled_features = pooled_features.permute(0, 2, 1)

        cls_token_final = x[:, :1, :].permute(0, 2, 1)
        x = torch.cat((cls_token_final, pooled_features), dim=2)

        return x, x_heatmap

    def init_weights(self):
        self._backbone.init_weights()

    def _freeze_layers(self):
        self._backbone._freeze_layers()


@MODELS.register_module()
class VisionTransformerSparseAdapterPOGUISE(BaseModule):
    """
    Vision Transformer with integrated Token Pruning and Merging.
    Uses Sparse Adapter with persistent lookup table.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dims: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: int = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_cfg: ConfigType = dict(type="LN", eps=1e-6),
        num_frames: int = 16,
        tubelet_size: int = 2,
        pretrained: Optional[str] = None,
        with_cp: bool = False,
        keep_rate: float = 0.6,
        adapter_mlp_ratio: float = 0.25,
        total_frames: int = 768,
        adapter_index: list = list(range(12)),
        adapter_conv_types=["2d_conv"] * 4 + ["sparse_conv"] * 8,
        deform_conv_groups: int = 1,
        adapter_use_attn: Union[bool, int] = False,
        init_cfg: Optional[Union[Dict, List[Dict]]] = [
            dict(type="TruncNormal", layer="Linear", std=0.02, bias=0.0),
            dict(type="Constant", layer="LayerNorm", val=1.0, bias=0.0),
        ],
        return_feat_map: bool = False,
        return_feat_map_hw: bool = False,
        n_landmarks: int = 0,  # Default to 0 to disable heatmap by default
        hw_out_conv: tuple = (10, 10),
        **kwargs,
    ) -> None:
        if pretrained:
            self.init_cfg = dict(type="Pretrained", checkpoint=pretrained)
        super().__init__(init_cfg=init_cfg)

        self.with_cp = with_cp
        self.embed_dims = embed_dims
        self.patch_size = patch_size
        self.return_feat_map = return_feat_map
        self.return_feat_map_hw = return_feat_map_hw
        self.n_landmarks = n_landmarks
        self.hw_out_conv = hw_out_conv
        self.n_heatmap_tokens = 0

        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            embed_dims=embed_dims,
            conv_type="Conv3d",
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
            padding=(0, 0, 0),
            dilation=(1, 1, 1),
        )

        grid_size = img_size // patch_size
        num_patches = grid_size**2 * (num_frames // tubelet_size)
        self.grid_size = (grid_size, grid_size)
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        pos_embed = get_sinusoid_encoding(num_patches, embed_dims)
        self.register_buffer("pos_embed", pos_embed)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        if depth == 12:
            keep_rate = [1, 1, 1, keep_rate, 1, 1, keep_rate, 1, 1, keep_rate, 1, 1]
        elif depth == 24:
            keep_rate = [
                1,
                1,
                1,
                keep_rate,
            ] * 4 + [1, 1, 1, 1] * 2
            print(f"Using keep_rate schedule: {keep_rate}")

        elif depth == 32:
            keep_rate = [
                1,
                1,
                1,
                keep_rate,
            ] * 8
        elif depth == 40:
            keep_rate = [
                1,
                1,
                1,
                keep_rate,
            ] * 10

        # Logic to handle the new adapter_conv_types parameter
        if adapter_conv_types is None:
            # Default to sparse_conv if not specified
            print(
                "adapter_conv_types not provided — default: 'sparse_conv' for all adapters."
            )
            self.adapter_conv_types = ["sparse_conv"] * depth
        elif isinstance(adapter_conv_types, str):
            self.adapter_conv_types = [adapter_conv_types] * depth
        else:
            self.adapter_conv_types = adapter_conv_types
        assert (
            len(self.adapter_conv_types) == depth
        ), "adapter_conv_types list must have the same length as depth"

        num_special_tokens = 1  # Start with 1 for the [CLS] token
        if self.n_landmarks > 0:
            self.n_heatmap_tokens = self.hw_out_conv[0] * self.hw_out_conv[1]
            self.heatmap_tokens = nn.Parameter(
                torch.zeros(1, self.n_heatmap_tokens, embed_dims)
            )
            nn.init.trunc_normal_(self.heatmap_tokens, std=0.02)
            self.heatmap_head = HeatmapHead(
                in_channels=embed_dims,
                in_size=self.hw_out_conv[0],
                out_channels=self.n_landmarks,
                deconv_out_channels=(256, 256),
                deconv_kernel_sizes=(4, 4),
            )
            num_special_tokens += self.n_heatmap_tokens
        self.use_adapter = any(i in adapter_index for i in range(depth))
        self.blocks = ModuleList(
            [
                Block(
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=dpr[i],
                    norm_cfg=norm_cfg,
                    with_cp=with_cp,
                    keep_rate=keep_rate[i],
                    use_adapter=i in adapter_index,
                    init_cfg=init_cfg,
                    adapter_mlp_ratio=adapter_mlp_ratio,
                    adapter_conv_type=self.adapter_conv_types[i],
                    deform_conv_groups=deform_conv_groups,
                    adapter_use_attn=adapter_use_attn,
                    n_key_tokens=num_special_tokens,
                    heatmap_grid_size=self.hw_out_conv,
                    **kwargs,
                )
                for i in range(depth)
            ]
        )

        self.norm = build_norm_layer(norm_cfg, embed_dims)[1]
        self._measure_adapter_time = False  # Flag to enable adapter timing
        self._measure_block_time = False  # Flag to enable per-block timing

        # Check if any block needs neighbor_indices (only sparse_conv does)
        self.needs_neighbor_indices = any(
            blk.needs_neighbor_indices for blk in self.blocks
        )

    def set_measure_adapter_time(self, enabled: bool) -> None:
        """Enable or disable adapter timing measurement."""
        self._measure_adapter_time = enabled

    def get_adapter_time_ms(self) -> float:
        """Get total accumulated adapter time in milliseconds."""
        return sum(blk._adapter_time_ms for blk in self.blocks if blk.use_adapter)

    def reset_adapter_time(self) -> None:
        """Reset adapter timing accumulators."""
        for blk in self.blocks:
            blk._adapter_time_ms = 0.0

    def set_measure_block_time(self, enabled: bool) -> None:
        """Enable or disable per-block timing measurement."""
        self._measure_block_time = enabled

    def get_block_times_ms(self) -> list:
        """Get per-block accumulated times in milliseconds."""
        return [blk._block_time_ms for blk in self.blocks]

    def reset_block_time(self) -> None:
        """Reset per-block timing accumulators."""
        for blk in self.blocks:
            blk._block_time_ms = 0.0

    def get_initial_neighbor_indices(self, h, w, T, device, B):
        # h, w are patch grid dimensions
        N_spatial = h * w
        N_total = N_spatial * T

        # Generate spatial neighbors for one frame (0..N_spatial-1)
        indices = torch.arange(N_spatial, device=device, dtype=torch.int32).view(h, w)
        padded = torch.full((h + 2, w + 2), -1, device=device, dtype=torch.int32)
        padded[1:-1, 1:-1] = indices

        neighbors = []
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                slice_y = slice(1 + dy, 1 + dy + h)
                slice_x = slice(1 + dx, 1 + dx + w)
                neighbors.append(padded[slice_y, slice_x].flatten())

        spatial_neighbor_indices = torch.stack(neighbors, dim=1)  # (N_spatial, 9)

        # Replicate for T frames
        # For frame t, indices are shifted by t * N_spatial

        # (T, N_spatial, 9)
        neighbor_indices = (
            spatial_neighbor_indices.unsqueeze(0).expand(T, -1, -1).clone()
        )

        # Add temporal offsets
        t_offsets = (
            torch.arange(T, device=device, dtype=torch.int32) * N_spatial
        ).view(T, 1, 1)
        mask = neighbor_indices != -1
        neighbor_indices[mask] += t_offsets.expand_as(neighbor_indices)[mask]

        # Flatten T and N_spatial -> (N_total, 9)
        neighbor_indices = neighbor_indices.reshape(N_total, 9)

        # Replicate for Batch
        neighbor_indices = neighbor_indices.unsqueeze(0).expand(B, -1, -1).clone()

        # Add batch offsets
        b_offsets = (torch.arange(B, device=device, dtype=torch.int32) * N_total).view(
            B, 1, 1
        )
        mask = neighbor_indices != -1
        neighbor_indices[mask] += b_offsets.expand_as(neighbor_indices)[mask]

        return neighbor_indices

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_adapter:
            self._freeze_layers()
        b, _, _, h, w = x.shape
        h_patch = h // self.patch_size
        w_patch = w // self.patch_size
        x, _ = self.patch_embed(x)

        current_total_patches = x.shape[1]

        if (h_patch, w_patch) != self.grid_size:
            pos_embed = self.pos_embed.reshape(-1, *self.grid_size, self.embed_dims)
            pos_embed = pos_embed.permute(0, 3, 1, 2)
            pos_embed = F.interpolate(
                pos_embed, size=(h_patch, w_patch), mode="bicubic", align_corners=False
            )
            pos_embed = pos_embed.permute(0, 2, 3, 1).flatten(1, 2)
            pos_embed = pos_embed.reshape(1, -1, self.embed_dims)
        else:
            pos_embed = self.pos_embed

        x = x + pos_embed
        x = self.pos_drop(x)
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        if self.n_landmarks > 0:
            heatmap_tokens = self.heatmap_tokens.expand(b, -1, -1)
            # Insert heatmap tokens after the [CLS] token
            x = torch.cat((x[:, :1, :], heatmap_tokens, x[:, 1:, :]), dim=1)
        idx = (
            torch.arange(0, current_total_patches, device=x.device)
            .unsqueeze(0)
            .repeat(b, 1)
        )

        # Initialize neighbor_indices only if needed (sparse_conv uses it)
        if self.needs_neighbor_indices:
            T_patches = current_total_patches // (h_patch * w_patch)
            neighbor_indices = self.get_initial_neighbor_indices(
                h_patch, w_patch, T_patches, x.device, b
            )
            # Pre-allocate global_mapping buffer for reuse across all pruning layers
            # Size is B * max_tokens (initial token count)
            global_mapping_buffer = torch.empty(
                b * current_total_patches, dtype=torch.int32, device=x.device
            )
        else:
            neighbor_indices = None
            global_mapping_buffer = None
        # print(neighbor_indices[0, :10, :10])  # Debug print to check neighbor indices initialization
        for blk in self.blocks:
            x, idx, neighbor_indices = blk(
                x,
                idx,
                h_patch,
                w_patch,
                current_total_patches,
                neighbor_indices,
                measure_adapter_time=self._measure_adapter_time,
                measure_block_time=self._measure_block_time,
                global_mapping_buffer=global_mapping_buffer,
            )
            # print(f"After block: x shape={x.shape}, idx shape={idx.shape}")

        x_heatmap = None
        if self.n_landmarks > 0:
            # Slice out the processed heatmap tokens (they are right after the [CLS] token)
            start_idx = 1
            end_idx = 1 + self.n_heatmap_tokens
            heatmap_tokens_out = x[:, start_idx:end_idx, :]

            # Reshape from a sequence to a 2D feature map
            heatmap_feats = heatmap_tokens_out.reshape(
                b, *self.hw_out_conv, self.embed_dims
            )
            heatmap_feats = heatmap_feats.permute(
                0, 3, 1, 2
            ).contiguous()  # (B, C, H, W)

            # Get the final heatmap prediction
            x_heatmap = self.heatmap_head(heatmap_feats)
        x = self.norm(x)

        num_special_tokens = 1 + self.n_heatmap_tokens

        final_patch_tokens = x[:, num_special_tokens:, :]  # Shape: (B, N_kept, C)
        final_patch_indices = idx  # Shape: (B, N_kept)

        T_out = self.num_frames // self.tubelet_size

        # Optimized pooling: Scatter add directly to (B, T, C)
        # Calculate frame index for each token
        patches_per_frame = h_patch * w_patch
        token_frame_indices = final_patch_indices // patches_per_frame  # [B, N_kept]

        # Initialize output tensors
        feature_sum = torch.zeros(
            b,
            T_out,
            self.embed_dims,
            device=final_patch_tokens.device,
            dtype=final_patch_tokens.dtype,
        )
        token_count = torch.zeros(
            b,
            T_out,
            1,
            device=final_patch_tokens.device,
            dtype=final_patch_tokens.dtype,
        )

        if final_patch_tokens.shape[1] > 0:
            # Expand indices for feature scatter
            idx_expanded = token_frame_indices.unsqueeze(-1).expand(
                -1, -1, self.embed_dims
            )
            feature_sum.scatter_add_(1, idx_expanded, final_patch_tokens)

            # Expand indices for count scatter
            idx_mask = token_frame_indices.unsqueeze(-1)
            token_count.scatter_add_(
                1, idx_mask, torch.ones_like(idx_mask, dtype=final_patch_tokens.dtype)
            )

        if self.return_feat_map_hw:
            # Reconstruct (B, C, T, H, W)
            # Initialize (B, T*H*W, C)
            total_grid_size = T_out * patches_per_frame
            full_feat = torch.zeros(
                b,
                total_grid_size,
                self.embed_dims,
                device=final_patch_tokens.device,
                dtype=final_patch_tokens.dtype,
            )

            # Scatter tokens back to their grid positions
            if final_patch_tokens.shape[1] > 0:
                # final_patch_indices is (B, N_kept)
                # We need to expand to (B, N_kept, C)
                # Note: indices are in range [0, total_grid_size-1]
                idx_expanded = final_patch_indices.unsqueeze(-1).expand(
                    -1, -1, self.embed_dims
                )
                full_feat.scatter_(1, idx_expanded, final_patch_tokens)

            # Reshape to (B, T, H, W, C) -> permute to (B, C, T, H, W)
            full_feat = full_feat.reshape(b, T_out, h_patch, w_patch, self.embed_dims)
            full_feat = full_feat.permute(0, 4, 1, 2, 3)  # B, C, T, H, W

            # Since this output is different from standard (B, C, T+1), we might skip CLS token concat or return it separately.
            # For compatibility with wrappers, we usually return features.
            # We can return a tuple or just the map.
            # Standard opentad wrappers expect features.
            return full_feat, x_heatmap

        epsilon = 1e-6
        pooled_features = feature_sum / (token_count + epsilon)

        # Permute to (B, C, T_out)
        pooled_features = pooled_features.permute(0, 2, 1)
        # concat with cls token
        cls_token_final = x[:, :1, :]  # Shape: (B, 1, C)
        cls_token_final = cls_token_final.permute(0, 2, 1)
        x = torch.cat(
            (cls_token_final, pooled_features), dim=2
        )  # Shape: (B, C, T_out+1)

        return x, x_heatmap

    def _freeze_layers(self):
        """Prevent all the parameters not in the adapters"""
        self.patch_embed.eval()
        for m in self.patch_embed.modules():
            for param in m.parameters():
                param.requires_grad = False

        for block in self.blocks:
            for m, n in block.named_children():
                if (
                    "adapter" not in m
                    and m != "drop_path"
                    and "cls_token" not in m
                    and m != "heatmap_head"
                    and m != "heatmap_tokens"
                ):
                    n.eval()
                    for param in n.parameters():
                        param.requires_grad = False


@MODELS.register_module()
class VisionTransformerSparseAdapterNorm(VisionTransformerSparseAdapterPOGUISE):
    """
    ``VisionTransformerSparseAdapterPOGUISE`` with L2-norm token selection.

    Tokens are scored by the L2 norm of their layer-normalised features
    (Patch Slimming, Li et al., CVPR 2022) instead of CLS-attention scores.
    Everything else — SparseConv2D adapter, keep_rate, cross-attention — is
    unchanged, making this a clean ablation of the selector strategy.
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("selector", "norm")
        super().__init__(**kwargs)
