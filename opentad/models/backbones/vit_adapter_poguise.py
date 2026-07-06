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
from torchvision.ops import DeformConv2d
from mmengine.model import BaseModule, ModuleList
from mmengine.model.weight_init import constant_init, kaiming_init, trunc_normal_init
from mmengine.registry import MODELS

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
    An efficient Adapter module with true index-based scatter-and-gather to
    handle arbitrary token pruning from anywhere in a video.
    conv_type: The type of convolution to use (e.g., '3d_conv', '2d_dwconv', etc.)
    """

    def __init__(
        self,
        embed_dims: int,
        mlp_ratio: float = 0.25,
        conv_type: str = "3d_conv",
        kernel_size: int = 3,
        temporal_kernel_size: int = 3,
        dilation: int = 1,
        deformable_groups: int = 1,
        **kwargs,
    ) -> None:
        super().__init__()
        self.conv_type = conv_type

        hidden_dims = int(embed_dims * mlp_ratio)
        self.hidden_dims = hidden_dims

        # Simplified non-gated projection
        self.down_proj = nn.Linear(embed_dims, hidden_dims)
        self.act = nn.GELU()
        self.up_proj = nn.Linear(hidden_dims, embed_dims)
        self.gamma = nn.Parameter(torch.ones(1))

        if self.conv_type == "deformable_conv":
            print(f"Using Deformable Convolution with {deformable_groups} groups")
            assert (
                hidden_dims % deformable_groups == 0
            ), f"hidden_dims ({hidden_dims}) must be divisible by deformable_groups ({deformable_groups})."
            self.offset_conv = nn.Conv2d(
                hidden_dims,
                deformable_groups * 2 * kernel_size * kernel_size,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )
            self.mask_conv = nn.Conv2d(
                hidden_dims,
                deformable_groups * 1 * kernel_size * kernel_size,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )
            self.deform_conv = DeformConv2d(
                hidden_dims,
                hidden_dims,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=False,
                groups=deformable_groups,
            )
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
            self.mask_conv = nn.Conv2d(
                hidden_dims,
                deformable_groups * 1 * kernel_size * kernel_size,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )
            self.deform_conv = DeformConv2d(
                hidden_dims,
                hidden_dims,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=False,
                groups=deformable_groups,
            )

            # 2. Temporal 1D Convolution (applied after the spatial one)
            self.temporal_conv = nn.Conv1d(
                hidden_dims,
                hidden_dims,
                kernel_size=temporal_kernel_size,
                padding=(temporal_kernel_size // 2) * dilation,
                dilation=dilation,
                groups=hidden_dims,  # Using a depthwise convolution for efficiency
            )
            self.temporal_pw_conv = nn.Conv1d(hidden_dims, hidden_dims, 1)  # Pointwise
        elif self.conv_type == "3d_conv":
            print("Using 3D Convolution")
            self.spatial_temporal_conv3d = nn.Conv3d(
                hidden_dims,
                hidden_dims,
                kernel_size=(temporal_kernel_size, kernel_size, kernel_size),
                padding=(temporal_kernel_size // 2, kernel_size // 2, kernel_size // 2),
            )
        # INTEGRATION: Added 'temporal_dwconv' option from Adatad
        elif self.conv_type == "temporal_dwconv":
            print("Using 1D Temporal Depthwise Separable Convolution (Adatad-like)")
            self.temporal_dwconv = nn.Conv1d(
                hidden_dims,
                hidden_dims,
                kernel_size=temporal_kernel_size,
                stride=1,
                padding=(temporal_kernel_size // 2) * dilation,
                dilation=dilation,
                groups=hidden_dims,
            )
            self.temporal_pwconv = nn.Conv1d(hidden_dims, hidden_dims, 1)
        elif self.conv_type == "2d_dwconv":
            print("Using 2D Depthwise Separable Convolution")
            self.spatial_dwconv = nn.Conv2d(
                hidden_dims,
                hidden_dims,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=hidden_dims,
            )
            self.spatial_pwconv = nn.Conv2d(hidden_dims, hidden_dims, 1)
        elif self.conv_type == "sparse_conv":
            print("Using Sparse Convolution")
            from ..bricks import SparseConv2d

            self.sparse_conv = SparseConv2d(hidden_dims, hidden_dims)
        else:
            raise ValueError(
                f"Unknown conv_type: '{self.conv_type}' in EfficientAdapter"
            )

        # Initialization
        self._init_weights()

    def _init_weights(self):
        trunc_normal_init(self.down_proj, std=0.02, bias=0)
        constant_init(self.up_proj, 0)
        if self.conv_type == "deformable_conv":
            constant_init(self.offset_conv, 0)
            constant_init(self.mask_conv, 0)
            kaiming_init(self.deform_conv, mode="fan_in", nonlinearity="relu")
        elif self.conv_type == "deformable_conv_t":
            constant_init(self.offset_conv, 0)
            constant_init(self.mask_conv, 0)
            kaiming_init(self.deform_conv, mode="fan_in", nonlinearity="relu")
            # Initialize the new temporal convolution layers
            self.temporal_conv.weight.data.normal_(
                mean=0.0, std=math.sqrt(2.0 / self.temporal_conv.kernel_size[0])
            )
            self.temporal_conv.bias.data.zero_()
            self.temporal_pw_conv.weight.data.normal_(
                mean=0.0, std=math.sqrt(2.0 / self.hidden_dims)
            )
            self.temporal_pw_conv.bias.data.zero_()
        elif self.conv_type == "3d_conv":
            kaiming_init(
                self.spatial_temporal_conv3d, mode="fan_in", nonlinearity="relu"
            )
        # INTEGRATION: Added weight initialization for 'temporal_dwconv'
        elif self.conv_type == "temporal_dwconv":
            self.temporal_dwconv.weight.data.normal_(
                mean=0.0, std=math.sqrt(2.0 / self.temporal_dwconv.kernel_size[0])
            )
            self.temporal_dwconv.bias.data.zero_()
            self.temporal_pwconv.weight.data.normal_(
                mean=0.0, std=math.sqrt(2.0 / self.hidden_dims)
            )
            self.temporal_pwconv.bias.data.zero_()
        elif self.conv_type == "2d_dwconv":
            kaiming_init(self.spatial_dwconv, mode="fan_in", nonlinearity="relu")
            kaiming_init(self.spatial_pwconv, mode="fan_in", nonlinearity="relu")
        elif self.conv_type == "sparse_conv":
            pass

    def forward(
        self, x: Tensor, h: int, w: int, indices: Tensor, total_num_patches: int
    ) -> Tensor:
        inputs = x
        B, N, C = x.shape
        x_proj = self.down_proj(x)
        x_proj = self.act(x_proj)
        num_special_tokens = N - indices.shape[1]
        special_tokens_proj = x_proj[:, :num_special_tokens, :]
        patch_tokens_proj = x_proj[:, num_special_tokens:, :]

        if patch_tokens_proj.shape[1] == 0:
            processed_tokens = self.up_proj(self.act(x_proj))
            return processed_tokens * self.gamma + inputs

        if self.conv_type == "sparse_conv":
            # patch_tokens_proj: [B, N, C]
            # indices: [B, N]
            B, N_kept, C_hidden = patch_tokens_proj.shape

            # Flatten batch and tokens
            x_flat = patch_tokens_proj.reshape(-1, C_hidden)  # [B*N, C]

            # Create lookup table for neighbor search
            # indices contains values in [0, total_num_patches)
            # We need to map (b, grid_idx) -> flat_idx in x_flat

            # Initialize lookup with -1
            lookup = torch.full(
                (B, total_num_patches), -1, dtype=torch.long, device=x.device
            )

            # Values to put in lookup: 0..(B*N-1)
            flat_indices = torch.arange(B * N_kept, device=x.device).view(B, N_kept)
            lookup.scatter_(1, indices, flat_indices)

            # Now compute neighbors for each token in indices
            frame_size = h * w
            frame_idx = indices // frame_size
            in_frame_idx = indices % frame_size
            y = in_frame_idx // w
            x_coord = in_frame_idx % w

            neighbor_indices_list = []
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    ny = y + dy
                    nx = x_coord + dx
                    valid = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)

                    ny_clamped = ny.clamp(0, h - 1)
                    nx_clamped = nx.clamp(0, w - 1)

                    n_idx = frame_idx * frame_size + ny_clamped * w + nx_clamped

                    n_flat = torch.gather(lookup, 1, n_idx)

                    # Apply mask
                    n_flat = n_flat.masked_fill(~valid, -1)
                    neighbor_indices_list.append(n_flat)

            neighbor_indices = torch.stack(neighbor_indices_list, dim=2).view(-1, 9)

            conv_output = self.sparse_conv(x_flat, neighbor_indices)
            processed_patches = conv_output.view(B, N_kept, self.hidden_dims)

        else:
            full_grid = torch.zeros(
                B,
                total_num_patches,
                self.hidden_dims,
                device=patch_tokens_proj.device,
                dtype=patch_tokens_proj.dtype,
            )
            indices_expanded = indices.unsqueeze(-1).expand(-1, -1, self.hidden_dims)
            scattered_tokens = full_grid.scatter_(
                1, indices_expanded, patch_tokens_proj
            )

            num_temporal_steps_full = total_num_patches // (h * w)

            if self.conv_type == "3d_conv":
                conv_input = scattered_tokens.reshape(
                    B, num_temporal_steps_full, h, w, self.hidden_dims
                )
                conv_input = conv_input.permute(0, 4, 1, 2, 3).contiguous()
                conv_output = self.spatial_temporal_conv3d(conv_input)
                processed_full_sequence = conv_output.permute(0, 2, 3, 4, 1).reshape(
                    B, -1, self.hidden_dims
                )
            # INTEGRATION: Added forward logic for 'temporal_dwconv'
            elif self.conv_type == "temporal_dwconv":
                # Reshape to (B, T, H, W, C)
                conv_input = scattered_tokens.reshape(
                    B, num_temporal_steps_full, h, w, self.hidden_dims
                )
                # Permute and flatten to (B*H*W, C, T) for Conv1d
                conv_input = conv_input.permute(0, 2, 3, 4, 1).flatten(0, 2)
                conv_output = self.temporal_pwconv(self.temporal_dwconv(conv_input))
                # Reverse the process to get back to (B, T*H*W, C)
                processed_full_sequence = (
                    conv_output.unflatten(0, (B, h, w))
                    .permute(0, 4, 1, 2, 3)
                    .reshape(B, -1, self.hidden_dims)
                )
            elif self.conv_type == "deformable_conv_t":
                # 1. Perform Spatial Deformable Convolution on each frame
                # Reshape to (B*T, C, H, W)
                conv_input_2d = scattered_tokens.reshape(
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
                # Permute from (B, H, W, C, T) to (B, T, H, W, C) and finally flatten to (B, T*H*W, C)
                processed_full_sequence = processed_full_sequence.permute(
                    0, 4, 1, 2, 3
                ).reshape(B, -1, self.hidden_dims)

            else:  # Covers 'deformable_conv' and '2d_dwconv'
                conv_input = scattered_tokens.reshape(
                    B * num_temporal_steps_full, h, w, self.hidden_dims
                )
                conv_input = conv_input.permute(0, 3, 1, 2).contiguous()
                if self.conv_type == "deformable_conv":
                    offset = self.offset_conv(conv_input)
                    modulation_mask = torch.sigmoid(self.mask_conv(conv_input))
                    conv_output = self.deform_conv(
                        conv_input, offset, mask=modulation_mask
                    )
                else:  # '2d_dwconv'
                    conv_output = self.spatial_pwconv(self.spatial_dwconv(conv_input))
                processed_full_sequence = conv_output.permute(0, 2, 3, 1).reshape(
                    B, -1, self.hidden_dims
                )

            processed_patches = torch.gather(
                processed_full_sequence, 1, indices_expanded
            )

        x = torch.cat([special_tokens_proj, processed_patches], dim=1)
        x = self.up_proj(x)
        return x * self.gamma + inputs


def sim_matrixv2_batch(a, b, eps=1e-8):
    a_n, b_n = a.norm(dim=-1)[:, :, None], b.norm(dim=-1)[:, :, None]
    a_norm = a / torch.clamp(a_n, min=eps)
    b_norm = b / torch.clamp(b_n, min=eps)
    sim_mt = torch.bmm(a_norm, b_norm.transpose(-2, -1))
    return sim_mt


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
        sim_metric: int = 0,
        topk_type: int = 0,
        n_key_tokens: int = 1,
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
        self.enhanced_weight_class = enhanced_weight_class
        self.topk_type = topk_type
        self.sim_metric = sim_metric
        self.n_key_tokens = n_key_tokens

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
        else:
            attn = (q * self.scale) @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)

            attn_for_output = self.attn_drop(attn)
            x = (attn_for_output @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        feature_for_pruning = k
        return x, attn, feature_for_pruning

    def forward(self, x: torch.Tensor, last_idx: Optional[torch.Tensor] = None):
        B, N, C = x.shape
        x, attn, feature = self.forward_part1(x)

        if self.keep_rate >= 1:
            return x, last_idx, feature

        num_s_tokens = self.n_key_tokens
        num_keep_tokens = math.ceil(self.keep_rate * (N - num_s_tokens))

        if num_keep_tokens <= 0:
            return x, torch.empty(B, 0, dtype=last_idx.dtype, device=x.device), feature

        if self.topk_type == 0:
            attn_topk = attn.sum(dim=-2).mean(dim=1)
        elif self.topk_type == 1:
            attn_topk = attn[:, :, : self.n_key_tokens].sum(dim=-2).mean(dim=1)
        else:
            attn_topk = attn[:, :, 0].mean(dim=1)

        # Up-weight the CLS column contribution without cloning the full attn tensor
        if self.enhanced_weight_class != 1:
            cls_boost = attn[:, :, 0].mean(dim=1) * (self.enhanced_weight_class - 1)
            attn_topk = attn_topk + cls_boost

        attn_topk = attn_topk[:, num_s_tokens:]
        num_keep_tokens = min(num_keep_tokens, attn_topk.shape[1])
        _, idx_evad = torch.topk(attn_topk, num_keep_tokens, dim=1, largest=True)

        idx = idx_evad.sort(dim=1)[0]
        return x, idx, feature


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
        keep_rate_merge: float = 1.0,
        merge_type: str = "sim",
        merge_mode: int = 0,
        n_key_tokens: int = 1,
        use_adapter: bool = False,
        adapter_mlp_ratio: float = 0.25,
        adapter_conv_type: str = "3d_conv",
        deform_conv_groups: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.with_cp = with_cp
        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.n_key_tokens = n_key_tokens
        self.keep_rate = keep_rate
        self.keep_rate_merge = keep_rate_merge
        self.merge_type = merge_type
        self.merge_mode = merge_mode
        self.num_heads = num_heads
        self.use_adapter = use_adapter

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
            # MODIFIED: Pass the string directly to EfficientAdapter
            self.adapter = EfficientAdapter(
                embed_dims=embed_dims,
                mlp_ratio=adapter_mlp_ratio,
                conv_type=adapter_conv_type,
                deformable_groups=deform_conv_groups,
            )

    def forward(
        self,
        x: torch.Tensor,
        last_idx: torch.Tensor,
        h: int,
        w: int,
        total_num_patches: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        def _inner_forward(x, last_idx):
            B, N, C = x.shape
            x_norm = self.norm1(x)
            attn_x, idx, feature_for_merge = self.attn(x_norm, last_idx)
            x = x + self.drop_path(attn_x)

            num_s_tokens = self.n_key_tokens

            if self.keep_rate < 1:
                x_key = x[:, :num_s_tokens]
                x_nonkey = x[:, num_s_tokens:]

                idx_expanded = idx.unsqueeze(-1).expand(-1, -1, C)
                x_nonkey_keep = torch.gather(x_nonkey, dim=1, index=idx_expanded)

                num_pruned_tokens = x_nonkey.shape[1]
                num_kept_tokens = idx.shape[1]

                updated_global_idx = torch.gather(last_idx, dim=1, index=idx)

                if self.keep_rate_merge < 1 and num_kept_tokens < num_pruned_tokens:
                    mask = torch.ones(
                        B, num_pruned_tokens, dtype=torch.bool, device=x.device
                    )
                    mask.scatter_(1, idx, False)
                    x_nonselected = x_nonkey[mask].reshape(B, -1, C)
                    feature_for_merge_nonkey = (
                        feature_for_merge[:, :, num_s_tokens:, :]
                        .transpose(1, 2)
                        .reshape(B, num_pruned_tokens, -1)
                    )
                    feature_nonselected = feature_for_merge_nonkey[mask].reshape(
                        B, -1, C
                    )
                    idx_nonselected_global = last_idx[mask].reshape(B, -1)
                    num_to_merge = x_nonselected.shape[1]
                    if num_to_merge > 1:
                        num_merged_tokens = math.ceil(
                            self.keep_rate_merge * num_to_merge
                        )
                        merge_fn, src_idx = self.bipartite_soft_matching(
                            feature_nonselected, num_merged_tokens
                        )
                        x_merged = self.merge_wavg(merge_fn, x_nonselected)
                        src_idx_global = src_idx.squeeze(-1)
                        idx_merged_global = torch.gather(
                            idx_nonselected_global, 1, src_idx_global
                        )
                        x_nonkey_keep = torch.cat([x_nonkey_keep, x_merged], dim=1)
                        updated_global_idx = torch.cat(
                            [updated_global_idx, idx_merged_global], dim=1
                        )
                x = torch.cat([x_key, x_nonkey_keep], dim=1)
                idx = updated_global_idx
            else:
                idx = last_idx

            x_post_mlp = self.mlp(self.norm2(x))
            x = x + self.drop_path(x_post_mlp)
            if self.use_adapter:
                x = self.adapter(x, h, w, idx, total_num_patches)
            return x, idx

        if self.with_cp and x.requires_grad:
            x, idx = cp.checkpoint(_inner_forward, x, last_idx, use_reentrant=False)
        else:
            x, idx = _inner_forward(x, last_idx)
        return x, idx

    def bipartite_soft_matching(self, metric, r):
        with torch.no_grad():
            scores = sim_matrixv2_batch(metric, metric)

            if scores.shape[-1] > 0:
                diag_mask = torch.eye(
                    scores.shape[-1], device=scores.device, dtype=torch.bool
                ).unsqueeze(0)
                scores.masked_fill_(diag_mask, -float("inf"))

            node_max, node_idx = scores.max(dim=-1)
            edge_idx = node_max.argsort(dim=-1, descending=True)

            src_idx = edge_idx[..., :r]
            dst_idx = torch.gather(node_idx, -1, src_idx)

            src_idx = src_idx.unsqueeze(-1)
            dst_idx = dst_idx.unsqueeze(-1)

        def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
            n, t, c = x.shape
            src = x.gather(dim=1, index=src_idx.expand(n, r, c))
            dst = x.gather(dim=1, index=dst_idx.expand(n, r, c))

            if mode == "mean":
                merged = (dst + src) / 2
            else:
                merged = dst + src
            return merged

        return merge, src_idx

    def merge_wavg(self, merge, x: torch.Tensor):
        mode = {0: "mean", 1: "sum"}
        x = merge(x, mode=mode[self.merge_mode])
        return x


@MODELS.register_module()
class VisionTransformerAdapterPOGUISE(BaseModule):
    """
    Vision Transformer with integrated Token Pruning and Merging.
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
        keep_rate_merge: float = 0.3,
        adapter_mlp_ratio: float = 0.25,
        total_frames: int = 768,
        adapter_index: list = [3, 5, 7, 11],
        # MODIFIED: Replaced boolean lists with a single list of strings
        adapter_conv_types=None,
        deform_conv_groups: int = 1,
        init_cfg: Optional[Union[Dict, List[Dict]]] = [
            dict(type="TruncNormal", layer="Linear", std=0.02, bias=0.0),
            dict(type="Constant", layer="LayerNorm", val=1.0, bias=0.0),
        ],
        return_feat_map: bool = False,
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
        trunc_normal_init(self.cls_token, std=0.02, bias=0.0)

        pos_embed = get_sinusoid_encoding(num_patches, embed_dims)
        self.register_buffer("pos_embed", pos_embed)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        if depth == 12:
            keep_rate = [1, 1, 1, keep_rate, 1, 1, keep_rate, 1, 1, keep_rate, 1, 1]
            keep_rate_merge = [
                1,
                1,
                1,
                keep_rate_merge,
                1,
                1,
                keep_rate_merge,
                1,
                1,
                keep_rate_merge,
                1,
                1,
            ]
        elif depth == 24:
            keep_rate = [
                1,
                1,
                1,
                keep_rate,
            ] * 6
            keep_rate_merge = [
                1,
                1,
                1,
                keep_rate_merge,
            ] * 6

        # MODIFIED: Logic to handle the new adapter_conv_types parameter
        if adapter_conv_types is None:
            # Recreate original default behavior if no list is provided
            print(
                "adapter_conv_types not provided — default: '3d_conv' for blocks 0-6 and 'deformable_conv' for 7-11."
            )
            self.adapter_conv_types = ["3d_conv"] * 7 + ["deformable_conv"] * 5
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
                torch.randn(1, self.n_heatmap_tokens, embed_dims)
            )
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
                    keep_rate_merge=keep_rate_merge[i],
                    use_adapter=i in adapter_index,
                    init_cfg=init_cfg,
                    adapter_mlp_ratio=adapter_mlp_ratio,
                    adapter_conv_type=self.adapter_conv_types[i],
                    deform_conv_groups=deform_conv_groups,
                    n_key_tokens=num_special_tokens,
                )
                for i in range(depth)
            ]
        )

        self.norm = build_norm_layer(norm_cfg, embed_dims)[1]

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

        for blk in self.blocks:
            x, idx = blk(x, idx, h_patch, w_patch, current_total_patches)
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
            heatmap_feats = heatmap_feats.permute(0, 3, 1, 2)  # (B, C, H, W)

            # Get the final heatmap prediction
            x_heatmap = self.heatmap_head(heatmap_feats)
        x = self.norm(x)

        num_special_tokens = 1 + self.n_heatmap_tokens

        final_patch_tokens = x[:, num_special_tokens:, :]  # Shape: (B, N_kept, C)
        final_patch_indices = idx  # Shape: (B, N_kept)

        T_out = self.num_frames // self.tubelet_size
        total_patches_in_volume = T_out * h_patch * w_patch

        # Create grids for both features and a mask to count non-zero elements
        scattered_features = torch.zeros(
            b,
            total_patches_in_volume,
            self.embed_dims,
            device=final_patch_tokens.device,
            dtype=final_patch_tokens.dtype,
        )
        # The mask will store 1 for every kept token and 0 for every pruned one.
        # We only need to track one value per token, so the last dim is 1.
        scattered_mask = torch.zeros(
            b,
            total_patches_in_volume,
            1,
            device=final_patch_tokens.device,
            dtype=final_patch_tokens.dtype,
        )

        if final_patch_tokens.shape[1] > 0:  # Proceed only if there are tokens left
            # Expand indices for both feature and mask scattering
            final_patch_indices_expanded = final_patch_indices.unsqueeze(-1).expand(
                -1, -1, self.embed_dims
            )
            final_patch_indices_mask = final_patch_indices.unsqueeze(
                -1
            )  # For mask, last dim is 1

            # Scatter the features and a tensor of '1's into the mask
            scattered_features.scatter_(
                1, final_patch_indices_expanded, final_patch_tokens
            )
            scattered_mask.scatter_(
                1, final_patch_indices_mask, torch.ones_like(scattered_mask)
            )

        # Reshape features and mask into 5D tensors (B, T, H, W, C)
        reshaped_features = scattered_features.reshape(
            b, T_out, h_patch, w_patch, self.embed_dims
        )
        reshaped_mask = scattered_mask.reshape(b, T_out, h_patch, w_patch, 1)

        # Sum features and the mask over the spatial dimensions (H and W)
        feature_sum_per_frame = reshaped_features.sum(
            dim=(2, 3)
        )  # Shape: (B, T_out, C)
        kept_token_count_per_frame = reshaped_mask.sum(
            dim=(2, 3)
        )  # Shape: (B, T_out, 1)

        # Add a small epsilon to the denominator to prevent division by zero
        # in the rare case that an entire frame's tokens are pruned.
        epsilon = 1e-6

        # Manually compute the mean
        pooled_features = feature_sum_per_frame / (kept_token_count_per_frame + epsilon)

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
