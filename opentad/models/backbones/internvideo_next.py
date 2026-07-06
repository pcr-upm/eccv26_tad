"""
InternVideoNext Backbone for OpenTAD.

Adapted from internvideo_next_large_p14_res224_f16/modeling_internvideo_next.py
to integrate with the OpenTAD framework using mmengine's module registration.
"""

import math
from functools import partial
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

import einops
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from mmengine.model import BaseModule
from mmengine.registry import MODELS

# Import adapter and HeatmapHead from vit_sparse_adapter_poguise
from .vit_sparse_adapter_poguise import EfficientAdapter, HeatmapHead


# =============================================================================
# Flash Attention and Fused Ops Availability Checks
# =============================================================================
FLASH_ATTN_AVAILABLE = False
try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
    from flash_attn.bert_padding import unpad_input, pad_input

    FLASH_ATTN_AVAILABLE = True
except ImportError:
    flash_attn_varlen_qkvpacked_func = None
    unpad_input = None
    pad_input = None

FUSED_MLP_AVAILABLE = False
try:
    from flash_attn.modules.mlp import FusedMLP

    FUSED_MLP_AVAILABLE = True
except ImportError:
    FusedMLP = None

FUSED_RMSNORM_AVAILABLE = False
try:
    from flash_attn.ops.rms_norm import DropoutAddRMSNorm

    FUSED_RMSNORM_AVAILABLE = True
except ImportError:
    DropoutAddRMSNorm = None


# =============================================================================
# Pre-Attention Token Scoring (for early memory reduction)
# =============================================================================
class SpatioTemporalTokenScorer(nn.Module):
    """
    Lightweight 3D-aware token scorer for pre-attention pruning.
    Reduces peak memory by selecting tokens BEFORE the first attention operation.

    Uses depthwise separable 3D convolutions to capture spatial and temporal context
    with minimal overhead.

    Args:
        embed_dims: Input token dimension
        reduction_ratio: Hidden dimension reduction (default: 4)
        use_temporal_info: Whether to use temporal context (default: True)
    """

    def __init__(
        self,
        embed_dims: int,
        reduction_ratio: int = 4,
        use_temporal_info: bool = True,
    ):
        super().__init__()

        self.embed_dims = embed_dims
        self.use_temporal_info = use_temporal_info

        hidden_dims = max(embed_dims // reduction_ratio, 32)

        # Layer norm for input normalization
        self.norm = nn.LayerNorm(embed_dims)

        # Spatial scoring: depthwise separable 2D conv (applied per frame)
        self.spatial_dw = nn.Conv2d(
            embed_dims,
            embed_dims,
            kernel_size=3,
            padding=1,
            groups=embed_dims,
            bias=False,
        )
        self.spatial_pw = nn.Conv2d(embed_dims, hidden_dims, kernel_size=1)

        # Temporal scoring: 1D conv along time dimension
        if use_temporal_info:
            self.temporal_conv = nn.Conv1d(
                hidden_dims,
                hidden_dims,
                kernel_size=3,
                padding=1,
                groups=min(hidden_dims, 8),
            )

        # Final scoring head
        self.score_head = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden_dims, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values for stable training start."""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        T: int,
        H: int,
        W: int,
        num_special_tokens: int = 4,
    ) -> torch.Tensor:
        """
        Compute importance scores for each patch token.

        Args:
            x: Input tokens [B, num_special + T*H*W, C]
            T: Number of temporal frames
            H: Patch grid height
            W: Patch grid width
            num_special_tokens: Number of special tokens (CLS, heatmap) to skip

        Returns:
            scores: Importance scores [B, T*H*W]
        """
        B, N, C = x.shape

        # Extract patch tokens (skip special tokens)
        patch_tokens = x[:, num_special_tokens:, :]  # [B, T*H*W, C]

        # Normalize
        patch_tokens = self.norm(patch_tokens)

        # Reshape to 3D: [B, T, H, W, C]
        patch_3d = patch_tokens.view(B, T, H, W, C)

        # Spatial scoring: process each frame
        # [B, T, H, W, C] -> [B*T, C, H, W]
        spatial_in = patch_3d.permute(0, 1, 4, 2, 3).reshape(B * T, C, H, W)
        spatial_out = self.spatial_pw(
            self.spatial_dw(spatial_in)
        )  # [B*T, hidden, H, W]
        hidden_dims = spatial_out.shape[1]

        # [B*T, hidden, H, W] -> [B, T, H, W, hidden]
        spatial_out = spatial_out.view(B, T, hidden_dims, H, W).permute(0, 1, 3, 4, 2)

        # Temporal scoring (if enabled)
        if self.use_temporal_info and T > 1:
            # [B, T, H, W, hidden] -> [B*H*W, hidden, T]
            temporal_in = spatial_out.permute(0, 2, 3, 1, 4).reshape(
                B * H * W, T, hidden_dims
            )
            temporal_in = temporal_in.permute(0, 2, 1)  # [B*H*W, hidden, T]
            temporal_out = self.temporal_conv(temporal_in)  # [B*H*W, hidden, T]
            # [B*H*W, hidden, T] -> [B, T, H, W, hidden]
            temporal_out = temporal_out.permute(0, 2, 1).view(B, H, W, T, hidden_dims)
            temporal_out = temporal_out.permute(0, 3, 1, 2, 4)  # [B, T, H, W, hidden]
            features = spatial_out + 0.1 * temporal_out
        else:
            features = spatial_out

        # Compute scores: [B, T, H, W, hidden] -> [B, T, H, W, 1] -> [B, T*H*W]
        scores = self.score_head(features).squeeze(-1)  # [B, T, H, W]
        scores = scores.reshape(B, T * H * W)

        return scores


def gumbel_softmax_topk(
    scores: torch.Tensor, k: int, tau: float = 1.0, hard: bool = True
) -> tuple:
    """
    Gumbel-softmax based differentiable top-k selection.

    Args:
        scores: [B, N] importance scores
        k: Number of tokens to select
        tau: Temperature (lower = harder selection)
        hard: If True, use straight-through estimator

    Returns:
        indices: [B, k] selected indices
        soft_weights: [B, N] soft selection weights (for gradient flow)
    """
    B, N = scores.shape

    if not scores.requires_grad or not hard:
        # Inference mode: simple top-k
        _, indices = torch.topk(scores, k, dim=1, largest=True)
        indices = indices.sort(dim=1)[0]
        soft_weights = torch.zeros_like(scores)
        soft_weights.scatter_(1, indices, 1.0)
        return indices, soft_weights

    # Training mode: Gumbel-softmax
    # Add Gumbel noise for stochastic selection
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(scores) + 1e-10) + 1e-10)
    noisy_scores = (scores + gumbel_noise) / tau

    # Soft selection weights
    soft_weights = F.softmax(noisy_scores, dim=-1)

    # Hard selection (straight-through)
    _, indices = torch.topk(noisy_scores, k, dim=1, largest=True)
    indices = indices.sort(dim=1)[0]

    # Create hard mask
    hard_weights = torch.zeros_like(scores)
    hard_weights.scatter_(1, indices, 1.0)

    # Straight-through: forward uses hard, backward uses soft
    soft_weights = hard_weights - soft_weights.detach() + soft_weights

    return indices, soft_weights


class PreAttentionTokenPruner(nn.Module):
    """
    Pre-attention token pruning module.

    Applies lightweight scoring and differentiable selection before the first
    transformer block to reduce peak memory usage.

    Args:
        embed_dims: Token embedding dimension
        keep_ratio: Fraction of tokens to keep (0.0-1.0)
        scorer_type: Type of scorer ('spatiotemporal', 'mlp', 'norm')
        reduction_ratio: Hidden dimension reduction for scorer
        use_temporal_info: Whether scorer uses temporal context
        tau_start: Initial Gumbel temperature
        tau_end: Final Gumbel temperature (annealed during training)
        tau_steps: Number of steps for temperature annealing
    """

    def __init__(
        self,
        embed_dims: int,
        keep_ratio: float = 0.7,
        scorer_type: str = "spatiotemporal",
        reduction_ratio: int = 4,
        use_temporal_info: bool = True,
        tau_start: float = 1.0,
        tau_end: float = 0.1,
        tau_steps: int = 10000,
    ):
        super().__init__()

        self.keep_ratio = keep_ratio
        self.scorer_type = scorer_type
        self.tau_start = tau_start
        self.tau_end = tau_end
        self.tau_steps = tau_steps

        # Current temperature (updated during training)
        self.register_buffer("tau", torch.tensor(tau_start))
        self.register_buffer("step_count", torch.tensor(0))

        # Initialize scorer based on type
        if scorer_type == "spatiotemporal":
            self.scorer = SpatioTemporalTokenScorer(
                embed_dims=embed_dims,
                reduction_ratio=reduction_ratio,
                use_temporal_info=use_temporal_info,
            )
        elif scorer_type == "mlp":
            hidden_dims = max(embed_dims // reduction_ratio, 32)
            self.scorer = nn.Sequential(
                nn.LayerNorm(embed_dims),
                nn.Linear(embed_dims, hidden_dims),
                nn.GELU(),
                nn.Linear(hidden_dims, 1),
            )
        elif scorer_type == "norm":
            # Norm-based scoring (parameter-free, fastest)
            self.scorer = nn.LayerNorm(embed_dims)
        else:
            raise ValueError(f"Unknown scorer_type: {scorer_type}")

    def _update_temperature(self):
        """Update Gumbel temperature with cosine annealing."""
        if self.training:
            progress = min(self.step_count.item() / self.tau_steps, 1.0)
            # Cosine annealing
            tau = self.tau_end + 0.5 * (self.tau_start - self.tau_end) * (
                1 + math.cos(math.pi * progress)
            )
            self.tau.fill_(tau)
            self.step_count.add_(1)

    def forward(
        self,
        x: torch.Tensor,
        T: int,
        H: int,
        W: int,
        num_special_tokens: int,
        idx: torch.Tensor,
        neighbor_indices: Optional[torch.Tensor] = None,
    ) -> tuple:
        """
        Apply pre-attention token pruning.

        Args:
            x: Input tokens [B, num_special + T*H*W, C]
            T: Number of temporal frames
            H: Patch grid height
            W: Patch grid width
            num_special_tokens: Number of special tokens to preserve
            idx: Current patch indices [B, T*H*W]
            neighbor_indices: Neighbor indices for sparse conv [B, T*H*W, 9] or None

        Returns:
            x_pruned: Pruned tokens [B, num_special + num_keep, C]
            idx_pruned: Updated indices [B, num_keep]
            neighbor_indices_pruned: Updated neighbor indices or None
            selection_weights: Soft selection weights for gradient flow
        """
        B, N, C = x.shape
        num_patches = T * H * W

        # Update temperature
        self._update_temperature()

        # Compute scores
        if self.scorer_type == "spatiotemporal":
            scores = self.scorer(x, T, H, W, num_special_tokens)  # [B, T*H*W]
        elif self.scorer_type == "mlp":
            patch_tokens = x[:, num_special_tokens:, :]
            scores = self.scorer(patch_tokens).squeeze(-1)  # [B, T*H*W]
        elif self.scorer_type == "norm":
            patch_tokens = x[:, num_special_tokens:, :]
            scores = self.scorer(patch_tokens).norm(dim=-1)  # [B, T*H*W]

        # Calculate number to keep
        num_keep = max(1, int(num_patches * self.keep_ratio))

        # Differentiable selection
        keep_indices, selection_weights = gumbel_softmax_topk(
            scores, num_keep, tau=self.tau.item(), hard=True
        )

        # Gather selected tokens
        special_tokens = x[:, :num_special_tokens, :]  # [B, num_special, C]
        patch_tokens = x[:, num_special_tokens:, :]  # [B, T*H*W, C]

        # Use repeat instead of expand to avoid stride issues with AMP
        indices_expanded = keep_indices.unsqueeze(-1).repeat(1, 1, C)
        selected_patches = torch.gather(
            patch_tokens, 1, indices_expanded
        )  # [B, num_keep, C]

        # During training, add gradient bridge to enable gradient flow to scorer
        # Uses trick: add (soft - hard) * tokens, which is 0 in forward but has gradients
        if self.training and scores.requires_grad:
            # Gather weights for selected tokens
            selected_soft = torch.gather(
                selection_weights, 1, keep_indices
            )  # [B, num_keep]
            # Hard weights are 1.0 for selected tokens (after straight-through)
            # Add (soft - 1) * patches as gradient bridge
            # Forward: adds ~0 since straight-through makes soft ≈ hard
            # Backward: gradients flow through soft_weights to scores to scorer
            grad_bridge = (selected_soft - 1.0).unsqueeze(
                -1
            ) * selected_patches.detach()
            selected_patches = selected_patches + grad_bridge

        # Reconstruct x with special tokens + selected patches
        x_pruned = torch.cat([special_tokens, selected_patches], dim=1)

        # Update global indices
        idx_pruned = torch.gather(idx, 1, keep_indices)

        # Update neighbor indices if present
        neighbor_indices_pruned = None
        if neighbor_indices is not None:
            neighbor_indices_pruned = self._update_neighbor_indices(
                neighbor_indices, keep_indices, B, num_patches, num_keep
            )

        return x_pruned, idx_pruned, neighbor_indices_pruned, selection_weights

    def _update_neighbor_indices(
        self,
        neighbor_indices: torch.Tensor,
        keep_indices: torch.Tensor,
        B: int,
        N_prev: int,
        N_new: int,
    ) -> torch.Tensor:
        """Update neighbor indices after token pruning."""
        device = neighbor_indices.device

        # Gather kept neighbor indices
        idx_expand_9 = keep_indices.unsqueeze(-1).repeat(1, 1, 9)
        neighbor_indices_kept = torch.gather(neighbor_indices, 1, idx_expand_9)

        # Create mapping from old indices to new indices
        global_mapping = torch.full((B * N_prev,), -1, device=device, dtype=torch.long)

        offsets_old = torch.arange(B, device=device) * N_prev
        old_global_kept = keep_indices + offsets_old.view(B, 1)

        offsets_new = torch.arange(B, device=device) * N_new
        new_local_kept = torch.arange(N_new, device=device).unsqueeze(0).expand(B, -1)
        new_global_kept = new_local_kept + offsets_new.view(B, 1)

        global_mapping[old_global_kept.reshape(-1)] = new_global_kept.reshape(-1)

        # Map old neighbor indices to new
        valid_mask = neighbor_indices_kept != -1
        flat_neighbors = neighbor_indices_kept.reshape(-1)
        flat_valid = valid_mask.reshape(-1)

        new_flat = torch.where(
            flat_valid,
            global_mapping[torch.clamp(flat_neighbors, min=0)],
            torch.tensor(-1, device=device, dtype=torch.long),
        )
        neighbor_indices_pruned = new_flat.view(B, N_new, 9)

        return neighbor_indices_pruned


# =============================================================================
# Position Embedding Functions
# =============================================================================
def get_3d_sincos_pos_embed(
    embed_dim, grid_size, t_size, cls_token=False, cls_token_num=4
):
    """
    grid_size: int of the grid height and width
    t_size: int of the temporal size
    return:
    pos_embed: [t_size*grid_size*grid_size, embed_dim] or
               [cls_token_num+t_size*grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    assert embed_dim % 4 == 0
    embed_dim_spatial = embed_dim // 4 * 3
    embed_dim_temporal = embed_dim // 4

    # spatial
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed_spatial = get_2d_sincos_pos_embed_from_grid(embed_dim_spatial, grid)

    # temporal
    grid_t = np.arange(t_size, dtype=np.float32)
    pos_embed_temporal = get_1d_sincos_pos_embed_from_grid(embed_dim_temporal, grid_t)

    # concat: [T, H, W] order
    pos_embed_temporal = pos_embed_temporal[:, np.newaxis, :]
    pos_embed_temporal = np.repeat(
        pos_embed_temporal, grid_size**2, axis=1
    )  # [T, H*W, D // 4]
    pos_embed_spatial = pos_embed_spatial[np.newaxis, :, :]
    pos_embed_spatial = np.repeat(
        pos_embed_spatial, t_size, axis=0
    )  # [T, H*W, D // 4 * 3]

    pos_embed = np.concatenate([pos_embed_temporal, pos_embed_spatial], axis=-1)
    pos_embed = pos_embed.reshape([-1, embed_dim])  # [T*H*W, D]

    if cls_token:
        pos_embed = np.concatenate(
            [np.zeros([cls_token_num, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim]
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_1d_sincos_pos_embed(embed_dim, t_size, cls_token=False):
    """
    t_size: int of the temporal size
    return:
    pos_embed: [t_size, embed_dim] or [1+t_size, embed_dim]
    """
    grid_t = np.arange(t_size, dtype=np.float32)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid_t)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# =============================================================================
# FlashAttention Module
# =============================================================================
class FlashAttention(nn.Module):
    """Implement the scaled dot product attention with softmax using flash_attn."""

    def __init__(
        self, softmax_scale=None, attention_dropout=0.0, device=None, dtype=None
    ):
        super().__init__()
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout

    def forward(
        self,
        qkv,
        key_padding_mask=None,
        causal=False,
        cu_seqlens=None,
        max_s=None,
        need_weights=False,
    ):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            qkv: The tensor containing the query, key, and value.
                 (B, S, 3, H, D) if key_padding_mask is None, or (nnz, 3, h, d) if unpadded
            key_padding_mask: a bool tensor of shape (B, S)
        """
        assert not need_weights
        # assert qkv.dtype in [torch.float16, torch.bfloat16]
        assert qkv.is_cuda

        if cu_seqlens is None:
            batch_size = qkv.shape[0]
            seqlen = qkv.shape[1]
            if key_padding_mask is None:
                qkv = rearrange(qkv, "b s ... -> (b s) ...")
                max_s = seqlen
                cu_seqlens = torch.arange(
                    0,
                    (batch_size + 1) * seqlen,
                    step=seqlen,
                    dtype=torch.int32,
                    device=qkv.device,
                )
                output = flash_attn_varlen_qkvpacked_func(
                    qkv,
                    cu_seqlens,
                    max_s,
                    self.dropout_p if self.training else 0.0,
                    softmax_scale=self.softmax_scale,
                    causal=causal,
                )
                output = rearrange(output, "(b s) ... -> b s ...", b=batch_size)
            else:
                nheads = qkv.shape[-2]
                x = rearrange(qkv, "b s three h d -> b s (three h d)")
                x_unpad, indices, cu_seqlens, max_s = unpad_input(x, key_padding_mask)
                x_unpad = rearrange(
                    x_unpad, "nnz (three h d) -> nnz three h d", three=3, h=nheads
                )
                output_unpad = flash_attn_varlen_qkvpacked_func(
                    x_unpad,
                    cu_seqlens,
                    max_s,
                    self.dropout_p if self.training else 0.0,
                    softmax_scale=self.softmax_scale,
                    causal=causal,
                )
                output = rearrange(
                    pad_input(
                        rearrange(output_unpad, "nnz h d -> nnz (h d)"),
                        indices,
                        batch_size,
                        seqlen,
                    ),
                    "b s (h d) -> b s h d",
                    h=nheads,
                )
        else:
            assert max_s is not None
            output = flash_attn_varlen_qkvpacked_func(
                qkv,
                cu_seqlens,
                max_s,
                self.dropout_p if self.training else 0.0,
                softmax_scale=self.softmax_scale,
                causal=causal,
            )

        return output, None


# =============================================================================
# RMSNorm and LayerScale
# =============================================================================
class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        # Cast weight to input_dtype to avoid float32 promotion
        return self.weight.to(input_dtype) * hidden_states.to(input_dtype)


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False, force_fp32=False):
        super().__init__()
        self.inplace = inplace
        self.lr_scale = nn.Parameter(init_values * torch.ones(dim))
        self.force_fp32 = force_fp32

    def forward(self, x):
        if self.force_fp32:
            output_type = x.dtype
            # Keep computation in float32 for stability but return original dtype
            out = (
                x.float().mul_(self.lr_scale.float())
                if self.inplace
                else x.float() * self.lr_scale.float()
            )
            return out.to(dtype=output_type)
        else:
            # Cast lr_scale to input dtype to avoid promotion
            lr_scale = self.lr_scale.to(x.dtype)
            out = x.mul_(lr_scale) if self.inplace else x * lr_scale
            return out


# =============================================================================
# Attention Modules
# =============================================================================
class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        attn_head_dim=None,
        out_dim=None,
    ):
        super().__init__()
        if out_dim is None:
            out_dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim**-0.5
        assert all_head_dim == dim

        self.q = nn.Linear(dim, all_head_dim, bias=False)
        self.k = nn.Linear(dim, all_head_dim, bias=False)
        self.v = nn.Linear(dim, all_head_dim, bias=False)

        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.k_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.k_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, k=None, v=None):
        B, N, C = x.shape
        N_k = k.shape[1]
        N_v = v.shape[1]

        q_bias, k_bias, v_bias = None, None, None
        if self.q_bias is not None:
            q_bias = self.q_bias
            k_bias = self.k_bias
            v_bias = self.v_bias

        q = F.linear(input=x, weight=self.q.weight, bias=q_bias)
        q = (
            q.reshape(B, N, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)
        )  # (B, N_head, N_q, dim)

        k = F.linear(input=k, weight=self.k.weight, bias=k_bias)
        k = k.reshape(B, N_k, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)

        v = F.linear(input=v, weight=self.v.weight, bias=v_bias)
        v = v.reshape(B, N_v, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # (B, N_head, N_q, N_k)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class AttentiveBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        attn_head_dim=None,
        out_dim=None,
    ):
        super().__init__()

        self.norm1_q = norm_layer(dim)
        self.norm1_k = norm_layer(dim)
        self.norm1_v = norm_layer(dim)
        self.cross_attn = CrossAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            attn_head_dim=attn_head_dim,
            out_dim=out_dim,
        )

        if drop_path > 0.0:
            print(f"Use DropPath in projector: {drop_path}")
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x_q, x_kv, pos_q, pos_k, bool_masked_pos, rel_pos_bias=None):
        x_q = self.norm1_q(x_q + pos_q)
        x_k = self.norm1_k(x_kv + pos_k)
        x_v = self.norm1_v(x_kv)
        x = self.cross_attn(x_q, k=x_k, v=x_v)
        return x


class AttentionPoolingBlock(AttentiveBlock):
    def forward(self, x):
        x_q = x.mean(1, keepdim=True)
        x_kv, pos_q, pos_k = x, 0, 0
        x = super().forward(
            x_q, x_kv, pos_q, pos_k, bool_masked_pos=None, rel_pos_bias=None
        )
        x = x.squeeze(1)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        use_flash_attn=False,
        causal=False,
        norm_layer=nn.LayerNorm,
        qk_normalization=False,
        use_fused_rmsnorm=False,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.use_flash_attn = use_flash_attn and FLASH_ATTN_AVAILABLE
        if self.use_flash_attn:
            self.causal = causal
            self.inner_attn = FlashAttention(attention_dropout=attn_drop)

        self.qk_normalization = qk_normalization
        self.q_norm = norm_layer(dim) if qk_normalization else nn.Identity()
        self.k_norm = norm_layer(dim) if qk_normalization else nn.Identity()
        self.use_fused_rmsnorm = use_fused_rmsnorm

    def _naive_attn(self, x):
        """Use PyTorch's scaled_dot_product_attention (includes flash attention backend)."""
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)

        if self.qk_normalization:
            B_, H_, N_, D_ = q.shape
            if self.use_fused_rmsnorm:
                q = (
                    self.q_norm(q.transpose(1, 2).flatten(-2, -1))[0]
                    .view(B_, N_, H_, D_)
                    .transpose(1, 2)
                )
                k = (
                    self.k_norm(k.transpose(1, 2).flatten(-2, -1))[0]
                    .view(B_, N_, H_, D_)
                    .transpose(1, 2)
                )
            else:
                q = (
                    self.q_norm(q.transpose(1, 2).flatten(-2, -1))
                    .view(B_, N_, H_, D_)
                    .transpose(1, 2)
                )
                k = (
                    self.k_norm(k.transpose(1, 2).flatten(-2, -1))
                    .view(B_, N_, H_, D_)
                    .transpose(1, 2)
                )

        # Use PyTorch's efficient SDPA (auto-selects flash attention when available)
        dropout_p = self.attn_drop.p if self.training else 0.0
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def _flash_attn(self, x, key_padding_mask=None, need_weights=False):
        qkv = self.qkv(x)
        qkv = rearrange(
            qkv, "b s (three h d) -> b s three h d", three=3, h=self.num_heads
        )

        if self.qk_normalization:
            q, k, v = qkv.unbind(2)
            if self.use_fused_rmsnorm:
                q = self.q_norm(q.flatten(-2, -1))[0].view(q.shape)
                k = self.k_norm(k.flatten(-2, -1))[0].view(k.shape)
            else:
                q = self.q_norm(q.flatten(-2, -1)).view(q.shape)
                k = self.k_norm(k.flatten(-2, -1)).view(k.shape)
            qkv = torch.stack([q, k, v], dim=2)

        context, _ = self.inner_attn(
            qkv,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            causal=self.causal,
        )
        outs = self.proj(rearrange(context, "b s h d -> b s (h d)"))
        outs = self.proj_drop(outs)
        return outs

    def forward(self, x):
        x = self._naive_attn(x) if not self.use_flash_attn else self._flash_attn(x)
        return x


# =============================================================================
# Attention with Token Selection (for token pruning)
# =============================================================================
class AttentionWithTokenSelection(nn.Module):
    """
    Attention module with token selection capability.

    Extends the standard attention to compute importance scores for token pruning.
    When keep_rate < 1.0, selects top-k tokens based on CLS token attention scores.
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        use_flash_attn=False,
        causal=False,
        norm_layer=nn.LayerNorm,
        qk_normalization=False,
        use_fused_rmsnorm=False,
        keep_rate=1.0,
        cls_token_num=4,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.keep_rate = keep_rate
        self.cls_token_num = cls_token_num
        assert 0 < keep_rate <= 1, f"keep_rate must be in (0, 1], got {keep_rate}"

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.use_flash_attn = use_flash_attn and FLASH_ATTN_AVAILABLE
        if self.use_flash_attn:
            self.causal = causal
            self.inner_attn = FlashAttention(attention_dropout=attn_drop)

        self.qk_normalization = qk_normalization
        self.q_norm = norm_layer(dim) if qk_normalization else nn.Identity()
        self.k_norm = norm_layer(dim) if qk_normalization else nn.Identity()
        self.use_fused_rmsnorm = use_fused_rmsnorm

    def forward(self, x, last_idx=None):
        """
        Forward pass with optional token selection.

        Args:
            x: Input tensor (B, N, C)
            last_idx: Previous token indices (B, N_prev) or None

        Returns:
            x: Output tensor (B, N, C)
            idx: New token indices (B, N_keep) if keep_rate < 1, else last_idx
        """
        B, N, C = x.shape

        # Compute QKV
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)

        if self.qk_normalization:
            B_, H_, N_, D_ = q.shape
            if self.use_fused_rmsnorm:
                q = (
                    self.q_norm(q.transpose(1, 2).flatten(-2, -1))[0]
                    .view(B_, N_, H_, D_)
                    .transpose(1, 2)
                )
                k = (
                    self.k_norm(k.transpose(1, 2).flatten(-2, -1))[0]
                    .view(B_, N_, H_, D_)
                    .transpose(1, 2)
                )
            else:
                q = (
                    self.q_norm(q.transpose(1, 2).flatten(-2, -1))
                    .view(B_, N_, H_, D_)
                    .transpose(1, 2)
                )
                k = (
                    self.k_norm(k.transpose(1, 2).flatten(-2, -1))
                    .view(B_, N_, H_, D_)
                    .transpose(1, 2)
                )

        if self.keep_rate >= 1.0:
            # Standard attention without token selection
            if self.use_flash_attn:
                qkv_packed = torch.stack([q, k, v], dim=2)
                qkv_packed = rearrange(qkv_packed, "b h n three d -> b n three h d")
                context, _ = self.inner_attn(qkv_packed, causal=self.causal)
                x = rearrange(context, "b n h d -> b n (h d)")
            else:
                # Use PyTorch's efficient SDPA (auto-selects flash attention when available)
                dropout_p = self.attn_drop.p if self.training else 0.0
                x = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
                x = x.transpose(1, 2).reshape(B, N, C)

            x = self.proj(x)
            x = self.proj_drop(x)
            return x, last_idx

        # Token selection: compute attention and select top-k
        # Use efficient attention for output
        if self.use_flash_attn:
            qkv_packed = torch.stack([q, k, v], dim=2)
            qkv_packed = rearrange(qkv_packed, "b h n three d -> b n three h d")
            context, _ = self.inner_attn(qkv_packed, causal=self.causal)
            x = rearrange(context, "b n h d -> b n (h d)")
        else:
            # Use PyTorch's efficient SDPA (auto-selects flash attention when available)
            dropout_p = self.attn_drop.p if self.training else 0.0
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
            x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        # Compute importance scores from CLS tokens (mean of all CLS token attentions)
        # q shape: (B, num_heads, N, head_dim)
        # Use only CLS tokens for importance scoring
        q_cls = q[:, :, : self.cls_token_num]  # (B, num_heads, cls_token_num, head_dim)
        attn_scores = (q_cls * self.scale) @ k.transpose(
            -2, -1
        )  # (B, num_heads, cls_token_num, N)
        attn_scores = attn_scores.softmax(dim=-1)

        # Sum over CLS tokens, mean over heads
        importance = attn_scores.sum(dim=2).mean(dim=1)  # (B, N)

        # Exclude CLS tokens from selection
        importance_patch = importance[:, self.cls_token_num :]  # (B, N - cls_token_num)

        # Select top-k tokens
        num_patch_tokens = N - self.cls_token_num
        num_keep = math.ceil(self.keep_rate * num_patch_tokens)
        num_keep = min(num_keep, importance_patch.shape[1])

        if num_keep <= 0:
            return x, torch.empty(B, 0, dtype=last_idx.dtype, device=x.device)

        _, idx_topk = torch.topk(importance_patch, num_keep, dim=1, largest=True)
        idx = idx_topk.sort(dim=1)[0]  # Sort for consistent ordering

        return x, idx


# =============================================================================
# MLP Module
# =============================================================================
class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        bias=True,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


# =============================================================================
# Transformer Block
# =============================================================================
class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        init_values=None,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        use_flash_attn=False,
        use_fused_mlp=False,
        fused_mlp_heuristic=1,
        with_cp=False,
        qk_normalization=False,
        layerscale_no_force_fp32=False,
        use_fused_rmsnorm=False,
        # Adapter parameters
        use_adapter=False,
        adapter_mlp_ratio=0.25,
        adapter_conv_type="sparse_conv",
        deform_conv_groups=1,
        adapter_use_attn=False,
        # Token selection parameters
        keep_rate=1.0,
        cls_token_num=4,
    ):
        super().__init__()

        # Validate fused rmsnorm availability
        if use_fused_rmsnorm and not FUSED_RMSNORM_AVAILABLE:
            use_fused_rmsnorm = False

        # Adapter configuration
        self.use_adapter = use_adapter
        self.adapter_conv_type = adapter_conv_type
        self.needs_neighbor_indices = use_adapter and adapter_conv_type == "sparse_conv"

        # Token selection configuration
        self.keep_rate = keep_rate
        self.cls_token_num = cls_token_num

        self.norm1 = norm_layer(dim)

        # Use AttentionWithTokenSelection when keep_rate < 1
        if keep_rate < 1.0:
            self.attn = AttentionWithTokenSelection(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=drop,
                use_flash_attn=use_flash_attn,
                causal=False,
                norm_layer=norm_layer,
                qk_normalization=qk_normalization,
                use_fused_rmsnorm=use_fused_rmsnorm,
                keep_rate=keep_rate,
                cls_token_num=cls_token_num,
            )
        else:
            self.attn = Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=drop,
                use_flash_attn=use_flash_attn,
                causal=False,
                norm_layer=norm_layer,
                qk_normalization=qk_normalization,
                use_fused_rmsnorm=use_fused_rmsnorm,
            )
        self.ls1 = (
            LayerScale(
                dim, init_values=init_values, force_fp32=(not layerscale_no_force_fp32)
            )
            if init_values
            else nn.Identity()
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if use_fused_mlp and FUSED_MLP_AVAILABLE and FusedMLP is not None:
            self.mlp = FusedMLP(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                heuristic=fused_mlp_heuristic,
            )
        else:
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                drop=drop,
            )
        self.ls2 = (
            LayerScale(
                dim, init_values=init_values, force_fp32=(not layerscale_no_force_fp32)
            )
            if init_values
            else nn.Identity()
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.with_cp = with_cp
        self.use_fused_rmsnorm = use_fused_rmsnorm

        # Initialize adapter if enabled
        if self.use_adapter:
            self.adapter = EfficientAdapter(
                embed_dims=dim,
                mlp_ratio=adapter_mlp_ratio,
                conv_type=adapter_conv_type,
                deformable_groups=deform_conv_groups,
                use_attn=adapter_use_attn,
                n_key_tokens=cls_token_num,
                heatmap_grid_size=None,
            )

    def forward(
        self,
        x,
        residual=None,
        idx=None,
        h=None,
        w=None,
        total_num_patches=None,
        neighbor_indices=None,
    ):
        """
        Forward pass with optional token selection.

        Args:
            x: Input tensor (B, N, C)
            residual: Residual tensor for fused rmsnorm
            idx: Current token indices (B, N_patches)
            h, w: Spatial dimensions
            total_num_patches: Total number of patches (T * H * W)
            neighbor_indices: Neighbor indices for sparse conv (B, N_patches, 9)

        Returns:
            x: Output tensor
            idx: Updated token indices (after pruning if keep_rate < 1)
            neighbor_indices: Updated neighbor indices
        """

        def _inner_forward(x, residual, idx, neighbor_indices):
            B, N, C = x.shape

            if self.keep_rate < 1.0:
                # Token selection path
                x_norm = self.norm1(x) if not self.use_fused_rmsnorm else None

                if self.use_fused_rmsnorm:
                    x_norm, residual = self.norm1(x, residual)
                    attn_out, new_idx = self.attn(x_norm, idx)
                    x = self.drop_path1(self.ls1(attn_out))
                else:
                    attn_out, new_idx = self.attn(x_norm, idx)
                    x = x + self.drop_path1(self.ls1(attn_out))

                # Apply token pruning if indices changed
                if new_idx is not None and new_idx.shape[1] < (N - self.cls_token_num):
                    # Separate CLS tokens and patch tokens
                    x_cls = x[:, : self.cls_token_num]
                    x_patches = x[:, self.cls_token_num :].contiguous()

                    # Gather kept patches - use repeat() instead of expand() to avoid
                    # stride 0 issues with AMP and checkpointing
                    idx_expanded = new_idx.unsqueeze(-1).repeat(1, 1, C)
                    x_patches_kept = torch.gather(x_patches, dim=1, index=idx_expanded)

                    # Update global indices
                    if idx is not None:
                        idx = torch.gather(idx, dim=1, index=new_idx)

                    # Update neighbor indices for sparse conv
                    if neighbor_indices is not None and self.needs_neighbor_indices:
                        neighbor_indices = self._update_neighbor_indices(
                            neighbor_indices,
                            new_idx,
                            B,
                            x_patches.shape[1],
                            x_patches_kept.shape[1],
                        )

                    # Concatenate back
                    x = torch.cat([x_cls, x_patches_kept], dim=1)

                    # Handle residual for fused rmsnorm
                    if residual is not None:
                        res_cls = residual[:, : self.cls_token_num]
                        res_patches = residual[:, self.cls_token_num :].contiguous()
                        res_patches_kept = torch.gather(
                            res_patches, dim=1, index=idx_expanded
                        )
                        residual = torch.cat([res_cls, res_patches_kept], dim=1)

                # MLP
                if self.use_fused_rmsnorm:
                    x_norm2, residual = self.norm2(x, residual)
                    x = self.drop_path2(self.ls2(self.mlp(x_norm2)))
                else:
                    x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
            else:
                # Standard path without token selection
                if self.use_fused_rmsnorm:
                    x, residual = self.norm1(x, residual)
                    x = self.drop_path1(self.ls1(self.attn(x)))
                    x, residual = self.norm2(x, residual)
                    x = self.drop_path2(self.ls2(self.mlp(x)))
                else:
                    assert residual is None
                    x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
                    x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))

            # Apply adapter after MLP if enabled
            if self.use_adapter and idx is not None:
                x = self.adapter(x, h, w, idx, total_num_patches, neighbor_indices)

            return x, idx, neighbor_indices

        if self.with_cp and x.requires_grad:
            x, idx, neighbor_indices = checkpoint.checkpoint(
                _inner_forward, x, residual, idx, neighbor_indices, use_reentrant=False
            )
        else:
            x, idx, neighbor_indices = _inner_forward(
                x, residual, idx, neighbor_indices
            )

        return x, idx, neighbor_indices

    def _update_neighbor_indices(
        self, neighbor_indices, idx_from_topk, B, N_prev, N_new
    ):
        """
        Update neighbor indices after token pruning.

        Maps old indices to new indices in the pruned token set.
        """
        device = neighbor_indices.device

        # Gather kept neighbor indices - use repeat() instead of expand()
        # to avoid stride 0 issues with AMP and checkpointing
        idx_expand_9 = idx_from_topk.unsqueeze(-1).repeat(1, 1, 9)
        neighbor_indices_kept = torch.gather(neighbor_indices, 1, idx_expand_9)

        # Create mapping from old indices to new indices
        # Pre-allocate mapping buffer filled with -1
        global_mapping = torch.full((B * N_prev,), -1, device=device, dtype=torch.long)

        # Compute offsets for batch indexing
        offsets_old = torch.arange(B, device=device) * N_prev
        old_global_kept = idx_from_topk + offsets_old.view(B, 1)

        offsets_new = torch.arange(B, device=device) * N_new
        new_local_kept = (
            torch.arange(idx_from_topk.shape[1], device=device)
            .unsqueeze(0)
            .expand(B, -1)
        )
        new_global_kept = new_local_kept + offsets_new.view(B, 1)

        global_mapping[old_global_kept.reshape(-1)] = new_global_kept.reshape(-1)

        # Map old neighbor indices to new indices
        valid_mask = neighbor_indices_kept != -1
        flat_neighbors = neighbor_indices_kept.reshape(-1)
        flat_valid = valid_mask.reshape(-1)

        new_flat = torch.where(
            flat_valid,
            global_mapping[torch.clamp(flat_neighbors, min=0)],
            torch.tensor(-1, device=device, dtype=torch.long),
        )
        neighbor_indices = new_flat.view(B, N_new, 9)

        return neighbor_indices


# =============================================================================
# Patch Embedding
# =============================================================================
class PatchEmbed(nn.Module):
    """3D Image to Patch Embedding"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        num_frames=8,
        tubelet_size=1,
        norm_layer=None,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.grid_size = (
            num_frames // tubelet_size,
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        )  # (T, H, W)
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]

        self.proj = nn.Conv3d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=(tubelet_size, patch_size[0], patch_size[1]),
            stride=(tubelet_size, patch_size[0], patch_size[1]),
        )

        self.norm = norm_layer(embed_dim)
        self.norm_before = norm_layer(tubelet_size * math.prod(patch_size) * 3)

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1)
        x = einops.rearrange(
            x,
            "b (t1 t2) (ht hp) (wt wp) c -> b (t1 ht wt) (t2 hp wp c)",
            t2=self.tubelet_size,
            hp=self.patch_size[0],
            wp=self.patch_size[1],
        )
        x = self.norm_before(x)
        x = einops.rearrange(
            x,
            "b (t1 ht wt) (t2 hp wp c) -> b (t1 t2) (ht hp) (wt wp) c",
            t1=T // self.tubelet_size,
            ht=H // self.patch_size[0],
            t2=self.tubelet_size,
            hp=self.patch_size[0],
            wp=self.patch_size[1],
        )
        x = x.permute(0, 4, 1, 2, 3)
        x = self.proj(x)
        x = x.flatten(3).permute(0, 2, 3, 1)
        x = self.norm(x)
        return x


# =============================================================================
# Main Backbone Class
# =============================================================================
@MODELS.register_module()
class InternVideoNextBackbone(BaseModule):
    """
    InternVideoNext Backbone for video understanding.

    Adapted for OpenTAD integration with mmengine's module registration.
    Output format: (B, C, T+1) where the first temporal position is the CLS token.

    Args:
        in_chans (int): Number of input channels. Default: 3.
        patch_size (int): Patch size. Default: 14.
        img_size (int): Input image size. Default: 224.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: False.
        drop_path_rate (float): Stochastic depth rate. Default: 0.25.
        embed_dim (int): Embedding dimension. Default: 1408.
        head_drop_path_rate (float): Drop path rate for attention pooling head. Default: 0.0.
        num_heads (int): Number of attention heads. Default: 16.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.3637.
        init_values (float): Init value for Layer Scale. Default: 1e-5.
        qk_normalization (bool): If True, apply normalization to q and k. Default: True.
        depth (int): Number of transformer blocks. Default: 40.
        use_flash_attn (bool): If True, use flash attention. Default: True.
        use_fused_rmsnorm (bool): If True, use fused RMSNorm. Default: True.
        use_fused_mlp (bool): If True, use fused MLP. Default: True.
        fused_mlp_heuristic (int): Heuristic for fused MLP. Default: 1.
        attn_pool_num_heads (int): Number of heads for attention pooling. Default: 16.
        clip_embed_dim (int): CLIP embedding dimension for projector. Default: 768.
        layerscale_no_force_fp32 (bool): If True, don't force fp32 in layer scale. Default: False.
        num_frames (int): Number of input frames. Default: 16.
        tubelet_size (int): Tubelet size for temporal patch embedding. Default: 1.
        sep_pos_embed (bool): If True, use separable position embedding. Default: False.
        use_checkpoint (bool): If True, use gradient checkpointing. Default: False.
        checkpoint_num (int): Number of blocks to checkpoint. Default: 0.
        cls_token_num (int): Number of CLS tokens. Default: 4.
        pretrained (str, optional): Path to pretrained weights. Default: None.
        return_feat_map (bool): If True, return spatial feature map. Default: False.
        init_cfg (dict or list[dict], optional): Initialization config dict. Default: None.
    """

    def __init__(
        self,
        in_chans: int = 3,
        patch_size: int = 14,
        img_size: int = 224,
        qkv_bias: bool = False,
        drop_path_rate: float = 0.25,
        embed_dim: int = 1408,
        head_drop_path_rate: float = 0.0,
        num_heads: int = 16,
        mlp_ratio: float = 4.3637,
        init_values: float = 1e-5,
        qk_normalization: bool = True,
        depth: int = 40,
        use_flash_attn: bool = True,
        use_fused_rmsnorm: bool = True,
        use_fused_mlp: bool = True,
        fused_mlp_heuristic: int = 1,
        attn_pool_num_heads: int = 16,
        clip_embed_dim: int = 768,
        layerscale_no_force_fp32: bool = False,
        num_frames: int = 16,
        tubelet_size: int = 1,
        sep_pos_embed: bool = False,
        use_checkpoint: bool = False,
        checkpoint_num: int = 0,
        cls_token_num: int = 4,
        pretrained: Optional[str] = None,
        return_feat_map: bool = False,
        # Adapter parameters
        adapter_index: List[int] = [],
        adapter_conv_type: Union[str, List[str]] = "sparse_conv",
        adapter_mlp_ratio: float = 0.25,
        deform_conv_groups: int = 1,
        adapter_use_attn: bool = False,
        # Token selection parameters
        keep_rate: float = 1.0,
        token_selection_index: List[int] = [],
        # Landmark parameters
        n_landmarks: int = 0,
        hw_out_conv: tuple = (10, 10),
        # Pre-attention token pruning parameters
        pre_attn_keep_ratio: float = 1.0,  # 1.0 = disabled
        pre_attn_scorer_type: str = "spatiotemporal",
        pre_attn_reduction_ratio: int = 4,
        pre_attn_tau_start: float = 1.0,
        pre_attn_tau_end: float = 0.1,
        pre_attn_tau_steps: int = 10000,
        init_cfg: Optional[Union[Dict, List[Dict]]] = [
            dict(type="TruncNormal", layer="Linear", std=0.02, bias=0.0),
            dict(type="Constant", layer="LayerNorm", val=1.0, bias=0.0),
        ],
    ):
        # Handle pretrained checkpoint loading via mmengine
        if pretrained:
            init_cfg = dict(type="Pretrained", checkpoint=pretrained)
        super().__init__(init_cfg=init_cfg)

        self.cls_token_num = cls_token_num
        self.return_feat_map = return_feat_map
        self.embed_dim = embed_dim
        self.n_landmarks = n_landmarks
        self.hw_out_conv = hw_out_conv
        self.n_heatmap_tokens = 0

        # Store adapter configuration
        self.adapter_index = adapter_index
        self.adapter_mlp_ratio = adapter_mlp_ratio
        self.deform_conv_groups = deform_conv_groups
        self.adapter_use_attn = adapter_use_attn

        # Normalize adapter_conv_type to a list (one per block)
        if isinstance(adapter_conv_type, str):
            self.adapter_conv_type = [adapter_conv_type] * depth
        else:
            self.adapter_conv_type = adapter_conv_type
            assert len(self.adapter_conv_type) == depth, (
                f"adapter_conv_type list length ({len(self.adapter_conv_type)}) "
                f"must match depth ({depth})"
            )

        # Check if any block uses sparse_conv and needs neighbor indices
        self.needs_neighbor_indices = (
            len(adapter_index) > 0 and "sparse_conv" in self.adapter_conv_type
        )

        # Store token selection configuration
        self.keep_rate = keep_rate
        self.token_selection_index = token_selection_index
        # Token selection is only applied if token_selection_index is explicitly provided
        # If empty, no blocks will perform token selection

        # Check if flash_attn is available and adjust settings if needed
        if use_flash_attn and not FLASH_ATTN_AVAILABLE:
            print(
                "Warning: flash_attn requested but not available, falling back to native attention"
            )
            use_flash_attn = False
        if use_fused_rmsnorm and not FUSED_RMSNORM_AVAILABLE:
            print(
                "Warning: fused_rmsnorm requested but not available, falling back to standard RMSNorm"
            )
            use_fused_rmsnorm = False
        if use_fused_mlp and not FUSED_MLP_AVAILABLE:
            print(
                "Warning: fused_mlp requested but not available, falling back to standard MLP"
            )
            use_fused_mlp = False

        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.use_flash_attn = use_flash_attn

        if (
            use_fused_rmsnorm
            and FUSED_RMSNORM_AVAILABLE
            and DropoutAddRMSNorm is not None
        ):
            norm_layer_for_blocks = partial(DropoutAddRMSNorm, eps=1e-6, prenorm=True)
        else:
            norm_layer_for_blocks = partial(RMSNorm, eps=1e-6)
        self.norm_layer_for_blocks = norm_layer_for_blocks

        self.patch_embed = PatchEmbed(
            img_size,
            patch_size,
            in_chans,
            embed_dim,
            num_frames=num_frames,
            tubelet_size=tubelet_size,
            norm_layer=partial(RMSNorm, eps=1e-6),
        )
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, cls_token_num, embed_dim))

        # Landmark/heatmap tokens setup
        num_special_tokens = cls_token_num
        if self.n_landmarks > 0:
            self.n_heatmap_tokens = self.hw_out_conv[0] * self.hw_out_conv[1]
            self.heatmap_tokens = nn.Parameter(
                torch.zeros(1, self.n_heatmap_tokens, embed_dim)
            )
            trunc_normal_(self.heatmap_tokens, std=0.02)
            self.heatmap_head = HeatmapHead(
                in_channels=embed_dim,
                in_size=self.hw_out_conv[0],
                out_channels=self.n_landmarks,
                deconv_out_channels=(256, 256),
                deconv_kernel_sizes=(4, 4),
            )
            num_special_tokens += self.n_heatmap_tokens
        self.num_special_tokens = num_special_tokens

        self.sep_pos_embed = sep_pos_embed
        if sep_pos_embed:
            print("Use separable position embedding")
            grid_size = self.patch_embed.grid_size
            self.grid_size = grid_size
            self.pos_embed_spatial = nn.Parameter(
                torch.zeros(1, grid_size[1] * grid_size[2], embed_dim)
            )
            self.pos_embed_temporal = nn.Parameter(
                torch.zeros(1, grid_size[0], embed_dim)
            )
            self.pos_embed_cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            print("Use joint position embedding")
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_patches + cls_token_num, embed_dim)
            )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        # choose which layer to use checkpoint
        with_cp_list = [False] * depth
        if use_checkpoint:
            for idx in range(depth):
                if idx < checkpoint_num:
                    with_cp_list[idx] = True
        print(f"Droppath rate: {dpr}")
        print(f"Checkpoint list: {with_cp_list}")

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer_for_blocks,
                    drop_path=dpr[i],
                    init_values=init_values,
                    attn_drop=0.0,
                    use_flash_attn=use_flash_attn,
                    use_fused_mlp=use_fused_mlp,
                    fused_mlp_heuristic=fused_mlp_heuristic,
                    with_cp=with_cp_list[i],
                    qk_normalization=qk_normalization,
                    layerscale_no_force_fp32=layerscale_no_force_fp32,
                    use_fused_rmsnorm=use_fused_rmsnorm,
                    # Adapter parameters - enable if block index is in adapter_index
                    use_adapter=(i in adapter_index),
                    adapter_mlp_ratio=adapter_mlp_ratio,
                    adapter_conv_type=self.adapter_conv_type[i],
                    deform_conv_groups=deform_conv_groups,
                    adapter_use_attn=adapter_use_attn,
                    # Token selection parameters - enable if block index is in token_selection_index
                    keep_rate=(keep_rate if i in token_selection_index else 1.0),
                    cls_token_num=num_special_tokens,  # Include heatmap tokens as special tokens
                )
                for i in range(depth)
            ]
        )

        # self.clip_projector = AttentionPoolingBlock(
        #     dim=embed_dim,
        #     num_heads=attn_pool_num_heads,
        #     qkv_bias=True,
        #     qk_scale=None,
        #     drop=0.0,
        #     attn_drop=0.0,
        #     drop_path=head_drop_path_rate,
        #     norm_layer=partial(nn.LayerNorm, eps=1e-5),
        #     out_dim=clip_embed_dim,
        # )

        # Initialize pre-attention token pruner if enabled
        self.pre_attn_pruner = None
        if pre_attn_keep_ratio < 1.0:
            self.pre_attn_pruner = PreAttentionTokenPruner(
                embed_dims=embed_dim,
                keep_ratio=pre_attn_keep_ratio,
                scorer_type=pre_attn_scorer_type,
                reduction_ratio=pre_attn_reduction_ratio,
                use_temporal_info=True,
                tau_start=pre_attn_tau_start,
                tau_end=pre_attn_tau_end,
                tau_steps=pre_attn_tau_steps,
            )
            print(
                f"Pre-attention pruner enabled: keep_ratio={pre_attn_keep_ratio}, "
                f"scorer={pre_attn_scorer_type}"
            )

        self.init_pos_embed()
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)
        self.fix_init_weight()

    def init_pos_embed(self):
        """Initialize position embeddings from sincos embeddings."""
        print("Init pos_embed from sincos pos_embed")
        if self.sep_pos_embed:
            pos_embed_spatial = get_2d_sincos_pos_embed(
                self.pos_embed_spatial.shape[-1],
                self.patch_embed.grid_size[1],
            )
            self.pos_embed_spatial.data.copy_(
                torch.from_numpy(pos_embed_spatial).float().unsqueeze(0)
            )
            pos_embed_temporal = get_1d_sincos_pos_embed(
                self.pos_embed_spatial.shape[-1],
                self.patch_embed.grid_size[0],
            )
            self.pos_embed_temporal.data.copy_(
                torch.from_numpy(pos_embed_temporal).float().unsqueeze(0)
            )
        else:
            pos_embed = get_3d_sincos_pos_embed(
                self.pos_embed.shape[-1],
                self.patch_embed.grid_size[1],
                self.patch_embed.grid_size[0],
                cls_token=True,
                cls_token_num=self.cls_token_num,
            )
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def get_initial_neighbor_indices(
        self, h: int, w: int, T: int, device, B: int
    ) -> torch.Tensor:
        """
        Compute neighbor indices for sparse convolution.

        Creates a lookup table where each spatial position has indices of its 8 neighbors
        plus itself (3x3 window). Used for sparse convolution operations.

        Args:
            h: Patch grid height
            w: Patch grid width
            T: Number of temporal frames
            device: Target device for the tensor
            B: Batch size

        Returns:
            Tensor of shape (B, T*H*W, 9) with neighbor indices, -1 for invalid neighbors
        """
        N_spatial = h * w
        N_total = N_spatial * T

        # Build spatial neighbor indices for one frame
        indices = torch.arange(N_spatial, device=device).view(h, w)
        padded = torch.full((h + 2, w + 2), -1, device=device, dtype=torch.long)
        padded[1:-1, 1:-1] = indices

        # Gather 3x3 neighborhood for each position
        neighbors = []
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                neighbors.append(
                    padded[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w].flatten()
                )
        spatial_neighbors = torch.stack(neighbors, dim=1)  # (N_spatial, 9)

        # Replicate for T frames with temporal offsets
        neighbor_indices = spatial_neighbors.unsqueeze(0).expand(T, -1, -1).clone()
        t_offsets = (torch.arange(T, device=device) * N_spatial).view(T, 1, 1)
        mask = neighbor_indices != -1
        neighbor_indices[mask] += t_offsets.expand_as(neighbor_indices)[mask]

        # Flatten to (T*N_spatial, 9) and replicate for batch with batch offsets
        neighbor_indices = neighbor_indices.reshape(N_total, 9)
        neighbor_indices = neighbor_indices.unsqueeze(0).expand(B, -1, -1).clone()
        b_offsets = (torch.arange(B, device=device) * N_total).view(B, 1, 1)
        mask = neighbor_indices != -1
        neighbor_indices[mask] += b_offsets.expand_as(neighbor_indices)[mask]

        return neighbor_indices

    def _freeze_layers(self):
        """
        Freeze all backbone parameters except adapter modules and heatmap modules.

        Use this for adapter-only fine-tuning where the pretrained backbone
        weights remain fixed and only adapters are trained.
        """
        # Freeze patch embedding
        self.patch_embed.eval()
        for param in self.patch_embed.parameters():
            param.requires_grad = False

        # Freeze position embeddings (keep cls_token trainable for adaptation)
        # self.cls_token.requires_grad = False  # Keep trainable
        if hasattr(self, "pos_embed"):
            self.pos_embed.requires_grad = False
        if hasattr(self, "pos_embed_spatial"):
            self.pos_embed_spatial.requires_grad = False
        if hasattr(self, "pos_embed_temporal"):
            self.pos_embed_temporal.requires_grad = False
        if hasattr(self, "pos_embed_cls"):
            self.pos_embed_cls.requires_grad = False

        # Keep heatmap tokens and head trainable
        # (heatmap_tokens and heatmap_head are NOT frozen)

        # Freeze blocks except adapter modules
        for block in self.blocks:
            for name, module in block.named_children():
                if "adapter" not in name.lower():
                    module.eval()
                    for param in module.parameters():
                        param.requires_grad = False

        # Freeze CLIP projector
        # self.clip_projector.eval()
        # for param in self.clip_projector.parameters():
        #     param.requires_grad = False

        # Keep pre-attention pruner trainable
        if self.pre_attn_pruner is not None:
            for param in self.pre_attn_pruner.parameters():
                param.requires_grad = True

        # Print trainable parameters summary
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        print(
            f"Trainable params: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)"
        )

    @property
    def dtype(self):
        return self.patch_embed.proj.weight.dtype

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        no_decay = {
            "pos_embed",
            "pos_embed_spatial",
            "pos_embed_temporal",
            "pos_embed_cls",
            "cls_token",
        }
        # Add pruner buffers to no_weight_decay
        if self.pre_attn_pruner is not None:
            no_decay.add("pre_attn_pruner.tau")
            no_decay.add("pre_attn_pruner.step_count")
        return no_decay

    def expand_pos_embed(self, pos_embed, new_t_size, L, num_extra_tokens=-1):
        """Expand position embedding for different temporal/spatial sizes."""
        pos_embed_checkpoint = pos_embed
        embedding_size = pos_embed_checkpoint.shape[-1]

        if num_extra_tokens == -1:
            num_extra_tokens = self.cls_token_num

        # height (== width) for the checkpoint position embedding
        orig_size = int(
            (
                (pos_embed_checkpoint.shape[-2] - num_extra_tokens)
                // (self.num_frames / self.tubelet_size)
            )
            ** 0.5
        )
        # height (== width) for the new position embedding
        new_size = int(L**0.5)

        # Handle temporal interpolation
        if self.num_frames != new_t_size:
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(1, self.num_frames, -1, embedding_size)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(
                -1, embedding_size, self.num_frames
            )
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=new_t_size, mode="linear"
            )
            pos_tokens = pos_tokens.reshape(1, -1, embedding_size, new_t_size)
            pos_tokens = pos_tokens.permute(0, 3, 1, 2).reshape(1, -1, embedding_size)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            pos_embed_checkpoint = new_pos_embed

        # Handle spatial interpolation
        if orig_size != new_size:
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(
                -1, new_t_size, orig_size, orig_size, embedding_size
            )
            pos_tokens = pos_tokens.reshape(
                -1, orig_size, orig_size, embedding_size
            ).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens,
                size=(new_size, new_size),
                mode="bicubic",
                align_corners=False,
            )
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(
                -1, new_t_size, new_size, new_size, embedding_size
            )
            pos_tokens = pos_tokens.flatten(1, 3)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            pos_embed_checkpoint = new_pos_embed

        return pos_embed_checkpoint

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C, T, H, W)

        Returns:
            Features of shape (B, C, T+1) where first temporal position is CLS token,
            or (B, C, T, H, W) if return_feat_map=True
        """

        x = self.patch_embed(x)

        B, T, L, C = x.shape  # T: temporal; L: spatial (H*W)
        x = x.view([B, T * L, C])

        # Compute spatial dimensions for adapters
        h_patch = w_patch = int(L**0.5)
        total_num_patches = T * L

        # Initialize token indices (all patches, no pruning in InternVideo)
        idx = (
            torch.arange(total_num_patches, device=x.device).unsqueeze(0).expand(B, -1)
        )

        # Compute neighbor indices for sparse_conv adapters
        if self.needs_neighbor_indices:
            neighbor_indices = self.get_initial_neighbor_indices(
                h_patch, w_patch, T, x.device, B
            )
        else:
            neighbor_indices = None

        # Append cls tokens (cast to x.dtype to preserve bf16 under autocast)
        cls_tokens = self.cls_token.to(x.dtype).expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add pos_embed (before heatmap tokens - they don't need pos_embed)
        if self.sep_pos_embed:
            pos_embed = self.pos_embed_spatial.repeat(
                1, self.grid_size[0], 1
            ) + torch.repeat_interleave(
                self.pos_embed_temporal,
                self.grid_size[1] * self.grid_size[2],
                dim=1,
            )
            pos_embed = torch.cat(
                [
                    self.pos_embed_cls.expand(pos_embed.shape[0], -1, -1),
                    pos_embed,
                ],
                1,
            )
        else:
            pos_embed = self.pos_embed

        target_shape = x[0].shape
        if self.pos_embed[0].shape != target_shape:
            pos_embed = self.expand_pos_embed(self.pos_embed, T, L)

        # Cast pos_embed to x.dtype to preserve bf16
        x = x + pos_embed.to(x.dtype)

        # Append heatmap tokens after CLS tokens if landmarks enabled
        # (after pos_embed addition since heatmap tokens don't need positional encoding)
        if self.n_landmarks > 0:
            heatmap_tokens = self.heatmap_tokens.to(x.dtype).expand(B, -1, -1)
            # Insert heatmap tokens after CLS tokens
            x = torch.cat(
                (
                    x[:, : self.cls_token_num, :],
                    heatmap_tokens,
                    x[:, self.cls_token_num :, :],
                ),
                dim=1,
            )

        # Apply pre-attention token pruning (before first transformer block)
        selection_weights = None
        if self.pre_attn_pruner is not None:
            x, idx, neighbor_indices, selection_weights = self.pre_attn_pruner(
                x, T, h_patch, w_patch, self.num_special_tokens, idx, neighbor_indices
            )

        # Apply transformer blocks with adapter spatial info
        residual = None
        for blk in self.blocks:
            if isinstance(x, tuple) and len(x) == 2:
                x, residual = x
            result = blk(
                x,
                residual=residual,
                idx=idx,
                h=h_patch,
                w=w_patch,
                total_num_patches=total_num_patches,
                neighbor_indices=neighbor_indices,
            )
            # Unpack block result (x, idx, neighbor_indices)
            x, idx, neighbor_indices = result
        if isinstance(x, tuple) and len(x) == 2:
            x, residual = x
            if residual is not None:
                x = x + residual

        # Process heatmap tokens if landmarks enabled
        x_heatmap = None
        if self.n_landmarks > 0:
            # Extract heatmap tokens (they are right after CLS tokens)
            start_idx = self.cls_token_num
            end_idx = self.cls_token_num + self.n_heatmap_tokens
            heatmap_tokens_out = x[:, start_idx:end_idx, :]

            # Reshape from sequence to 2D feature map
            heatmap_feats = heatmap_tokens_out.reshape(B, *self.hw_out_conv, C)
            heatmap_feats = heatmap_feats.permute(
                0, 3, 1, 2
            ).contiguous()  # (B, C, H, W)

            # Get the final heatmap prediction
            x_heatmap = self.heatmap_head(heatmap_feats)

        # Extract CLS token (mean of all cls tokens)
        cls_token_out = x[:, : self.cls_token_num, :].mean(
            dim=1, keepdim=True
        )  # (B, 1, C)

        # Extract patch tokens (skip both CLS and heatmap tokens)
        patch_tokens = x[:, self.num_special_tokens :, :]  # (B, N_remaining, C)
        N_remaining = patch_tokens.shape[1]

        if self.return_feat_map:
            # Return spatial feature map: (B, C, T, H, W)
            # Note: Cannot return proper spatial map after token selection
            if N_remaining == T * L:
                H = W = int(L**0.5)
                feat_map = patch_tokens.reshape(B, T, H, W, C).permute(0, 4, 1, 2, 3)
                return feat_map
            else:
                # After token selection, return mean-pooled representation
                # replicated across time
                pooled = patch_tokens.mean(dim=1, keepdim=True)  # (B, 1, C)
                pooled = pooled.permute(0, 2, 1)  # (B, C, 1)
                return pooled.expand(-1, -1, T).unsqueeze(-1).unsqueeze(-1)

        # Pool spatial dimension per frame
        if N_remaining == T * L:
            # No token selection - can reshape cleanly
            patch_tokens = patch_tokens.reshape(B, T, L, C).mean(dim=2)  # (B, T, C)
        else:
            # Token selection occurred - use idx to pool per frame
            # idx contains indices in range [0, T*H*W), frame = idx // (H*W)
            L_per_frame = L  # Original patches per frame
            frame_indices = idx // L_per_frame  # (B, N_remaining)

            # Pool per frame using scatter
            patch_tokens_out = torch.zeros(B, T, C, device=x.device, dtype=x.dtype)
            counts = torch.zeros(B, T, 1, device=x.device, dtype=x.dtype)

            for b in range(B):
                for t in range(T):
                    mask = frame_indices[b] == t
                    if mask.sum() > 0:
                        patch_tokens_out[b, t] = patch_tokens[b, mask].mean(dim=0)
                        counts[b, t] = 1

            # Handle frames with no remaining tokens (use mean of all)
            mean_patch = patch_tokens.mean(dim=1)  # (B, C)
            for b in range(B):
                for t in range(T):
                    if counts[b, t] == 0:
                        patch_tokens_out[b, t] = mean_patch[b]

            patch_tokens = patch_tokens_out  # (B, T, C)

        # Permute to (B, C, T)
        patch_tokens = patch_tokens.permute(0, 2, 1)  # (B, C, T)

        # Concatenate CLS token: (B, C, T+1)
        cls_token_out = cls_token_out.permute(0, 2, 1)  # (B, C, 1)
        x = torch.cat([cls_token_out, patch_tokens], dim=2)  # (B, C, T+1)

        return x, x_heatmap

    def forward_projected(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with CLIP projection (for retrieval/alignment tasks).

        Args:
            x: Input tensor of shape (B, C, T, H, W)

        Returns:
            CLIP embedding of shape (B, clip_embed_dim)
        """
        print(
            f"[VitSparseAdapter] Input dtype: {x.dtype}, "
            f"Autocast enabled: {torch.is_autocast_enabled()}, "
            f"Autocast dtype: {torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else 'N/A'}"
        )

        x = self.patch_embed(x)
        print(f"[VitSparseAdapter] After patch_embed dtype: {x.dtype}")

        B, T, L, C = x.shape
        x = x.view([B, T * L, C])

        # Compute spatial dimensions for adapters
        h_patch = w_patch = int(L**0.5)
        total_num_patches = T * L

        # Initialize token indices (all patches, no pruning in InternVideo)
        idx = (
            torch.arange(total_num_patches, device=x.device).unsqueeze(0).expand(B, -1)
        )

        # Compute neighbor indices for sparse_conv adapters
        if self.needs_neighbor_indices:
            neighbor_indices = self.get_initial_neighbor_indices(
                h_patch, w_patch, T, x.device, B
            )
        else:
            neighbor_indices = None

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        if self.sep_pos_embed:
            pos_embed = self.pos_embed_spatial.repeat(
                1, self.grid_size[0], 1
            ) + torch.repeat_interleave(
                self.pos_embed_temporal,
                self.grid_size[1] * self.grid_size[2],
                dim=1,
            )
            pos_embed = torch.cat(
                [
                    self.pos_embed_cls.expand(pos_embed.shape[0], -1, -1),
                    pos_embed,
                ],
                1,
            )
        else:
            pos_embed = self.pos_embed

        target_shape = x[0].shape
        if self.pos_embed[0].shape != target_shape:
            pos_embed = self.expand_pos_embed(self.pos_embed, T, L)

        x = x + pos_embed

        # Append heatmap tokens after CLS tokens if landmarks enabled
        # (after pos_embed addition since heatmap tokens don't need positional encoding)
        if self.n_landmarks > 0:
            heatmap_tokens = self.heatmap_tokens.expand(B, -1, -1)
            x = torch.cat(
                (
                    x[:, : self.cls_token_num, :],
                    heatmap_tokens,
                    x[:, self.cls_token_num :, :],
                ),
                dim=1,
            )

        # Apply transformer blocks with adapter spatial info
        residual = None
        for blk in self.blocks:
            if isinstance(x, tuple) and len(x) == 2:
                x, residual = x
            result = blk(
                x,
                residual=residual,
                idx=idx,
                h=h_patch,
                w=w_patch,
                total_num_patches=total_num_patches,
                neighbor_indices=neighbor_indices,
            )
            # Unpack block result (x, idx, neighbor_indices)
            x, idx, neighbor_indices = result
        if isinstance(x, tuple) and len(x) == 2:
            x, residual = x
            if residual is not None:
                x = x + residual

        return self.clip_projector(x)


# =============================================================================
# Model Constructors
# =============================================================================
def internvideo_next_base_patch14_224(pretrained=None, **kwargs):
    """InternVideoNext Base model with patch size 14 and input size 224."""
    model = InternVideoNextBackbone(
        img_size=224,
        patch_size=14,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        attn_pool_num_heads=16,
        clip_embed_dim=768,
        pretrained=pretrained,
        **kwargs,
    )
    return model


def internvideo_next_large_patch14_224(pretrained=None, **kwargs):
    """InternVideoNext Large model with patch size 14 and input size 224."""
    model = InternVideoNextBackbone(
        img_size=224,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        attn_pool_num_heads=16,
        clip_embed_dim=768,
        pretrained=pretrained,
        **kwargs,
    )
    return model
