"""
VisionTransformerTRAMPOGUISE — POGUISE backbone with TRAM-style centrality-based
token selection instead of KTP attention-topk selection.

TRAM computes graph centrality from attention matrices and accumulates it across
pruning layers. Tokens are selected via topk on accumulated centrality scores.

Reference: https://github.com/DavideTraini/TRAM
"""

import math
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks import DropPath
from mmcv.cnn.bricks.transformer import FFN, PatchEmbed
from mmengine.model import BaseModule, ModuleList
from mmengine.model.weight_init import trunc_normal_init
from mmengine.registry import MODELS

from mmaction.models.backbones.vit_mae import get_sinusoid_encoding
from mmaction.utils import ConfigType, OptConfigType

# Reuse shared components from the original POGUISE backbone
from .vit_adapter_poguise import (
    EfficientAdapter,
    HeatmapHead,
    sim_matrixv2_batch,
)


class TRAMAttention(BaseModule):
    """
    Multi-head attention with TRAM-style centrality-based token selection.

    At pruning layers (keep_rate < 1), instead of using raw attention scores
    for topk selection (KTP), this module:
    1. Takes the max across attention heads to build an adjacency matrix.
    2. Computes in-degree centrality from the adjacency matrix.
    3. Rescales the adjacency matrix by centrality and re-computes centrality.
    4. Accumulates centrality across layers with linearly increasing weight.
    5. Selects tokens via topk on accumulated centrality.

    At non-pruning layers (keep_rate >= 1), uses flash attention for speed
    and passes centrality through unchanged.
    """

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
        self.n_key_tokens = n_key_tokens

    def forward(
        self,
        x: torch.Tensor,
        last_idx: Optional[torch.Tensor] = None,
        centrality_prev: Optional[torch.Tensor] = None,
        layer_idx: int = 0,
        depth: int = 12,
    ):
        B, N, C = x.shape

        # Build QKV with bias
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
            # No pruning — use flash attention for speed
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
            x = x.transpose(1, 2).reshape(B, N, -1)
            x = self.proj(x)
            x = self.proj_drop(x)
            feature_for_pruning = k
            return x, last_idx, feature_for_pruning, centrality_prev

        # Pruning layer — compute full attention matrix
        attn = (q * self.scale) @ k.transpose(-2, -1)  # (B, H, N, N)
        attn = attn.softmax(dim=-1)

        attn_for_output = self.attn_drop(attn)
        x = (attn_for_output @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        feature_for_pruning = k

        # --- TRAM centrality-based token selection ---
        num_s_tokens = self.n_key_tokens
        num_keep_tokens = math.ceil(self.keep_rate * (N - num_s_tokens))

        if num_keep_tokens <= 0:
            return (
                x,
                torch.empty(B, 0, dtype=last_idx.dtype, device=x.device),
                feature_for_pruning,
                centrality_prev,
            )

        # Take max across heads to build adjacency matrix (TRAM style)
        att_matrix = attn.max(dim=1)[0]  # (B, N, N)

        # Extract patch-to-patch portion (exclude special tokens)
        att_tokens = att_matrix[
            :, num_s_tokens:, num_s_tokens:
        ]  # (B, N_patch, N_patch)

        # Compute in-degree centrality
        centrality_in = att_tokens.sum(dim=1)  # (B, N_patch)

        # Rescale adjacency matrix by centrality and recompute
        matrices_rescaled = att_tokens * centrality_in.unsqueeze(
            -1
        )  # (B, N_patch, N_patch)
        centrality_rescaled = matrices_rescaled.sum(dim=1)  # (B, N_patch)

        # Accumulate centrality with linearly increasing weight
        weight = (layer_idx + 1) / depth
        if centrality_prev is None or (
            isinstance(centrality_prev, (int, float)) and centrality_prev == 0
        ):
            centrality = weight * centrality_rescaled
        else:
            centrality = weight * centrality_rescaled + centrality_prev

        # Select topk tokens based on accumulated centrality
        num_keep_tokens = min(num_keep_tokens, centrality.shape[1])
        _, idx_topk = torch.topk(centrality, num_keep_tokens, dim=1, largest=True)
        idx = idx_topk.sort(dim=1)[0]

        # Update centrality to only keep selected tokens' scores
        centrality_kept = torch.gather(centrality, 1, idx)

        return x, idx, feature_for_pruning, centrality_kept


class TRAMBlock(BaseModule):
    """
    Transformer block with TRAM centrality-based token selection, optional
    bipartite merging, and efficient adapter.
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

        self.attn = TRAMAttention(
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
            )

    def forward(
        self,
        x: torch.Tensor,
        last_idx: torch.Tensor,
        h: int,
        w: int,
        total_num_patches: int,
        centrality_prev,
        layer_idx: int,
        depth: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        def _inner_forward(x, last_idx, centrality_prev):
            B, N, C = x.shape
            x_norm = self.norm1(x)
            attn_x, idx, feature_for_merge, centrality = self.attn(
                x_norm, last_idx, centrality_prev, layer_idx, depth
            )
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
                        # Extend centrality with zeros for merged tokens so it
                        # matches the total non-key token count in the next layer.
                        centrality = torch.cat(
                            [
                                centrality,
                                centrality.new_zeros(B, x_merged.shape[1]),
                            ],
                            dim=1,
                        )
                x = torch.cat([x_key, x_nonkey_keep], dim=1)
                idx = updated_global_idx
            else:
                idx = last_idx

            x_post_mlp = self.mlp(self.norm2(x))
            x = x + self.drop_path(x_post_mlp)
            if self.use_adapter:
                x = self.adapter(x, h, w, idx, total_num_patches)
            return x, idx, centrality

        if self.with_cp and x.requires_grad:
            x, idx, centrality = cp.checkpoint(
                _inner_forward, x, last_idx, centrality_prev, use_reentrant=False
            )
        else:
            x, idx, centrality = _inner_forward(x, last_idx, centrality_prev)
        return x, idx, centrality

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
class VisionTransformerTRAMPOGUISE(BaseModule):
    """
    Vision Transformer with TRAM centrality-based token selection and
    integrated adapters (POGUISE framework).
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
        adapter_conv_types=None,
        deform_conv_groups: int = 1,
        init_cfg: Optional[Union[Dict, List[Dict]]] = [
            dict(type="TruncNormal", layer="Linear", std=0.02, bias=0.0),
            dict(type="Constant", layer="LayerNorm", val=1.0, bias=0.0),
        ],
        return_feat_map: bool = False,
        n_landmarks: int = 0,
        hw_out_conv: tuple = (10, 10),
        **kwargs,
    ) -> None:
        if pretrained:
            self.init_cfg = dict(type="Pretrained", checkpoint=pretrained)
        super().__init__(init_cfg=init_cfg)

        self.with_cp = with_cp
        self.embed_dims = embed_dims
        self.patch_size = patch_size
        self.depth = depth
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

        if adapter_conv_types is None:
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

        num_special_tokens = 1  # [CLS] token
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
                TRAMBlock(
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
            x = torch.cat((x[:, :1, :], heatmap_tokens, x[:, 1:, :]), dim=1)

        idx = (
            torch.arange(0, current_total_patches, device=x.device)
            .unsqueeze(0)
            .repeat(b, 1)
        )

        # Initialize TRAM centrality accumulator
        centrality = 0

        for layer_idx, blk in enumerate(self.blocks):
            x, idx, centrality = blk(
                x,
                idx,
                h_patch,
                w_patch,
                current_total_patches,
                centrality,
                layer_idx,
                self.depth,
            )

        x_heatmap = None
        if self.n_landmarks > 0:
            start_idx = 1
            end_idx = 1 + self.n_heatmap_tokens
            heatmap_tokens_out = x[:, start_idx:end_idx, :]
            heatmap_feats = heatmap_tokens_out.reshape(
                b, *self.hw_out_conv, self.embed_dims
            )
            heatmap_feats = heatmap_feats.permute(0, 3, 1, 2)
            x_heatmap = self.heatmap_head(heatmap_feats)

        x = self.norm(x)

        num_special_tokens = 1 + self.n_heatmap_tokens

        final_patch_tokens = x[:, num_special_tokens:, :]
        final_patch_indices = idx

        T_out = self.num_frames // self.tubelet_size
        total_patches_in_volume = T_out * h_patch * w_patch

        scattered_features = torch.zeros(
            b,
            total_patches_in_volume,
            self.embed_dims,
            device=final_patch_tokens.device,
            dtype=final_patch_tokens.dtype,
        )
        scattered_mask = torch.zeros(
            b,
            total_patches_in_volume,
            1,
            device=final_patch_tokens.device,
            dtype=final_patch_tokens.dtype,
        )

        if final_patch_tokens.shape[1] > 0:
            final_patch_indices_expanded = final_patch_indices.unsqueeze(-1).expand(
                -1, -1, self.embed_dims
            )
            final_patch_indices_mask = final_patch_indices.unsqueeze(-1)

            scattered_features.scatter_(
                1, final_patch_indices_expanded, final_patch_tokens
            )
            scattered_mask.scatter_(
                1, final_patch_indices_mask, torch.ones_like(scattered_mask)
            )

        reshaped_features = scattered_features.reshape(
            b, T_out, h_patch, w_patch, self.embed_dims
        )
        reshaped_mask = scattered_mask.reshape(b, T_out, h_patch, w_patch, 1)

        feature_sum_per_frame = reshaped_features.sum(dim=(2, 3))
        kept_token_count_per_frame = reshaped_mask.sum(dim=(2, 3))

        epsilon = 1e-6
        pooled_features = feature_sum_per_frame / (kept_token_count_per_frame + epsilon)

        pooled_features = pooled_features.permute(0, 2, 1)
        cls_token_final = x[:, :1, :].permute(0, 2, 1)
        x = torch.cat((cls_token_final, pooled_features), dim=2)

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
