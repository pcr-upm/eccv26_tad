import math
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks import DropPath
from mmcv.cnn.bricks.transformer import FFN, PatchEmbed

# Import DeformConv2d
from torchvision.ops import DeformConv2d
from torch import Tensor
from mmengine.model import BaseModule, ModuleList
from mmengine.model.weight_init import constant_init, kaiming_init, trunc_normal_init
from mmengine.registry import MODELS

from mmaction.models.backbones.vit_mae import get_sinusoid_encoding
from mmaction.utils import ConfigType, OptConfigType


class Adapter(BaseModule):
    def __init__(
        self,
        embed_dims: int,
        mlp_ratio: float = 0.25,
        kernel_size: int = 3,
        dilation: int = 1,
        temporal_size: int = 384,
        # new parameters for deformable conv
        conv_type: str = "temporal_dwconv",
        temporal_kernel_size: int = 3,
        deformable_groups: int = 1,
    ) -> None:
        super().__init__()

        hidden_dims = int(embed_dims * mlp_ratio)
        self.hidden_dims = hidden_dims
        self.conv_type = conv_type

        # adapter projection
        self.down_proj = nn.Linear(embed_dims, hidden_dims)
        self.act = nn.GELU()
        self.up_proj = nn.Linear(hidden_dims, embed_dims)
        self.gamma = nn.Parameter(torch.ones(1))
        trunc_normal_init(self.down_proj, std=0.02, bias=0)
        constant_init(self.up_proj, 0)

        if self.conv_type == "temporal_dwconv":
            # original temporal depth-wise convolution
            self.temporal_size = temporal_size
            self.dwconv = nn.Conv1d(
                hidden_dims,
                hidden_dims,
                kernel_size=kernel_size,
                stride=1,
                padding=(kernel_size // 2) * dilation,
                dilation=dilation,
                groups=hidden_dims,
            )
            self.conv = nn.Conv1d(hidden_dims, hidden_dims, 1)
            self.dwconv.weight.data.normal_(mean=0.0, std=math.sqrt(2.0 / kernel_size))
            self.dwconv.bias.data.zero_()
            self.conv.weight.data.normal_(mean=0.0, std=math.sqrt(2.0 / hidden_dims))
            self.conv.bias.data.zero_()

        elif self.conv_type == "deformable_conv_t":
            print(
                f"Using TEMPORAL Deformable Convolution ((2+1)D) with {deformable_groups} groups"
            )
            assert (
                hidden_dims % deformable_groups == 0
            ), "hidden_dims must be divisible by deformable_groups."
            self.temporal_size = temporal_size

            # 1. Spatial Deformable Convolution
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

            # 2. Temporal 1D Convolution
            self.temporal_conv = nn.Conv1d(
                hidden_dims,
                hidden_dims,
                kernel_size=temporal_kernel_size,
                padding=(temporal_kernel_size // 2) * dilation,
                dilation=dilation,
                groups=hidden_dims,  # Using a depthwise convolution for efficiency
            )
            self.temporal_pw_conv = nn.Conv1d(hidden_dims, hidden_dims, 1)  # Pointwise

            # Weight initialization for deformable_conv_t
            constant_init(self.offset_conv, 0)
            constant_init(self.mask_conv, 0)
            kaiming_init(self.deform_conv, mode="fan_in", nonlinearity="relu")
            self.temporal_conv.weight.data.normal_(
                mean=0.0, std=math.sqrt(2.0 / self.temporal_conv.kernel_size[0])
            )
            self.temporal_conv.bias.data.zero_()
            self.temporal_pw_conv.weight.data.normal_(
                mean=0.0, std=math.sqrt(2.0 / self.hidden_dims)
            )
            self.temporal_pw_conv.bias.data.zero_()

        elif self.conv_type == "adatad_plus_plus":
            print("Using AdaTAD++ (Transformer-Enhanced) Adapter")
            self.temporal_size = temporal_size

            # 1. Spatial 2D Depthwise Convolution
            self.spatial_conv = nn.Conv2d(
                hidden_dims,
                hidden_dims,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=hidden_dims,
            )

            # 2. Spatial Transformer Encoder (TransEnc1)
            self.spatial_pool = nn.AvgPool2d(2, stride=2)
            self.spatial_attn = Attention(
                hidden_dims,
                num_heads=max(1, hidden_dims // 64),
                qkv_bias=True,
            )
            self.spatial_norm1 = nn.LayerNorm(hidden_dims)
            self.spatial_fc = nn.Linear(hidden_dims, hidden_dims)
            self.act_spa = nn.GELU()

            # 3. Temporal Transformer Encoder (TransEnc2)
            self.temporal_attn = Attention(
                hidden_dims,
                num_heads=max(1, hidden_dims // 64),  # Auto-scaling heads
                qkv_bias=True,
            )
            self.temporal_norm = nn.LayerNorm(hidden_dims)
            self.temporal_fc = nn.Linear(hidden_dims, hidden_dims)
            self.act_temp = nn.GELU()

            # Init
            constant_init(self.spatial_conv, 0)
            trunc_normal_init(self.spatial_fc, std=0.02, bias=0)
            trunc_normal_init(self.temporal_fc, std=0.02, bias=0)

        else:
            raise ValueError(f"Unknown conv_type: '{self.conv_type}' in Adapter")

    def forward(self, x: Tensor, h: int, w: int) -> Tensor:
        inputs = x
        B, N, C = x.shape

        # down projection
        x_proj = self.down_proj(x)
        x_proj = self.act(x_proj)

        if self.conv_type == "temporal_dwconv":
            # temporal depth-wise convolution
            attn = x_proj.reshape(-1, self.temporal_size, h, w, x_proj.shape[-1])
            attn = attn.permute(0, 2, 3, 4, 1).flatten(0, 2)
            attn = self.dwconv(attn)
            attn = self.conv(attn)
            attn = attn.unflatten(0, (-1, h, w)).permute(0, 4, 1, 2, 3)
            attn = attn.reshape(B, N, self.hidden_dims)
            x = x_proj + attn

        elif self.conv_type == "deformable_conv_t":
            # Reshape using the fixed temporal window, similar to the original adapter.
            # This assumes the input's total tokens are divisible by the window's total tokens.
            x_5d = x_proj.reshape(-1, self.temporal_size, h, w, self.hidden_dims)
            b_new = x_5d.shape[0]

            # 1. Spatial Deformable Convolution
            # Reshape for spatial conv by combining the new batch and temporal dims.
            conv_input_2d = (
                x_5d.reshape(b_new * self.temporal_size, h, w, self.hidden_dims)
                .permute(0, 3, 1, 2)
                .contiguous()
            )

            offset = self.offset_conv(conv_input_2d)
            modulation_mask = torch.sigmoid(self.mask_conv(conv_input_2d))
            spatial_out = self.deform_conv(conv_input_2d, offset, mask=modulation_mask)

            # 2. Temporal 1D Convolution
            # Reshape back to 5D to prepare for temporal convolution.
            temporal_input = spatial_out.permute(0, 2, 3, 1).view(
                b_new, self.temporal_size, h, w, self.hidden_dims
            )
            # Permute and flatten for temporal conv by combining batch and spatial dims.
            temporal_input = temporal_input.permute(0, 2, 3, 4, 1).flatten(0, 2)

            temporal_out = self.temporal_pw_conv(self.temporal_conv(temporal_input))

            # 3. Reshape back to the original sequence format (B, N, C)
            processed_full_sequence = temporal_out.unflatten(0, (b_new, h, w))
            x = processed_full_sequence.permute(0, 4, 1, 2, 3).reshape(
                B, N, self.hidden_dims
            )

        elif self.conv_type == "adatad_plus_plus":
            # Using TIA++ logic
            # Reshape [B, N, C] -> [B, T, H, W, C]
            x_5d = x_proj.reshape(-1, self.temporal_size, h, w, self.hidden_dims)
            B_new, T, H, W, C_hid = x_5d.shape

            # 1. Spatial Conv (per frame)
            # Input: [B*T, C, H, W]
            spatial_input = x_5d.view(B_new * T, H, W, C_hid).permute(0, 3, 1, 2)
            # Eq. 5: Fd' = 2DCNN(Fd)
            spatial_out = self.spatial_conv(spatial_input)  # [B*T, C, H, W]

            # 2. Spatial Attention (TransEnc1)
            # Eq. 6: Fd'' = Fd' * TransEnc1(AveragePool2x2(Fd'))
            # Spatial pooling (2x2)
            spatial_avg = self.spatial_pool(spatial_out)  # [B*T, C, H/2, W/2]
            H_small, W_small = spatial_avg.shape[2], spatial_avg.shape[3]

            # Flatten for Transformer: [B*T, C, H/2, W/2] -> [B*T, H/2*W/2, C]
            spatial_tokens = spatial_avg.flatten(2).permute(0, 2, 1)

            # Apply TransEnc1
            spa_attn_out = self.spatial_attn(spatial_tokens)
            spa_attn_out = self.spatial_norm1(spa_attn_out)
            spa_attn_out = self.act_spa(
                self.spatial_fc(spa_attn_out)
            )  # [B*T, N_small, C]

            # Reshape and Upsample back to H, W
            spa_attn_map = spa_attn_out.permute(0, 2, 1).view(
                -1, C_hid, H_small, W_small
            )
            # F.interpolate 'bilinear' requires float32. Cast if needed (e.g. bfloat16)
            original_dtype = spa_attn_map.dtype
            spa_attn_map = F.interpolate(
                spa_attn_map.to(torch.float32),
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            ).to(original_dtype)

            # Modulate
            spatial_modulated = spatial_out * spa_attn_map

            # 3. Global Average Pooling -> [B*T, C]
            # Eq. 7: Fd'' = 1/(H*W) * Sum(Fd') (applied on modulated features)
            spatial_pooled = spatial_modulated.mean(dim=(2, 3))

            # 4. Temporal Attention
            # Input: [B, T, C]
            temporal_input = spatial_pooled.view(B_new, T, C_hid)

            # Apply Attention
            # Eq. 8: Ftemp = TransEnc2(Fd'')
            temporal_out = self.temporal_attn(temporal_input)  # [B, T, C]
            temporal_out = self.temporal_norm(temporal_out)
            temporal_out = self.act_temp(self.temporal_fc(temporal_out))

            # 4. Broadcast back and Add
            # [B, T, C] -> [B, T, H, W, C]
            temporal_out_expanded = (
                temporal_out.unsqueeze(2).unsqueeze(3).expand(-1, -1, H, W, -1)
            )

            # Flatten to [B, N, C]
            attn = temporal_out_expanded.reshape(B, N, C_hid)

            x = x_proj + attn

        else:
            # Fallback for other potential conv_types, or if none is specified
            x = x_proj

        # up projection
        x = self.up_proj(x)
        return x * self.gamma + inputs


class PlainAdapter(BaseModule):
    def __init__(
        self,
        embed_dims: int,
        mlp_ratio: float = 0.25,
        **kwargs,
    ) -> None:
        super().__init__()

        hidden_dims = int(embed_dims * mlp_ratio)

        # adapter projection
        self.down_proj = nn.Linear(embed_dims, hidden_dims)
        self.act = nn.GELU()
        self.up_proj = nn.Linear(hidden_dims, embed_dims)
        self.gamma = nn.Parameter(torch.ones(1))
        trunc_normal_init(self.down_proj, std=0.02, bias=0)
        constant_init(self.up_proj, 0)

    def forward(self, x: Tensor, h: int, w: int) -> Tensor:
        inputs = x

        # down and up projection
        x = self.down_proj(x)
        x = self.act(x)
        x = self.up_proj(x)
        return x * self.gamma + inputs


class Attention(BaseModule):
    """Multi-head Self-attention."""

    def __init__(
        self,
        embed_dims: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop_rate: float = 0.0,
        drop_rate: float = 0.0,
        init_cfg: OptConfigType = None,
        **kwargs,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        head_embed_dims = embed_dims // num_heads

        self.scale = qk_scale or head_embed_dims**-0.5

        if qkv_bias:
            self._init_qv_bias()

        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=False)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(drop_rate)

    def _init_qv_bias(self) -> None:
        self.q_bias = nn.Parameter(torch.zeros(self.embed_dims))
        self.v_bias = nn.Parameter(torch.zeros(self.embed_dims))

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape

        if hasattr(self, "q_bias"):
            k_bias = torch.zeros_like(self.v_bias, requires_grad=False)
            qkv_bias = torch.cat((self.q_bias, k_bias, self.v_bias))
            qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        else:
            qkv = self.qkv(x)

        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
        x = x.transpose(1, 2).reshape(B, N, -1)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(BaseModule):
    """The basic block in the Vision Transformer."""

    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        mlp_ratio: int = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        act_cfg: ConfigType = dict(type="GELU"),
        norm_cfg: ConfigType = dict(type="LN", eps=1e-6),
        init_cfg: OptConfigType = None,
        with_cp: bool = False,
        use_adapter: bool = False,
        adapter_mlp_ratio: float = 0.25,
        temporal_size: int = 384,
        # new parameters for adapter
        adapter_conv_type: str = "temporal_dwconv",
        adapter_deformable_groups: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        self.with_cp = with_cp
        self.use_adapter = use_adapter
        self._adapter_time_ms = 0.0  # Accumulated adapter time in ms

        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.attn = Attention(
            embed_dims,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            drop_rate=drop_rate,
        )

        self.drop_path = nn.Identity()
        if drop_path_rate > 0.0:
            self.drop_path = DropPath(drop_path_rate)
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
            self.adapter = Adapter(
                embed_dims=embed_dims,
                kernel_size=3,
                dilation=1,
                temporal_size=temporal_size,
                mlp_ratio=adapter_mlp_ratio,
                conv_type=adapter_conv_type,
                deformable_groups=adapter_deformable_groups,
            )

    def forward(self, x: Tensor, h, w, measure_adapter_time: bool = False) -> Tensor:
        def _inner_forward(x):
            """Forward wrapper for utilizing checkpoint."""
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))

            if self.use_adapter:
                if measure_adapter_time:
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                    x = self.adapter(x, h, w)
                    end_event.record()
                    torch.cuda.synchronize()
                    self._adapter_time_ms += start_event.elapsed_time(end_event)
                else:
                    x = self.adapter(x, h, w)
            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


@MODELS.register_module()
class VisionTransformerAdapter(BaseModule):
    """Vision Transformer with Adapter and support for deformable convolutions."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dims: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: int = 4.0,
        qkv_bias: bool = True,
        qk_scale: int = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_cfg: ConfigType = dict(type="LN", eps=1e-6),
        num_frames: int = 16,
        tubelet_size: int = 2,
        use_mean_pooling: int = True,
        pretrained: Optional[str] = None,
        return_feat_map: bool = False,
        with_cp: bool = False,
        adapter_mlp_ratio: float = 0.25,
        total_frames: int = 768,
        adapter_index: list = [3, 5, 7, 11],
        # New parameters to control adapter type
        adapter_conv_type: str = "temporal_dwconv",  # Options: "temporal_dwconv", "deformable_conv_t", "adatad_plus_plus"
        adapter_deformable_groups: int = 2,
        init_cfg: Optional[Union[Dict, List[Dict]]] = [
            dict(type="TruncNormal", layer="Linear", std=0.02, bias=0.0),
            dict(type="Constant", layer="LayerNorm", val=1.0, bias=0.0),
        ],
        **kwargs,
    ) -> None:
        if pretrained:
            self.init_cfg = dict(type="Pretrained", checkpoint=pretrained)
        super().__init__(init_cfg=init_cfg)

        self.with_cp = with_cp
        self.embed_dims = embed_dims
        self.patch_size = patch_size

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

        pos_embed = get_sinusoid_encoding(num_patches, embed_dims)
        self.register_buffer("pos_embed", pos_embed)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

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
                    init_cfg=init_cfg,
                    use_adapter=i in adapter_index,
                    adapter_mlp_ratio=adapter_mlp_ratio,
                    temporal_size=total_frames // tubelet_size,
                    adapter_conv_type=adapter_conv_type,
                    adapter_deformable_groups=adapter_deformable_groups,
                )
                for i in range(depth)
            ]
        )

        if use_mean_pooling:
            self.norm = nn.Identity()
            self.fc_norm = build_norm_layer(norm_cfg, embed_dims)[1]
        else:
            self.norm = build_norm_layer(norm_cfg, embed_dims)[1]
            self.fc_norm = None

        self.return_feat_map = return_feat_map
        self._measure_adapter_time = False  # Flag to enable adapter timing

        num_vit_param = sum(
            p.numel() for name, p in self.named_parameters() if "adapter" not in name
        )
        num_adapter_param = sum(
            p.numel() for name, p in self.named_parameters() if "adapter" in name
        )
        ratio = num_adapter_param / num_vit_param * 100
        print(
            "ViT's param: {}, Adapter's params: {}, ratio: {:2.1f}%".format(
                num_vit_param, num_adapter_param, ratio
            )
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

    def forward(self, x: Tensor) -> Tensor:
        self._freeze_layers()

        b, _, _, h, w = x.shape
        h //= self.patch_size
        w //= self.patch_size
        x = self.patch_embed(x)[0]
        if (h, w) != self.grid_size:
            pos_embed = self.pos_embed.reshape(-1, *self.grid_size, self.embed_dims)
            pos_embed = pos_embed.permute(0, 3, 1, 2)
            pos_embed = F.interpolate(
                pos_embed, size=(h, w), mode="bicubic", align_corners=False
            )
            pos_embed = pos_embed.permute(0, 2, 3, 1).flatten(1, 2)
            pos_embed = pos_embed.reshape(1, -1, self.embed_dims)
        else:
            pos_embed = self.pos_embed

        x = x + pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x, h, w, measure_adapter_time=self._measure_adapter_time)

        x = self.norm(x)

        if self.return_feat_map:
            x = x.reshape(b, -1, h, w, self.embed_dims)
            x = x.permute(0, 4, 1, 2, 3)
            return x

        if self.fc_norm is not None:
            return self.fc_norm(x.mean(1))

        return x[:, 0]

    def _freeze_layers(self):
        """Prevent all the parameters not in the adapters"""
        self.patch_embed.eval()
        for m in self.patch_embed.modules():
            for param in m.parameters():
                param.requires_grad = False

        for block in self.blocks:
            for m, n in block.named_children():
                if "adapter" not in m and m != "drop_path":
                    n.eval()
                    for param in n.parameters():
                        param.requires_grad = False
