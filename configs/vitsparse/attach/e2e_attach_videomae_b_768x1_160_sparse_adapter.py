_base_ = ["e2e_attach_videomae_s_768x1_160_sparse_adapter.py"]

model = dict(
    backbone=dict(
        backbone=dict(
            embed_dims=768,
            depth=12,
            num_heads=12,
            keep_rate=0.6,
            keep_rate_merge=1,
            n_landmarks=32,
            adapter_use_attn=2,
        ),
        custom=dict(pretrain="pretrained/videomaev2_base_converted.pth"),
    ),
    projection=dict(in_channels=768),
)

optimizer = dict(
    backbone=dict(
        custom=[
            dict(name="adapter", lr=3e-05, weight_decay=0.01),
            dict(name="cls_token", lr=2e-05, weight_decay=0.01),
            dict(name="heatmap_tokens", lr=3e-05, weight_decay=0.01),
            dict(name="heatmap_head", lr=3e-05, weight_decay=0.01),
        ],
    )
)

work_dir = "exps/attach/vitsparse/e2e_actionformer_videomae_b_768x1_160_sparse_adapter"
