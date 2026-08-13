_base_ = ["e2e_thumos_videomae_s_768x1_160_sparse_adapter.py"]

model = dict(
    backbone=dict(
        backbone=dict(
            embed_dims=1024,
            depth=24,
            num_heads=16,
            adapter_index=list(range(24)),
            keep_rate=0.6,
            adapter_use_attn=1,
            adapter_conv_types=["2d_conv"] * 3 + ["sparse_conv"] * 21,
        ),
        custom=dict(pretrain="pretrained/videomaev2_large_converted.pth"),
    ),
    projection=dict(in_channels=1024),
)

optimizer = dict(
    backbone=dict(
        custom=[
            dict(name="adapter", lr=2e-4, weight_decay=0.01),
            dict(name="cls_token", lr=2e-4, weight_decay=0.01),
        ],
    )
)

work_dir = "exps/thumos/vitsparse/e2e_actionformer_videomae_l_768x1_160_adapter"
