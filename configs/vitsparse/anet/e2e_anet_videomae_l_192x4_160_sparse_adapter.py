_base_ = ["e2e_anet_videomae_s_192x4_160_sparse_adapter.py"]

model = dict(
    backbone=dict(
        backbone=dict(
            embed_dims=1024,
            depth=24,
            num_heads=16,
            adapter_index=list(range(24)),
            adapter_conv_types=["2d_conv"] * 8 + ["sparse_conv"] * 16,
        ),
        custom=dict(pretrain="pretrained/videomaev2_large_converted.pth"),
    ),
    projection=dict(in_channels=1024),
)

work_dir = "exps/anet/adatad/e2e_actionformer_videomae_l_192x4_160_sparse_adapter"
