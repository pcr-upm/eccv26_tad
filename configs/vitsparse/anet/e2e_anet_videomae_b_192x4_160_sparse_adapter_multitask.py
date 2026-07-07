_base_ = ["e2e_anet_videomae_s_192x4_160_sparse_adapter_multitask.py"]

model = dict(
    backbone=dict(
        backbone=dict(
            embed_dims=768,
            depth=12,
            num_heads=12,
        ),
        custom=dict(pretrain="pretrained/videomaev2_base_converted.pth"),
    ),
    projection=dict(in_channels=768),
)

work_dir = "exps/anet/adatad/e2e_actionformer_videomae_b_192x4_160_sparse_adapter_multitask"
