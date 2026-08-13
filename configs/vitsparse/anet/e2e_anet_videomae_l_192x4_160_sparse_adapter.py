_base_ = ["e2e_anet_videomae_s_192x4_160_sparse_adapter.py"]

model = dict(
    backbone=dict(
        backbone=dict(
            embed_dims=1024,
            depth=24,
            num_heads=16,
            adapter_index=list(range(24)),
            adapter_conv_types=["2d_conv"] * 3 + ["sparse_conv"] * 21,
            adapter_use_attn=3,
            keep_rate=0.8

        ),
        custom=dict(pretrain="pretrained/videomaev2_large_converted.pth"),
    ),
    projection=dict(in_channels=1024),
)

solver = dict(
    train=dict(
        batch_size=16,
        num_workers=16,
        persistent_workers=True,  
        prefetch_factor=8,  
        pin_memory=True,  
        multiprocessing_context="spawn", 
    ),
    val=dict(
        batch_size=16,
        num_workers=16,
        persistent_workers=False,
        prefetch_factor=8,
        multiprocessing_context="spawn",
    ),
    test=dict(
        batch_size=16,
        num_workers=16,
        persistent_workers=False,
        prefetch_factor=8,
        multiprocessing_context="spawn",
    ),
    clip_grad_norm=1,
    amp=True,
    fp16_compress=True,
    static_graph=True,
    ema=True,
)



work_dir = "exps/anet/vitsparse/e2e_actionformer_videomae_l_192x4_160_sparse_adapter"
