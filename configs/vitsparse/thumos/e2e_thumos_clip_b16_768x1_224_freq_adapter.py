_base_ = [
    "../../_base_/datasets/thumos-14/e2e_train_trunc_test_sw_256x224x224.py",  # dataset config
    "../../_base_/models/actionformer.py",  # model config
]
custom_imports = dict(
    imports=["tools.transforms.vid_transforms"], allow_failed_imports=False
)
window_size = 768
scale_factor = 1
chunk_num = (
    window_size * scale_factor // 16
)  # 768/16=48 chunks, the CLIP backbone takes 16-frame chunks (one "T" group)

# CLIP's own normalization stats (mean/std in 0-255 range), required to match
# the pretrained CLIP visual encoder.
img_norm_cfg = dict(
    mean=[122.771, 116.746, 104.094],
    std=[68.500, 66.632, 70.323],
    to_rgb=False,
)
n_rand_aug = 2
m_rand_aug = 9
dataset = dict(
    train=dict(
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(
                type="LoadFrames",
                num_clips=1,
                method="random_trunc",
                trunc_len=window_size,
                trunc_thresh=0.75,
                crop_ratio=[0.9, 1.0],
                scale_factor=scale_factor,
            ),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 224)),
            dict(type="mmaction.RandomResizedCrop"),
            dict(type="mmaction.Resize", scale=(224, 224), keep_ratio=False),
            dict(type="mmaction.Flip", flip_ratio=0.5),
            dict(
                type="mmaction.ImgAug",
                transforms=[dict(type="RandAugment", n=n_rand_aug, m=m_rand_aug)],
            ),
            dict(type="mmaction.ColorJitter"),
            dict(type="mmaction.VideoNormalize", **img_norm_cfg),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs", "gt_segments", "gt_labels"]),
            dict(
                type="Collect",
                inputs="imgs",
                keys=["masks", "gt_segments", "gt_labels"],
            ),
        ],
    ),
    val=dict(
        window_size=window_size,
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(
                type="LoadFrames",
                num_clips=1,
                method="sliding_window",
                scale_factor=scale_factor,
            ),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 224)),
            dict(type="mmaction.CenterCrop", crop_size=224),
            dict(type="mmaction.VideoNormalize", **img_norm_cfg),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs", "gt_segments", "gt_labels"]),
            dict(
                type="Collect",
                inputs="imgs",
                keys=["masks", "gt_segments", "gt_labels"],
            ),
        ],
    ),
    test=dict(
        window_size=window_size,
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(
                type="LoadFrames",
                num_clips=1,
                method="sliding_window",
                scale_factor=scale_factor,
            ),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 224)),
            dict(type="mmaction.CenterCrop", crop_size=224),
            dict(type="mmaction.VideoNormalize", **img_norm_cfg),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs"]),
            dict(type="Collect", inputs="imgs", keys=["masks"]),
        ],
    ),
)


model = dict(
    backbone=dict(
        type="mmaction.Recognizer3D",
        backbone=dict(
            type="VisionTransformerCLIPFreqAdapter",
            input_resolution=224,
            patch_size=16,
            embed_dims=768,
            depth=12,
            num_heads=12,
            num_frames=16,
            with_cp=True,  # enable activation checkpointing
            # Frame2Freq-style temporal + multi-resolution frequency (FFT)
            # adapter (https://github.com/th-nesh/Frame2Freq), inserted before
            # the MLP of every transformer block.
            adapter_width=384,
            adapter_kernel_size=(3, 1, 1),
            adapter_pre_attn=False,
            adapter_pre_mlp=True,
        ),
        data_preprocessor=dict(
            type="mmaction.ActionDataPreprocessor",
            # mean=[123.675, 116.28, 103.53],
            # std=[58.395, 57.12, 57.375],
            format_shape="NCTHW",
        ),
        custom=dict(
            pretrain="pretrained/clip-base-16.pth",
            pre_processing_pipeline=[
                dict(
                    type="Rearrange",
                    keys=["frames"],
                    ops="b n c (t1 t) h w -> (b t1) n c t h w",
                    t1=chunk_num,
                ),
            ],
            post_processing_pipeline=[
                dict(
                    type="Reduce",
                    keys=["feats"],
                    ops="b n t c -> b t c",
                    reduction="mean",
                ),
                dict(
                    type="Rearrange",
                    keys=["feats"],
                    ops="(b t1) c t -> b c (t1 t)",
                    t1=chunk_num,
                ),
                dict(type="Interpolate", keys=["feats"], size=window_size),
            ],
            norm_eval=False,  # also update the norm layers
            freeze_backbone=False,  # unfreeze the backbone
        ),
    ),
    projection=dict(
        in_channels=768,
        max_seq_len=window_size,
        attn_cfg=dict(n_mha_win_size=-1),
    ),
)

solver = dict(
    train=dict(
        batch_size=1,
        num_workers=8,
        persistent_workers=True,
        prefetch_factor=4,
        pin_memory=True,
        multiprocessing_context="spawn",
    ),
    val=dict(
        batch_size=4,
        num_workers=2,
        persistent_workers=False,
        multiprocessing_context="spawn",
    ),
    test=dict(
        batch_size=4,
        num_workers=2,
        persistent_workers=False,
        multiprocessing_context="spawn",
    ),
    clip_grad_norm=1,
    amp=True,
    fp16_compress=True,
    static_graph=True,
    ema=True,
)

optimizer = dict(
    type="AdamW",
    lr=1e-4,
    weight_decay=0.05,
    paramwise=True,
    backbone=dict(
        lr=0,
        weight_decay=0,
        custom=[
            dict(name="adapter", lr=2e-4, weight_decay=0.05),
        ],
        exclude=["backbone"],
    ),
)
scheduler = dict(type="LinearWarmupCosineAnnealingLR", warmup_epoch=5, max_epoch=100)

inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)
post_processing = dict(
    nms=dict(
        use_soft_nms=True,
        sigma=0.7,
        max_seg_num=2000,
        multiclass=True,
        voting_thresh=0.7,  #  set 0 to disable
    ),
    save_dict=False,
)


workflow = dict(
    logging_interval=50,
    checkpoint_interval=2,
    val_loss_interval=10,
    val_eval_interval=15,
    val_start_epoch=1,
    end_epoch=150,
    early_stopping_patience=5,
)

work_dir = "exps/thumos/vitsparse/e2e_actionformer_clip_b16_768x1_224_freq_adapter"
