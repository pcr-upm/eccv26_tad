_base_ = [
    "../../_base_/datasets/attach/e2e_multitask.py",
    "../../_base_/models/actionformer.py",
]
custom_imports = dict(
    imports=["tools.transforms.vid_transforms"],
    allow_failed_imports=False,
)

video_decode_threads = 4
video_backend = "decord_cpu"  # options: 'decord_gpu', 'decord_cpu', 'opencv'

try:
    import decord
except ImportError:
    pass

_video_backend_registry = {
    "decord_gpu": dict(
        init=dict(
            type="DecordInitEfficient",
            num_threads=video_decode_threads,
            use_gpu=True,
            fallback_to_cpu=True,
        ),
        decode=dict(type="mmaction.DecordDecode"),
    ),
    "decord_cpu": dict(
        init=dict(
            type="DecordInitEfficient",
            num_threads=video_decode_threads,
            use_gpu=False,
        ),
        decode=dict(type="mmaction.DecordDecode"),
    ),
    "opencv": dict(
        init=dict(type="OpenCVInitEfficient"),
        decode=dict(type="mmaction.OpenCVDecode"),
    ),
}


def _get_backend_pair():
    if video_backend not in _video_backend_registry:
        raise ValueError(f"Video backend {video_backend} is not supported")
    return (
        _video_backend_registry[video_backend]["init"],
        _video_backend_registry[video_backend]["decode"],
    )


_train_backend_init, _train_backend_decode = _get_backend_pair()
_val_backend_init, _val_backend_decode = _get_backend_pair()
_test_backend_init, _test_backend_decode = _get_backend_pair()
window_size = 384
scale_factor = 1
chunk_num = (
    window_size * scale_factor // 16
)  # 768/16=48 chunks, since videomae takes 16 frames as input
max_skeleton_len = window_size  # Corresponds to window_size * feature_stride

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=False,
)

dataset = dict(
    train=dict(
        pipeline=[
            _train_backend_init,
            dict(
                type="LoadFrames",
                num_clips=1,
                method="random_trunc",
                trunc_len=window_size,
                trunc_thresh=0.3,
                crop_ratio=[0.9, 1.0],
                scale_factor=scale_factor,
            ),
            _train_backend_decode,
            dict(
                type="mmaction.Augment_Keypoints_and_Imgs",
                resize_cfg=dict(scale=(-1, 256)),  # Slightly larger for better crops
                crop_cfg=dict(area_range=(0.8, 1.0), aspect_ratio_range=(0.8, 1.2)),
                last_resize_cfg=dict(
                    scale=(224, 224), keep_ratio=False
                ),  # Match backbone!
                flip_cfg=dict(flip_ratio=0.5),
            ),
            dict(
                type="mmaction.ImgAug",
                transforms=[
                    dict(
                        type="SomeOf",
                        n=(1, 4),
                        random_order=True,
                        children=[
                            dict(
                                type="Sometimes",
                                p=0.5,
                                then_list=[
                                    dict(type="Equalize"),
                                    dict(type="Autocontrast", cutoff=(0, 10)),
                                    dict(type="EnhanceColor", factor=(0.1, 1.9)),
                                    dict(type="EnhanceContrast", factor=(0.1, 1.9)),
                                    dict(type="EnhanceBrightness", factor=(0.1, 1.9)),
                                    dict(type="EnhanceSharpness", factor=(0.1, 1.9)),
                                    dict(type="Posterize", nb_bits=(4, 8)),
                                    dict(type="Solarize", p=1.0, threshold=(32, 224)),
                                    dict(type="Invert", p=1.0, per_channel=0.5),
                                    dict(type="GaussianBlur", sigma=(0.1, 2.0)),
                                    dict(
                                        type="AdditiveGaussianNoise",
                                        scale=(0, 0.05 * 255),
                                    ),
                                ],
                            )
                        ],
                    )
                ],
            ),
            dict(type="mmaction.ColorJitter"),
            dict(type="mmaction.VideoNormalize", **img_norm_cfg),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(
                type="mmaction.PadSkelSequence",
                max_len=max_skeleton_len,
                key="keypoint",
            ),
            dict(
                type="mmaction.FormatSkelShape",
                input_format="T_V_C",
                target_format="C_T_V_M",
                key="keypoint",
            ),
            dict(
                type="ConvertToTensor",
                keys=["imgs", "keypoint", "gt_segments", "gt_labels"],
            ),
            dict(
                type="Collect",
                inputs="imgs",
                keys=["masks", "gt_segments", "gt_labels", "keypoint"],
            ),
        ],
    ),
    val=dict(
        window_size=window_size,
        pipeline=[
            _val_backend_init,
            dict(
                type="LoadFrames",
                num_clips=1,
                method="sliding_window",
                scale_factor=scale_factor,
            ),
            _val_backend_decode,
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
            _test_backend_init,
            dict(
                type="LoadFrames",
                num_clips=1,
                method="sliding_window",
                scale_factor=scale_factor,
            ),
            _test_backend_decode,
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
)

model = dict(
    backbone=dict(
        type="mmaction.Recognizer3D",
        backbone=dict(
            type="VisionTransformerSparseAdapterPOGUISE",
            img_size=224,
            patch_size=16,
            embed_dims=384,
            depth=12,
            num_heads=6,
            mlp_ratio=4,
            qkv_bias=True,
            num_frames=16,
            drop_path_rate=0.1,
            norm_cfg=dict(type="LN", eps=1e-5),
            return_feat_map=False,
            with_cp=True,
            total_frames=window_size * scale_factor,
            adapter_index=list(range(12)),
            keep_rate=0.4,
            adapter_conv_types=["2d_conv"] * 4 + ["sparse_conv"] * 8,
            adapter_use_attn=0,
            n_landmarks=0,
        ),
        data_preprocessor=dict(
            type="mmaction.ActionDataPreprocessor",
            format_shape="NCTHW",
        ),
        custom=dict(
            pretrain="pretrained/videomaev2_small_converted.pth",
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
            norm_eval=False,
            freeze_backbone=False,
        ),
    ),
    projection=dict(
        in_channels=384,
        max_seq_len=window_size,
        attn_cfg=dict(n_mha_win_size=-1),
    ),
    rpn_head=dict(
        num_classes=51,
        prior_generator=dict(
            strides=[1, 2, 4, 8, 16, 32],
            regression_range=[
                (0, 4),      # Level 0: Ultra-fast clicks/releases
                (4, 16),     # Level 1: < 0.5s
                (16, 32),    # Level 2: 0.5s - 1s
                (32, 64),    # Level 3: 1s - 2s (Median range)
                (64, 128),   # Level 4: 2s - 4s (Mean range)
                (128, 10000) # Level 5: Everything else
            ],
        ),
        label_smoothing=0.1,
        loss_normalizer=200,  # Adjust based on your average actions per window
        loss_weight=1.0,
        
    ),
)

solver = dict(
    train=dict(
        batch_size=4,
        num_workers=4,
        persistent_workers=True,
        prefetch_factor=8,
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
            dict(name="cls_token", lr=2e-4, weight_decay=0.05),
            dict(name="heatmap_tokens", lr=2e-4, weight_decay=0.01),
            dict(name="heatmap_head", lr=2e-4, weight_decay=0.01),
        ],
        exclude=["backbone"],
    ),
)
scheduler = dict(type="LinearWarmupCosineAnnealingLR", warmup_epoch=15, max_epoch=250)

inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)
post_processing = dict(
    pre_nms_topk=5000,     # Increase from default (usually 2000)
    nms=dict(
        use_soft_nms=True,
        sigma=0.3,         # Lower sigma is "gentler" for high overlap
        max_seg_num=1000,   # High density needs more slots for final predictions
        min_score=0.001,
        multiclass=True,
        voting_thresh=0.85, # Helps refine boundaries by averaging overlapping hits
    ),
    save_dict=False,

)

workflow = dict(
    logging_interval=50,
    checkpoint_interval=2,
    val_loss_interval=10,
    val_eval_interval=15,
    val_start_epoch=1,
    end_epoch=250,
    early_stopping_patience=10,
)

work_dir = "exps/attach/vitsparse/e2e_actionformer_videomae_s_768x1_160_sparse_adapter"
