_base_ = [
    "../../_base_/datasets/activitynet-1.3/e2e_resize_768_1x224x224.py",  # dataset config
    "../../_base_/models/actionformer.py",  # model config
]

custom_imports = dict(
    imports=["tools.transforms.vid_transforms"], allow_failed_imports=False
)

resize_length = 192
scale_factor = 4
chunk_num = (
    resize_length * scale_factor // 16
)  # 768/16=48 chunks, since videomae takes 16 frames as input

# --- Keypoint Path ---
skeleton_data_path_2d = "data/activitynet-1.3/pose_results"
# Preprocessed keypoints (run: python tools/prepare_data/activitynet/preprocess_keypoints.py)
preprocessed_skeleton_path = "data/activitynet-1.3/pose_results_processed"
# ---------------------

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=False,
)

dataset = dict(
    train=dict(
        skeleton_data_path_2d=skeleton_data_path_2d,
        preprocessed_skeleton_path=preprocessed_skeleton_path,
        resize_length=resize_length,
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4", prefix="v_"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(
                type="LoadFrames",
                num_clips=1,
                method="resize",
                scale_factor=scale_factor,
            ),  # load 192x4=768 frames
            dict(type="mmaction.DecordDecode"),
            # Keypoint-aware augmentation
            dict(
                type="mmaction.Augment_Keypoints_and_Imgs",
                resize_cfg=dict(scale=(-1, 182)),
                crop_cfg=dict(area_range=(0.7, 1.0), aspect_ratio_range=(3 / 4, 4 / 3)),
                last_resize_cfg=dict(scale=(160, 160), keep_ratio=False),
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
                type="ConvertToTensor",
                keys=["imgs", "gt_segments", "gt_labels", "keypoint"],
            ),
            dict(
                type="Collect",
                inputs="imgs",
                keys=["masks", "gt_segments", "gt_labels", "keypoint"],
            ),
        ],
    ),
    val=dict(
        skeleton_data_path_2d=skeleton_data_path_2d,
        preprocessed_skeleton_path=preprocessed_skeleton_path,
        resize_length=resize_length,
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4", prefix="v_"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(
                type="LoadFrames",
                num_clips=1,
                method="resize",
                scale_factor=scale_factor,
            ),  # load 192x4=768 frames
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 160)),
            dict(type="mmaction.CenterCrop", crop_size=160),
            dict(type="mmaction.VideoNormalize", **img_norm_cfg),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(
                type="ConvertToTensor",
                keys=["imgs", "gt_segments", "gt_labels", "keypoint"],
            ),
            dict(
                type="Collect",
                inputs="imgs",
                keys=["masks", "gt_segments", "gt_labels", "keypoint"],
            ),
        ],
    ),
    test=dict(
        skeleton_data_path_2d=skeleton_data_path_2d,
        preprocessed_skeleton_path=preprocessed_skeleton_path,
        resize_length=resize_length,
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4", prefix="v_"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(
                type="LoadFrames",
                num_clips=1,
                method="resize",
                scale_factor=scale_factor,
            ),  # load 192x4=768 frames
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 160)),
            dict(type="mmaction.CenterCrop", crop_size=160),
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
            return_feat_map=True,
            with_cp=True,
            total_frames=resize_length * scale_factor,
            adapter_index=list(range(12)),
            keep_rate=0.6,
            adapter_conv_types=["2d_conv"] * 4 + ["sparse_conv"] * 8,
            adapter_use_attn=1,
            n_landmarks=26,
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
                dict(type="Interpolate", keys=["feats"], size=resize_length),
            ],
            norm_eval=False,  # also update the norm layers
            freeze_backbone=False,  # unfreeze the backbone
        ),
    ),
    projection=dict(
        in_channels=384,
        out_channels=256,
        attn_cfg=dict(n_mha_win_size=-1),
        use_abs_pe=True,
        max_seq_len=resize_length,
    ),
    neck=dict(in_channels=256, out_channels=256),
    rpn_head=dict(
        in_channels=256,
        feat_channels=256,
        num_classes=1,
        label_smoothing=0.1,
        loss_weight=2.0,
        loss_normalizer=200,
    ),
)

solver = dict(
    train=dict(
        batch_size=8,
        num_workers=8,
        persistent_workers=True,
        prefetch_factor=4,
        pin_memory=True,
        multiprocessing_context="spawn",
    ),
    val=dict(
        batch_size=8,
        num_workers=4,
        persistent_workers=False,
        multiprocessing_context="spawn",
    ),
    test=dict(
        batch_size=8,
        num_workers=4,
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
    lr=1e-3,
    weight_decay=0.05,
    paramwise=True,
    backbone=dict(
        lr=0,
        weight_decay=0,
        custom=[
            dict(name="adapter", lr=1e-4, weight_decay=0.05),
            dict(name="cls_token", lr=1e-4, weight_decay=0.05),
            dict(name="heatmap_tokens", lr=1e-4, weight_decay=0.01),
            dict(name="heatmap_head", lr=1e-4, weight_decay=0.01),
        ],
        exclude=["backbone"],
    ),
)
scheduler = dict(type="LinearWarmupCosineAnnealingLR", warmup_epoch=5, max_epoch=15)

inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)
post_processing = dict(
    nms=dict(
        use_soft_nms=True,
        sigma=0.7,
        max_seg_num=100,
        min_score=0.001,
        multiclass=False,
        voting_thresh=0.9,  # set 0 to disable
    ),
    external_cls=dict(
        type="CUHKANETClassifier",
        path="data/activitynet-1.3/classifiers/cuhk_val_simp_7.json",
        topk=2,
    ),
    save_dict=False,
)

workflow = dict(
    logging_interval=200,
    checkpoint_interval=1,
    val_loss_interval=-1,
    val_eval_interval=1,
    val_start_epoch=8,
)

work_dir = (
    "exps/anet/vitsparse/e2e_actionformer_videomae_s_192x4_160_sparse_adapter_multitask"
)
