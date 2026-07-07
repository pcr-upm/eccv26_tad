_base_ = [
    "../../_base_/datasets/thumos-14/e2e_train_trunc_test_sw_256x224x224.py",
    "../../_base_/models/actionformer.py",
]
custom_imports = dict(
    imports=["tools.transforms.vid_transforms"], allow_failed_imports=False
)
window_size = 768
scale_factor = 1
# InternVideo uses 16 frames per clip
chunk_num = window_size * scale_factor // 16  # 768/16=48 chunks

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=False,
)

# --- Keypoint Path ---
skeleton_data_path_2d = "/media/ricardo/data/datasets/THUMOS14/raw_data/pose_results"
# Preprocessed keypoints (run: python tools/prepare_data/thumos/preprocess_keypoints.py)
preprocessed_skeleton_path = (
    "/media/ricardo/data/datasets/THUMOS14/raw_data/pose_results_processed"
)
# ---------------------

dataset = dict(
    train=dict(
        skeleton_data_path_2d=skeleton_data_path_2d,
        preprocessed_skeleton_path=preprocessed_skeleton_path,
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
            # KP-aware augmentation for landmark compatibility
            dict(
                type="mmaction.Augment_Keypoints_and_Imgs",
                resize_cfg=dict(scale=(-1, 224)),
                crop_cfg=dict(area_range=(0.7, 1.0), aspect_ratio_range=(3 / 4, 4 / 3)),
                last_resize_cfg=dict(scale=(224, 224), keep_ratio=False),
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
                                p=0.05,
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
            type="InternVideoNextBackbone",
            img_size=224,
            patch_size=14,
            embed_dim=1024,  # Large model
            depth=24,
            num_heads=16,
            mlp_ratio=4,
            qkv_bias=False,
            num_frames=16,
            tubelet_size=1,
            drop_path_rate=0.2,
            cls_token_num=4,
            return_feat_map=False,  # Output (B, C, T+1) for temporal modeling
            use_flash_attn=False,
            use_fused_rmsnorm=True,
            use_fused_mlp=True,
            use_checkpoint=True,
            checkpoint_num=24,  # Checkpoint first half of layers
            # Adapter configuration
            adapter_index=list(range(24)),  # Adapters at all layers
            adapter_conv_type=["2d_conv"] * 3 + ["sparse_conv"] * 21,
            keep_rate=0.7,
            adapter_mlp_ratio=0.25,
            adapter_use_attn=3,
            token_selection_index=[
                3,
                7,
                11,
                15,
            ],  # Insert adapter after these blocks
            # Pre-attention token pruning (reduces peak memory)
            pre_attn_keep_ratio=1.0,  # Keep 70% of tokens before first block
            pre_attn_scorer_type="spatiotemporal",
            pre_attn_reduction_ratio=4,
            pre_attn_tau_start=1.0,
            pre_attn_tau_end=0.1,
            pre_attn_tau_steps=10000,
            # Landmark configuration (set n_landmarks > 0 to enable)
            n_landmarks=26,  # Number of keypoint landmarks (0 to disable)
            hw_out_conv=(10, 10),  # Heatmap spatial resolution
        ),
        data_preprocessor=dict(
            type="mmaction.ActionDataPreprocessor",
            format_shape="NCTHW",
        ),
        custom=dict(
            pretrain="pretrained/internvideo_next_large_native.pth",
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
                    ops="b n c t -> b c t",  # Reduce across num_segs (n=1 typically)
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
            freeze_backbone=False,  # Freeze backbone, train only adapters
        ),
    ),
    projection=dict(
        in_channels=1024,  # Match embed_dim
        max_seq_len=window_size,
        attn_cfg=dict(n_mha_win_size=-1),
    ),
)

solver = dict(
    train=dict(
        batch_size=1,  # Reduced due to larger model
        num_workers=8,
        persistent_workers=True,
        prefetch_factor=1,
        pin_memory=True,
        multiprocessing_context="spawn",
    ),
    val=dict(
        batch_size=2,
        num_workers=2,
        persistent_workers=False,
        multiprocessing_context="spawn",
    ),
    test=dict(
        batch_size=2,
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
            dict(name="pre_attn_pruner", lr=2e-4, weight_decay=0.01),
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
        voting_thresh=0.7,
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

work_dir = (
    "exps/thumos/vitsparse/e2e_actionformer_internvideo_l_768x1_224_sparse_adapter"
)
# #%%
# input_tokens = 4096
# output_tokens = 985
# #ratio
# token_ratio = output_tokens / input_tokens
# print(f"Token ratio after pruning: {token_ratio:.4f}")