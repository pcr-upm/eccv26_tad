# config_attach_multimodal.py
custom_imports = dict(
    imports=["opentad.datasets.kp_transforms"], allow_failed_imports=False
)
# --- User-configurable paths ---
annotation_path = "/datasets/ATTACH/141.24.24.111:50021/attach_person_split_ann.json"
class_map = "/datasets/ATTACH/141.24.24.111:50021/attach_category_idx.txt"
video_data_path = "/datasets/ATTACH/141.24.24.111:50021/raw_attach_dataset/color_resize/"
skeleton_data_path_2d = (
    "/datasets/ATTACH/141.24.24.111:50021/raw_attach_dataset/2d_azure_body_skeletons/"
)
block_list = None

# --- Model/Training hyperparameters ---
window_size = 256
# --- NEW: Max length for skeleton sequences within a window ---
max_skeleton_len = 1024  # Corresponds to window_size * feature_stride

# --- Dataset Configuration ---
dataset = dict(
    train=dict(
        # For simplicity, we'll use the sliding window for training too in this example.
        # You would apply the same logic to AttachPaddingDataset if needed.
        type="AttachPaddingDataset",
        ann_file=annotation_path,
        subset_name="train",
        block_list=block_list,
        class_map=class_map,
        data_path=video_data_path,
        skeleton_data_path_2d=skeleton_data_path_2d,
        filter_gt=False,
        feature_stride=4,
        sample_stride=1,
        pipeline=[
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(type="LoadFrames", num_clips=1, method="sliding_window"),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 256)),
            dict(type="mmaction.RandomResizedCrop"),
            dict(type="mmaction.Resize", scale=(224, 224), keep_ratio=False),
            dict(type="mmaction.Flip", flip_ratio=0.5),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(
                type="PadSkeletonSequence",
                max_len=max_skeleton_len,
                key="raw_keypoints",
                out_key="keypoints",
            ),
            dict(
                type="FormatSkeletonShape",
                input_format="T_V_C",
                target_format="C_T_V_M",
                key="keypoints",
            ),
            # Final steps to combine modalities
            dict(
                type="ConvertToTensor",
                keys=["imgs", "keypoints", "gt_segments", "gt_labels"],
            ),
            dict(
                type="Collect",
                inputs=["imgs", "keypoints"],
                keys=["masks", "gt_segments", "gt_labels"],
            ),
        ],
    ),
    val=dict(
        type="AttachSlidingDataset",
        ann_file=annotation_path,
        subset_name="val",
        block_list=block_list,
        class_map=class_map,
        data_path=video_data_path,
        skeleton_data_path_2d=skeleton_data_path_2d,
        filter_gt=False,
        feature_stride=4,
        sample_stride=1,
        window_size=window_size,
        window_overlap_ratio=0.25,
        pipeline=[
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(type="LoadFrames", num_clips=1, method="sliding_window"),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 224)),
            dict(type="mmaction.CenterCrop", crop_size=224),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            # Final
            dict(
                type="ConvertToTensor",
                keys=["imgs", "keypoint", "gt_segments", "gt_labels"],
            ),
            dict(
                type="Collect",
                inputs=["imgs", "keypoint"],
                keys=["masks", "gt_segments", "gt_labels"],
            ),
        ],
    ),
    test=dict(
        type="AttachSlidingDataset",
        ann_file=annotation_path,
        subset_name="test",
        block_list=block_list,
        class_map=class_map,
        data_path=video_data_path,
        skeleton_data_path_2d=skeleton_data_path_2d,
        filter_gt=False,
        feature_stride=4,
        sample_stride=1,
        window_size=window_size,
        window_overlap_ratio=0.25,
        pipeline=[
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(type="LoadFrames", num_clips=1, method="sliding_window"),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 224)),
            dict(type="mmaction.CenterCrop", crop_size=224),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            # Final
            dict(
                type="ConvertToTensor",
                keys=["imgs", "keypoint", "gt_segments", "gt_labels"],
            ),
            dict(
                type="Collect",
                inputs=["imgs", "keypoint"],
                keys=["masks", "gt_segments", "gt_labels"],
            ),
        ],
    ),
)

# --- Evaluation Configuration ---
evaluation = dict(
    type="mAP",
    subset="test",
    tiou_thresholds=[0.1, 0.2, 0.3, 0.4, 0.5],
    ground_truth_filename=annotation_path,
)
