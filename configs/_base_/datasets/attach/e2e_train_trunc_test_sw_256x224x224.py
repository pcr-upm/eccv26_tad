# --- User-configurable paths ---
# Please update these paths to match your system's layout.

# Path to the JSON annotation file created by the preprocessing script.
# This example uses the 'person' split. If you created a 'camera' split, use that file instead.
annotation_path = "/media/ricardo/data/datasets/ATTACH/attach_debug_ann.json"

# Path where the class map file (e.g., 'attach_category_idx.txt') will be saved.
# The dataloader will generate this file automatically if it doesn't exist.
class_map = "/media/ricardo/data/datasets/ATTACH/attach_category_idx.txt"

# Path to the root of the raw ATTACH color video data.
# The dataloader expects the video subfolders (e.g., '00__0__spike') to be inside this directory.
data_path = "/media/ricardo/data/datasets/ATTACH/raw_attach_dataset/color/"

# No block list is needed for ATTACH based on the documentation.
block_list = None


# This window size corresponds to 256 features. With a feature_stride of 4,
# it covers 256 * 4 = 1024 frames (~34 seconds at 30fps). This is a reasonable baseline.
window_size = 256


# --- Dataset Configuration ---
dataset = dict(
    train=dict(
        type="AttachPaddingDataset",
        ann_file=annotation_path,
        subset_name="train",
        block_list=block_list,
        class_map=class_map,
        data_path=data_path,
        filter_gt=False,
        # Dataloader settings - these are good starting points from the THUMOS config
        feature_stride=4,
        sample_stride=1,
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(
                type="LoadFrames",
                num_clips=1,
                method="random_trunc",
                trunc_len=window_size,
                trunc_thresh=0.5,
                crop_ratio=[0.9, 1.0],
            ),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 256)),
            dict(type="mmaction.RandomResizedCrop"),
            dict(type="mmaction.Resize", scale=(224, 224), keep_ratio=False),
            dict(type="mmaction.Flip", flip_ratio=0.5),
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
        type="AttachSlidingDataset",
        ann_file=annotation_path,
        subset_name="val",
        block_list=block_list,
        class_map=class_map,
        data_path=data_path,
        filter_gt=False,
        # Dataloader settings
        feature_stride=4,
        sample_stride=1,
        window_size=window_size,
        window_overlap_ratio=0.25,
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(type="LoadFrames", num_clips=1, method="sliding_window"),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 224)),
            dict(type="mmaction.CenterCrop", crop_size=224),
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
        type="AttachSlidingDataset",
        ann_file=annotation_path,
        subset_name="test",
        block_list=block_list,
        class_map=class_map,
        data_path=data_path,
        filter_gt=False,
        test_mode=True,
        # Dataloader settings
        feature_stride=4,
        sample_stride=1,
        window_size=window_size,
        window_overlap_ratio=0.5,
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(type="LoadFrames", num_clips=1, method="sliding_window"),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 224)),
            dict(type="mmaction.CenterCrop", crop_size=224),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs"]),
            dict(type="Collect", inputs="imgs", keys=["masks"]),
        ],
    ),
)


# --- Evaluation Configuration ---
evaluation = dict(
    type="mAP",
    subset="test",
    tiou_thresholds=[0.3, 0.4, 0.5, 0.6, 0.7],
    ground_truth_filename=annotation_path,
)
