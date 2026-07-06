import os
import json
import numpy as np
import pandas as pd
from copy import deepcopy
from .base import SlidingWindowDataset, PaddingDataset, filter_same_annotation
from .builder import DATASETS
import re  # Import the regular expression library
from scipy.spatial.transform import Rotation as R_scipy
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import decord
import cv2

ORIGINAL_IMG_WIDTH = 1280
ORIGINAL_IMG_HEIGHT = 720


def _save_debug_plot(img_frame, keypoints_2d, save_path):
    """
    A simple utility to plot an image frame and its corresponding 2D keypoints.
    """
    fig, ax = plt.subplots(1, figsize=(16, 9), dpi=150)
    ax.imshow(img_frame)

    if keypoints_2d.size > 0:
        x_coords, y_coords = keypoints_2d[:, 0], keypoints_2d[:, 1]
        ax.scatter(
            x_coords, y_coords, s=20, c="cyan", marker="o", edgecolors="black", zorder=2
        )
        skeleton_connections = [
            (2, 3),
            (3, 26),
            (26, 27),
            (2, 11),
            (11, 12),
            (12, 13),
            (13, 14),
            (14, 15),
            (15, 16),
            (14, 17),
            (2, 4),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 8),
            (8, 9),
            (7, 10),
            (0, 1),
            (1, 2),
            (0, 22),
            (22, 23),
            (23, 24),
            (24, 25),
            (0, 18),
            (18, 19),
            (19, 20),
            (20, 21),
        ]
        for start, end in skeleton_connections:
            if start < len(x_coords) and end < len(x_coords):
                ax.plot(
                    [x_coords[start], x_coords[end]],
                    [y_coords[start], y_coords[end]],
                    "c-",
                    lw=1.5,
                    zorder=1,
                )

    ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


# Labels to be ignored during processing
VOID_LABELS = ["Error", "object__lift__both"]


def find_file_by_extension(directory, extensions):
    """
    Scans a directory to find the first file matching a list of extensions.
    """
    if not os.path.isdir(directory):
        return None
    for filename in os.listdir(directory):
        for ext in extensions:
            if filename.lower().endswith(ext):
                return os.path.join(directory, filename)
    return None


def find_filenames_txt(directory):
    if not os.path.isdir(directory):
        return None
    for filename in os.listdir(directory):
        if filename.endswith("_filenames.txt"):
            return os.path.join(directory, filename)
    fallback_path = os.path.join(directory, "filenames.txt")
    if os.path.exists(fallback_path):
        return fallback_path
    return None


# --- Main Dataset Classes ---


@DATASETS.register_module()
class AttachSlidingDataset(SlidingWindowDataset):
    # --- MODIFIED: Simplified constructor ---
    def __init__(self, skeleton_data_path_2d=None, debug=False, **kwargs):
        super().__init__(**kwargs)
        self.skeleton_data_path_2d = skeleton_data_path_2d
        self.skeleton_cache_2d = {}
        self.filenames_cache = {}
        self.debug = debug
        self.debug_run_once = False

    def get_gt(self, video_info, thresh=0.0):
        # ... (this method is unchanged) ...
        gt_segment, gt_label = [], []
        for anno in video_info["annotations"]:
            if anno["label"] in VOID_LABELS:
                continue
            gt_start = int(
                anno["segment"][0] / video_info["duration"] * video_info["frame"]
            )
            gt_end = int(
                anno["segment"][1] / video_info["duration"] * video_info["frame"]
            )
            if (not self.filter_gt) or (gt_end - gt_start > thresh):
                gt_segment.append([gt_start, gt_end])
                gt_label.append(self.class_map.index(anno["label"]))
        if len(gt_segment) == 0:
            return None
        return filter_same_annotation(
            dict(
                gt_segments=np.array(gt_segment, dtype=np.float32),
                gt_labels=np.array(gt_label, dtype=np.int32),
            )
        )

    def __getitem__(self, index):
        # ... (debug block can be added here if needed, similar to previous versions) ...
        video_name, video_info, video_anno, window_snippet_centers = self.data_list[
            index
        ]
        video_directory = os.path.join(self.data_path, video_name)
        full_video_path = find_file_by_extension(video_directory, [".mp4"])

        # --- MODIFIED: Load 2D Skeletons instead of 3D ---
        # if self.skeleton_cache_2d is not None and video_name not in self.skeleton_cache_2d:
        #     csv_path = os.path.join(self.skeleton_data_path_2d, video_name)
        #     csv_path = find_file_by_extension(csv_path, [".csv"])
        #     try:
        #         self.skeleton_cache_2d[video_name] = pd.read_csv(csv_path, index_col=0)
        #     except (FileNotFoundError, TypeError):
        #         self.skeleton_cache_2d[video_name] = None
        # if self.skeleton_cache_2d is not None:
        #     full_skeleton_df_2d = self.skeleton_cache_2d.get(video_name)

        # Load timestamps for frame mapping
        if video_name not in self.filenames_cache:
            filenames_path = find_filenames_txt(video_directory)
            timestamps = []
            if filenames_path:
                with open(filenames_path, "r", encoding="latin-1") as f:
                    for line in f:
                        match = re.search(r"(\d{10,})", line)
                        if match:
                            timestamps.append(int(match.group(0)))
            self.filenames_cache[video_name] = timestamps
        timestamp_map = self.filenames_cache.get(video_name, [])

        # --- MODIFIED: Filter 2D skeletons and create hybrid keypoints ---
        # hybrid_keypoints_for_aug = np.array([])
        # if full_skeleton_df_2d is not None and timestamp_map:
        #     start_frame_idx = int(window_snippet_centers[0])
        #     end_frame_idx = int(window_snippet_centers[-1])
        #     start_timestamp = timestamp_map[start_frame_idx]
        #     end_timestamp = timestamp_map[end_frame_idx]

        #     filtered_2d = full_skeleton_df_2d[
        #         (full_skeleton_df_2d.index >= start_timestamp)
        #         & (full_skeleton_df_2d.index <= end_timestamp)
        #     ]
        #     if not filtered_2d.empty:
        #         # 2D data has 64 columns (32 joints * 2 coords)
        #         keypoints_2d = filtered_2d.iloc[:, :64].to_numpy()
        #         t, _ = keypoints_2d.shape
        #         # Reshape to (T, 32, 2)
        #         keypoints_2d_reshaped = keypoints_2d.reshape(t, 32, 2)
        #         keypoints_2d_reshaped[..., 0] *= ORIGINAL_IMG_WIDTH  # Scale X by width
        #         keypoints_2d_reshaped[
        #             ..., 1
        #         ] *= ORIGINAL_IMG_HEIGHT  # Scale Y by height
        #         # Create a dummy Z channel of zeros
        #         dummy_z = np.zeros((t, 32, 1), dtype=np.float32)

        #         # Stack to create (u, v, Z_dummy) for compatibility
        #         hybrid_keypoints_for_aug = np.concatenate(
        #             [keypoints_2d_reshaped, dummy_z], axis=-1
        #         )

        # Prepare results dictionary
        if video_anno:
            video_anno = deepcopy(video_anno)
            video_anno["gt_segments"] = (
                video_anno["gt_segments"]
                - window_snippet_centers[0]
                - self.offset_frames
            ) / self.snippet_stride

        results_dict = dict(
            filename=full_video_path,
            # keypoint=hybrid_keypoints_for_aug,
            **video_anno,
            video_name=video_name,
            modality="RGB",
            data_path=self.data_path,
            window_size=self.window_size,
            feature_start_idx=int(window_snippet_centers[0] / self.snippet_stride),
            feature_end_idx=int(window_snippet_centers[-1] / self.snippet_stride),
            sample_stride=self.sample_stride,
            fps=video_info["frame"] / video_info["duration"],
            snippet_stride=self.snippet_stride,
            window_start_frame=window_snippet_centers[0],
            duration=video_info["duration"],
            offset_frames=self.offset_frames,
            original_shape_kp=(ORIGINAL_IMG_HEIGHT, ORIGINAL_IMG_WIDTH)
        )
        return self.pipeline(results_dict)


@DATASETS.register_module()
class AttachPaddingDataset(PaddingDataset):
    def __init__(self, skeleton_data_path_2d=None, debug=False, **kwargs):
        super().__init__(**kwargs)
        self.skeleton_data_path_2d = skeleton_data_path_2d
        self.skeleton_cache_2d = {}
        self.filenames_cache = {}  # Added for consistency
        self.debug = debug
        self.debug_run_once = False

    def get_gt(self, video_info, thresh=0.0):
        gt_segment, gt_label = [], []
        for anno in video_info["annotations"]:
            if anno["label"] in VOID_LABELS:
                continue
            gt_start = int(
                anno["segment"][0] / video_info["duration"] * video_info["frame"]
            )
            gt_end = int(
                anno["segment"][1] / video_info["duration"] * video_info["frame"]
            )
            if (not self.filter_gt) or (gt_end - gt_start > thresh):
                gt_segment.append([gt_start, gt_end])
                gt_label.append(self.class_map.index(anno["label"]))
        if len(gt_segment) == 0:
            return None
        return filter_same_annotation(
            dict(
                gt_segments=np.array(gt_segment, dtype=np.float32),
                gt_labels=np.array(gt_label, dtype=np.int32),
            )
        )

    def __getitem__(self, index):
        video_name, video_info, video_anno = self.data_list[index]
        video_directory = os.path.join(self.data_path, video_name)
        full_video_path = find_file_by_extension(video_directory, [".mp4"])

        # --- Get total frames for the video ---
        total_frames = video_info["frame"]

        # --- Load 2D Skeletons (if not cached) ---
        if (
            self.skeleton_data_path_2d is not None
            and video_name not in self.skeleton_cache_2d
        ):
            csv_path = os.path.join(self.skeleton_data_path_2d, video_name)
            csv_path = find_file_by_extension(csv_path, [".csv"])
            try:
                # Load and sort by timestamp index to ensure correct alignment
                df = pd.read_csv(csv_path, index_col=0)
                df.sort_index(inplace=True)
                self.skeleton_cache_2d[video_name] = df
            except (FileNotFoundError, TypeError):
                self.skeleton_cache_2d[video_name] = None
        if self.skeleton_data_path_2d is not None:
            full_skeleton_df_2d = self.skeleton_cache_2d.get(video_name)

        # --- Load frame timestamps from filenames.txt (if not cached) ---
        if video_name not in self.filenames_cache:
            filenames_path = find_filenames_txt(video_directory)
            timestamps = []
            if filenames_path:
                with open(filenames_path, "r", encoding="latin-1") as f:
                    for line in f:
                        match = re.search(r"(\d{10,})", line)
                        if match:
                            timestamps.append(int(match.group(0)))
            self.filenames_cache[video_name] = timestamps
        frame_timestamps = self.filenames_cache.get(video_name, [])

        # --- START: MODIFIED KEYPOINT ALIGNMENT LOGIC ---
        hybrid_keypoints_for_aug = None
        if (
            self.skeleton_data_path_2d is not None
            and full_skeleton_df_2d is not None
            and frame_timestamps
        ):
            hybrid_keypoints_for_aug = np.zeros((total_frames, 32, 3), dtype=np.float32)

            # Ensure the number of timestamps matches the number of frames
            # Truncate or pad if necessary, though they should ideally match
            if len(frame_timestamps) != total_frames:
                # This is a data sanity warning; you might want to log it
                frame_timestamps = frame_timestamps[:total_frames]

            # Create a DataFrame for video frames with their timestamps
            frame_df = pd.DataFrame(index=frame_timestamps)

            # Use pd.merge_asof to find the nearest keypoint for each frame.
            # This is highly efficient for time-series data.
            # 'direction=nearest' finds the absolute closest keypoint in time.
            # 'tolerance' can be set if keypoints too far in time should be ignored (e.g., pd.Timedelta('100ms'))
            aligned_df = pd.merge_asof(
                left=frame_df,
                right=full_skeleton_df_2d,
                left_index=True,
                right_index=True,
                direction="nearest",
            )

            # Fill any remaining NaNs (e.g., at the very beginning) by back-filling
            aligned_df.bfill(inplace=True)  # First, back-fill for NaNs at the start.
            aligned_df.ffill(
                inplace=True
            )  # Then, forward-fill for any remaining NaNs at the end.

            if not aligned_df.empty:
                # Process the aligned keypoints
                keypoints_2d = aligned_df.iloc[:, :64].to_numpy()
                t, _ = keypoints_2d.shape
                keypoints_2d_reshaped = keypoints_2d.reshape(t, 32, 2)
                keypoints_2d_reshaped[..., 0] *= ORIGINAL_IMG_WIDTH
                keypoints_2d_reshaped[..., 1] *= ORIGINAL_IMG_HEIGHT
                dummy_z = np.zeros((t, 32, 1), dtype=np.float32)

                # Assign to the final array, ensuring shape matches total_frames
                processed_keypoints = np.concatenate(
                    [keypoints_2d_reshaped, dummy_z], axis=-1
                )

                # The length of processed_keypoints should match total_frames,
                # but we slice just in case of minor discrepancies.
                num_aligned_frames = min(total_frames, len(processed_keypoints))
                hybrid_keypoints_for_aug[:num_aligned_frames] = processed_keypoints[
                    :num_aligned_frames
                ]
                if num_aligned_frames > 0 and num_aligned_frames < total_frames:
                    last_valid_keypoint = hybrid_keypoints_for_aug[
                        num_aligned_frames - 1
                    ]
                    # Repeat this last keypoint for all remaining frames
                    hybrid_keypoints_for_aug[num_aligned_frames:] = last_valid_keypoint
        # --- END: MODIFIED KEYPOINT ALIGNMENT LOGIC ---

        if video_anno:
            video_anno = deepcopy(video_anno)
            video_anno["gt_segments"] = (
                video_anno["gt_segments"] - self.offset_frames
            ) / self.snippet_stride
        results_dict = dict(
            filename=full_video_path,
            keypoint=hybrid_keypoints_for_aug,
            total_frames=total_frames,  # Pass total_frames to the pipeline
            **video_anno,
            video_name=video_name,
            modality="RGB",
            data_path=self.data_path,
            sample_stride=self.sample_stride,
            snippet_stride=self.snippet_stride,
            fps=video_info["frame"] / video_info["duration"],
            duration=video_info["duration"],
            offset_frames=self.offset_frames,
            original_shape_kp=(ORIGINAL_IMG_HEIGHT, ORIGINAL_IMG_WIDTH)
        )
        if results_dict["keypoint"] is None:
            results_dict.pop("keypoint")
        return self.pipeline(results_dict)

    def _run_debug_visualization(self, index, save_path):
        video_name, video_info, _ = self.data_list[index]
        print(f"\n--- Running Debug Visualization for video: {video_name} ---")

        video_directory = os.path.join(self.data_path, video_name)
        full_video_path = find_file_by_extension(video_directory, [".mp4"])

        full_skeleton_df_2d = self.skeleton_cache_2d.get(video_name)
        if full_skeleton_df_2d is None:
            csv_path = os.path.join(self.skeleton_data_path_2d, video_name)
            csv_path = find_file_by_extension(csv_path, [".csv"])
            try:
                full_skeleton_df_2d = pd.read_csv(csv_path, index_col=0)
                self.skeleton_cache_2d[video_name] = full_skeleton_df_2d
            except (FileNotFoundError, TypeError):
                print("DEBUG ERROR: 2D skeleton file not found.")
                return

        frame_idx_to_load = 0
        try:
            vr = decord.VideoReader(full_video_path)
            img_frame = vr[frame_idx_to_load].asnumpy()
        except Exception as e:
            print(f"DEBUG ERROR: Failed to read frame {frame_idx_to_load}. Error: {e}")
            return
        keypoints_2d_pixels = np.array([])
        filenames_path = find_filenames_txt(video_directory)
        if filenames_path:
            with open(filenames_path, "r", encoding="latin-1") as f:
                lines = f.readlines()
            if frame_idx_to_load < len(lines):
                line = lines[frame_idx_to_load]
                match = re.search(r"(\d{10,})", line)
                if match:
                    timestamp = int(match.group(0))
                    time_diffs = np.abs(full_skeleton_df_2d.index - timestamp)
                    closest_iloc = time_diffs.argmin()
                    kps_norm = full_skeleton_df_2d.iloc[closest_iloc, :64].to_numpy()
                    kps_reshaped = kps_norm.reshape(32, 2)
                    keypoints_2d_pixels = kps_reshaped.copy()
                    keypoints_2d_pixels[:, 0] *= ORIGINAL_IMG_WIDTH
                    keypoints_2d_pixels[:, 1] *= ORIGINAL_IMG_HEIGHT

        _save_debug_plot(img_frame, keypoints_2d_pixels, save_path)
