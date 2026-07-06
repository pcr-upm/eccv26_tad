import numpy as np
import os
import pickle
from copy import deepcopy
from .base import (
    ResizeDataset,
    PaddingDataset,
    SlidingWindowDataset,
    filter_same_annotation,
)
from .builder import DATASETS


def _load_keypoints_impl(
    video_name,
    total_frames,
    preprocessed_skeleton_path,
    skeleton_data_path_2d,
    aligned_cache,
    file_exists_cache,
    skeleton_cache,
):
    """Shared keypoint loading implementation with caching optimizations.

    Args:
        video_name: Name of video (without extension)
        total_frames: Expected number of frames
        preprocessed_skeleton_path: Path to preprocessed .npy files
        skeleton_data_path_2d: Path to pickle files (fallback)
        aligned_cache: Dict to cache aligned keypoint arrays (video_name -> array)
        file_exists_cache: Dict to cache file existence checks (video_name -> bool)
        skeleton_cache: Dict to cache processed pickle data (video_name -> array)

    Returns:
        Keypoints array of shape (T, MAX_PEOPLE, K, 2) or None
    """
    if skeleton_data_path_2d is None and preprocessed_skeleton_path is None:
        return None

    # Check aligned cache first - returns fully processed array
    if video_name in aligned_cache:
        return aligned_cache[video_name]

    # Try preprocessed .npy first (memory-mapped for efficient multi-worker access)
    if preprocessed_skeleton_path is not None:
        # Check file existence (with caching)
        if video_name not in file_exists_cache:
            npy_path = os.path.join(preprocessed_skeleton_path, video_name + ".npy")
            file_exists_cache[video_name] = os.path.exists(npy_path)

        if file_exists_cache[video_name]:
            npy_path = os.path.join(preprocessed_skeleton_path, video_name + ".npy")
            try:
                # Load with mmap, then copy to regular array for fast subsequent access
                keypoints_mmap = np.load(npy_path, mmap_mode="r")
                T_pkl = keypoints_mmap.shape[0]

                if T_pkl == total_frames:
                    # Copy to regular array (faster for repeated access)
                    keypoints = np.array(keypoints_mmap)
                elif T_pkl < total_frames:
                    # Need padding - allocate once and copy
                    MAX_PEOPLE, K = keypoints_mmap.shape[1], keypoints_mmap.shape[2]
                    keypoints = np.zeros(
                        (total_frames, MAX_PEOPLE, K, 2), dtype=keypoints_mmap.dtype
                    )
                    keypoints[:T_pkl] = keypoints_mmap
                else:
                    # Truncate
                    keypoints = np.array(keypoints_mmap[:total_frames])

                aligned_cache[video_name] = keypoints
                return keypoints
            except Exception as e:
                print(f"Error loading preprocessed keypoints for {video_name}: {e}")
                aligned_cache[video_name] = None
                return None
        else:
            aligned_cache[video_name] = None
            return None

    # Fallback to pickle processing (with per-worker cache)
    if skeleton_data_path_2d is None:
        return None

    if video_name in skeleton_cache:
        return skeleton_cache[video_name]

    pkl_path = os.path.join(skeleton_data_path_2d, video_name + ".pkl")
    if not os.path.exists(pkl_path):
        skeleton_cache[video_name] = None
        return None

    try:
        with open(pkl_path, "rb") as f:
            video_results = pickle.load(f)

        # Extract keypoints
        final_kps = []

        # Determine K from first valid person if possible, else 26
        K = 26
        for frame_res in video_results:
            kps = frame_res.get("keypoints", [])
            if len(kps) > 0 and len(kps[0]) > 0:
                K = kps[0].shape[0]
                break

        # Determine MAX_PEOPLE
        MAX_PEOPLE = 0
        for frame_res in video_results:
            kps = frame_res.get("keypoints", [])
            MAX_PEOPLE = max(MAX_PEOPLE, len(kps))

        if MAX_PEOPLE == 0:
            MAX_PEOPLE = 1

        for frame_res in video_results:
            kps = frame_res.get("keypoints", [])
            scores = frame_res.get("scores", [])

            frame_person_kps = []
            num_people = len(kps)

            for i in range(MAX_PEOPLE):
                if i < num_people:
                    kp = kps[i]
                    if len(scores) > i:
                        score = scores[i]
                        low_conf_mask = score < 0.5
                        kp[low_conf_mask] = 0
                        score[low_conf_mask] = 0
                else:
                    kp = np.zeros((K, 2), dtype=np.float32)

                frame_person_kps.append(kp)

            frame_kps_stacked = np.stack(frame_person_kps, axis=0)
            final_kps.append(frame_kps_stacked)

        keypoints = np.stack(final_kps)

        # Align with total_frames
        T_pkl = keypoints.shape[0]
        if T_pkl != total_frames:
            if T_pkl < total_frames:
                padding = np.zeros(
                    (total_frames - T_pkl, MAX_PEOPLE, K, 2), dtype=np.float32
                )
                keypoints = np.concatenate([keypoints, padding], axis=0)
            else:
                keypoints = keypoints[:total_frames]

        skeleton_cache[video_name] = keypoints
        return keypoints
    except Exception as e:
        print(f"Error loading skeleton for {video_name}: {e}")
        skeleton_cache[video_name] = None
        return None


@DATASETS.register_module()
class AnetResizeDataset(ResizeDataset):
    def __init__(
        self, skeleton_data_path_2d=None, preprocessed_skeleton_path=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.skeleton_data_path_2d = skeleton_data_path_2d
        self.preprocessed_skeleton_path = preprocessed_skeleton_path
        self.skeleton_cache = {}  # Only used for pickle fallback
        # Optimization: Cache aligned keypoint arrays per video
        self._aligned_cache = {}  # video_name -> aligned keypoint array
        self._file_exists_cache = {}  # video_name -> bool

    def _load_keypoints(self, video_name, total_frames):
        """Load keypoints using shared implementation with caching."""
        return _load_keypoints_impl(
            video_name,
            total_frames,
            self.preprocessed_skeleton_path,
            self.skeleton_data_path_2d,
            self._aligned_cache,
            self._file_exists_cache,
            self.skeleton_cache,
        )

    def get_gt(self, video_info, thresh=0.01):
        gt_segment = []
        gt_label = []
        for anno in video_info["annotations"]:
            gt_start = float(anno["segment"][0])
            gt_end = float(anno["segment"][1])
            gt_scale = (gt_end - gt_start) / float(video_info["duration"])

            if (not self.filter_gt) or (gt_scale > thresh):
                gt_segment.append([gt_start, gt_end])
                if self.class_agnostic:
                    gt_label.append(0)
                else:
                    gt_label.append(self.class_map.index(anno["label"]))

        if len(gt_segment) == 0:  # have no valid gt
            return None
        else:
            annotation = dict(
                gt_segments=np.array(gt_segment, dtype=np.float32),
                gt_labels=np.array(gt_label, dtype=np.int32),
            )
            return filter_same_annotation(annotation)

    def __getitem__(self, index):
        video_name, video_info, video_anno = self.data_list[index]

        # Load keypoints if skeleton_data_path_2d is provided
        # fps is 15 across videos, calculate total frames from duration
        total_frames = int(float(video_info["duration"]) * 15)
        keypoint = self._load_keypoints("v_" + video_name, total_frames)

        if video_anno != {}:
            video_anno = deepcopy(video_anno)  # avoid modify the original dict

        results_dict = dict(
            video_name=video_name,
            data_path=self.data_path,
            resize_length=self.resize_length,
            sample_stride=self.sample_stride,
            # resize post process setting
            fps=-1,
            duration=float(video_info["duration"]),
            keypoint=keypoint,
            original_shape_kp=(
                video_info.get("height", 720),
                video_info.get("width", 1280),
            ),
            **video_anno,
        )
        if results_dict["keypoint"] is None:
            results_dict.pop("keypoint")

        results = self.pipeline(results_dict)
        return results


@DATASETS.register_module()
class AnetPaddingDataset(PaddingDataset):
    def get_gt(self, video_info, thresh=0.0):
        # if fps is not set, use the original fps
        fps = (
            self.fps
            if self.fps > 0
            else float(video_info["frame"]) / float(video_info["duration"])
        )

        gt_segment = []
        gt_label = []
        for anno in video_info["annotations"]:
            gt_start = float(anno["segment"][0] * fps)
            gt_end = float(anno["segment"][1] * fps)

            valid_gt = (
                (gt_end - gt_start > thresh)  # duration > thresh (eg. 0.0)
                and (gt_end - self.offset_frames > 0)  # end > 0
                and (
                    gt_start - self.offset_frames <= float(video_info["duration"]) * fps
                )  # start < video_length
            )
            if (not self.filter_gt) or valid_gt:
                gt_segment.append([gt_start, gt_end])
                if self.class_agnostic:
                    gt_label.append(0)
                else:
                    gt_label.append(self.class_map.index(anno["label"]))

        if len(gt_segment) == 0:  # have no valid gt
            return None
        else:
            annotation = dict(
                gt_segments=np.array(gt_segment, dtype=np.float32),
                gt_labels=np.array(gt_label, dtype=np.int32),
            )
            return filter_same_annotation(annotation)

    def __getitem__(self, index):
        video_name, video_info, video_anno = self.data_list[index]

        if video_anno != {}:
            video_anno = deepcopy(video_anno)  # avoid modify the original dict
            video_anno["gt_segments"] = video_anno["gt_segments"] - self.offset_frames
            video_anno["gt_segments"] = video_anno["gt_segments"] / self.snippet_stride

        results = self.pipeline(
            dict(
                video_name=video_name,
                data_path=self.data_path,
                sample_stride=self.sample_stride,
                snippet_stride=self.snippet_stride,
                # if fps is not set, use the original fps
                fps=(
                    self.fps
                    if self.fps > 0
                    else float(video_info["frame"]) / float(video_info["duration"])
                ),
                duration=float(video_info["duration"]),
                offset_frames=self.offset_frames,
                **video_anno,
            )
        )
        return results


@DATASETS.register_module()
class AnetSlidingDataset(SlidingWindowDataset):
    def get_gt(self, video_info, thresh=0.0):
        # if fps is not set, use the original fps
        fps = (
            self.fps
            if self.fps > 0
            else float(video_info["frame"]) / float(video_info["duration"])
        )

        gt_segment = []
        gt_label = []
        for anno in video_info["annotations"]:
            gt_start = float(anno["segment"][0] * fps)
            gt_end = float(anno["segment"][1] * fps)

            valid_gt = (
                (gt_end - gt_start > thresh)  # duration > thresh (eg. 0.0)
                and (gt_end - self.offset_frames > 0)  # end > 0
                and (
                    gt_start - self.offset_frames <= float(video_info["duration"]) * fps
                )  # start < video_length
            )
            if (not self.filter_gt) or valid_gt:
                gt_segment.append([gt_start, gt_end])
                if self.class_agnostic:
                    gt_label.append(0)
                else:
                    gt_label.append(self.class_map.index(anno["label"]))

        if len(gt_segment) == 0:  # have no valid gt
            return None
        else:
            annotation = dict(
                gt_segments=np.array(gt_segment, dtype=np.float32),
                gt_labels=np.array(gt_label, dtype=np.int32),
            )
            return filter_same_annotation(annotation)

    def __getitem__(self, index):
        video_name, video_info, video_anno, window_snippet_centers = self.data_list[
            index
        ]

        if video_anno != {}:
            video_anno = deepcopy(video_anno)  # avoid modify the original dict
            video_anno["gt_segments"] = (
                video_anno["gt_segments"]
                - window_snippet_centers[0]
                - self.offset_frames
            )
            video_anno["gt_segments"] = video_anno["gt_segments"] / self.snippet_stride

        results = self.pipeline(
            dict(
                video_name=video_name,
                data_path=self.data_path,
                window_size=self.window_size,
                # trunc window setting
                feature_start_idx=int(window_snippet_centers[0] / self.snippet_stride),
                feature_end_idx=int(window_snippet_centers[-1] / self.snippet_stride),
                sample_stride=self.sample_stride,
                # sliding post process setting
                fps=(
                    self.fps
                    if self.fps > 0
                    else float(video_info["frame"]) / float(video_info["duration"])
                ),
                snippet_stride=self.snippet_stride,
                window_start_frame=window_snippet_centers[0],
                duration=video_info["duration"],
                offset_frames=self.offset_frames,
                # training setting
                **video_anno,
            )
        )
        return results
