import numpy as np
import os
import pickle
from copy import deepcopy
from .base import SlidingWindowDataset, PaddingDataset, filter_same_annotation
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
    """
    Shared implementation for loading keypoints with optimized caching.

    Args:
        video_name: Name of the video
        total_frames: Expected number of frames
        preprocessed_skeleton_path: Path to preprocessed .npy files
        skeleton_data_path_2d: Path to raw pickle files (fallback)
        aligned_cache: Dict to cache aligned keypoint arrays (video_name -> array)
        file_exists_cache: Dict to cache file existence (video_name -> bool)
        skeleton_cache: Dict to cache pickle-loaded keypoints

    Returns:
        Keypoints array or None
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

        final_kps = []
        K = 26
        for frame_res in video_results:
            kps = frame_res.get("keypoints", [])
            if len(kps) > 0 and len(kps[0]) > 0:
                K = kps[0].shape[0]
                break

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
                else:
                    kp = np.zeros((K, 2), dtype=np.float32)
                frame_person_kps.append(kp)

            frame_kps_stacked = np.stack(frame_person_kps, axis=0)
            final_kps.append(frame_kps_stacked)

        keypoints = np.stack(final_kps)
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
class ThumosSlidingDataset(SlidingWindowDataset):
    def __init__(
        self,
        skeleton_data_path_2d=None,
        preprocessed_skeleton_path=None,
        debug=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.skeleton_data_path_2d = skeleton_data_path_2d
        self.preprocessed_skeleton_path = preprocessed_skeleton_path
        self.skeleton_cache = {}  # Only used for pickle fallback
        self.debug = debug
        # Optimization: Cache aligned keypoint arrays per video
        self._aligned_cache = {}  # video_name -> aligned keypoint array
        self._file_exists_cache = {}  # video_name -> bool

    def get_gt(self, video_info, thresh=0.0):
        gt_segment = []
        gt_label = []
        for anno in video_info["annotations"]:
            if anno["label"] == "Ambiguous":
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

        if len(gt_segment) == 0:  # have no valid gt
            return None
        else:
            annotation = dict(
                gt_segments=np.array(gt_segment, dtype=np.float32),
                gt_labels=np.array(gt_label, dtype=np.int32),
            )
            return filter_same_annotation(annotation)

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

    def __getitem__(self, index):
        video_name, video_info, video_anno, window_snippet_centers = self.data_list[
            index
        ]

        # keypoint = self._load_keypoints(video_name, video_info["frame"])

        if video_anno != {}:
            video_anno = deepcopy(video_anno)  # avoid modify the original dict
            # frame divided by snippet stride inside current window
            # this is only valid gt inside this window
            video_anno["gt_segments"] = (
                video_anno["gt_segments"]
                - window_snippet_centers[0]
                - self.offset_frames
            )
            video_anno["gt_segments"] = video_anno["gt_segments"] / self.snippet_stride

        results_dict = dict(
            video_name=video_name,
            data_path=self.data_path,
            window_size=self.window_size,
            # trunc window setting
            feature_start_idx=int(window_snippet_centers[0] / self.snippet_stride),
            feature_end_idx=int(window_snippet_centers[-1] / self.snippet_stride),
            sample_stride=self.sample_stride,
            # sliding post process setting
            fps=video_info["frame"] / video_info["duration"],
            snippet_stride=self.snippet_stride,
            window_start_frame=window_snippet_centers[0],
            duration=video_info["duration"],
            offset_frames=self.offset_frames,
            # training setting
            # keypoint=keypoint,
            original_shape_kp=(
                video_info.get("height", 720),
                video_info.get("width", 1280),
            ),
            **video_anno,
        )
        if "keypoint" in results_dict and results_dict["keypoint"] is None:
            results_dict.pop("keypoint")

        results = self.pipeline(results_dict)
        return results


@DATASETS.register_module()
class ThumosPaddingDataset(PaddingDataset):
    def __init__(
        self,
        skeleton_data_path_2d=None,
        preprocessed_skeleton_path=None,
        debug=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.skeleton_data_path_2d = skeleton_data_path_2d
        self.preprocessed_skeleton_path = preprocessed_skeleton_path
        self.skeleton_cache = {}  # Only used for pickle fallback
        self.debug = debug
        # Optimization: Cache aligned keypoint arrays per video
        self._aligned_cache = {}  # video_name -> aligned keypoint array
        self._file_exists_cache = {}  # video_name -> bool

    def get_gt(self, video_info, thresh=0.0):
        gt_segment = []
        gt_label = []
        for anno in video_info["annotations"]:
            if anno["label"] == "Ambiguous":
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

        if len(gt_segment) == 0:  # have no valid gt
            return None
        else:
            annotation = dict(
                gt_segments=np.array(gt_segment, dtype=np.float32),
                gt_labels=np.array(gt_label, dtype=np.int32),
            )
            return filter_same_annotation(annotation)

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

    def __getitem__(self, index):
        video_name, video_info, video_anno = self.data_list[index]

        keypoint = self._load_keypoints(video_name, video_info["frame"])

        if video_anno != {}:
            video_anno = deepcopy(video_anno)  # avoid modify the original dict
            video_anno["gt_segments"] = video_anno["gt_segments"] - self.offset_frames
            video_anno["gt_segments"] = video_anno["gt_segments"] / self.snippet_stride

        results_dict = dict(
            video_name=video_name,
            data_path=self.data_path,
            sample_stride=self.sample_stride,
            snippet_stride=self.snippet_stride,
            fps=video_info["frame"] / video_info["duration"],
            duration=video_info["duration"],
            offset_frames=self.offset_frames,
            keypoint=keypoint,
            original_shape_kp=(
                video_info.get("height", 180),
                video_info.get("width", 320),
            ),
            **video_anno,
        )
        if results_dict["keypoint"] is None:
            results_dict.pop("keypoint")
        results = self.pipeline(results_dict)
        return results
