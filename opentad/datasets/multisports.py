import os
import json
import pickle
import numpy as np
import copy
from .base import PaddingDataset, SlidingWindowDataset, filter_same_annotation
from .base.sliding_dataset import compute_gt_completeness
from .builder import DATASETS


@DATASETS.register_module()
class MultiSportsPaddingDataset(PaddingDataset):
    def get_class_map(self, class_map_path):
        # Override to handle pickle annotation file
        if not os.path.exists(class_map_path):
            if self.ann_file.endswith(".pkl"):
                self.logger(f"Generating class map from pickle file: {self.ann_file}")
                with open(self.ann_file, "rb") as f:
                    data = pickle.load(f)

                # MultiSports PKL has 'labels' key which is likely a list of class names
                # or we might need to deduce it if it's not straightforward.
                # Assuming data['labels'] exists and is a list of strings
                if "labels" in data:
                    class_map = data["labels"]
                    # Ensure it's a list even if it's None or something else
                    if class_map is None:
                        # Fallback if labels not present
                        class_map = []
                else:
                    class_map = []

                # Save it
                try:
                    with open(class_map_path, "w") as f:
                        for name in class_map:
                            if name:
                                f.write(str(name) + "\n")
                    self.logger(
                        f"Class map saved to {class_map_path}, total {len(class_map)} classes."
                    )
                except Exception as e:
                    self.logger(
                        f"Warning: Could not save class map to {class_map_path}: {e}"
                    )

                return class_map
            else:
                # Fallback to default behavior for JSON
                return super().get_class_map(class_map_path)
        else:
            return super().get_class_map(class_map_path)

    def get_dataset(self):
        # Override to load PKL instead of JSON if ann_file ends with .pkl
        if self.ann_file.endswith(".pkl"):
            self.logger(f"Loading annotations from pickle file: {self.ann_file}")
            with open(self.ann_file, "rb") as f:
                data = pickle.load(f)

            self.data_list = []

            # Load keys from pickle structure described in README
            # dict_keys(['labels', 'train_videos', 'test_videos', 'nframes', 'resolution', 'gttubes'])
            train_videos = set(
                data.get("train_videos", [])[0]
            )  # README says list with one split element
            test_videos = set(data.get("test_videos", [])[0])
            val_videos = set()
            if "val_videos" in data:
                val_videos = set(data.get("val_videos", [])[0])

            nframes_dict = data.get("nframes", {})
            resolution_dict = data.get("resolution", {})
            gttubes_dict = data.get("gttubes", {})

            # Prepare block list
            blocked_videos = []
            if self.block_list:
                if isinstance(self.block_list, list):
                    blocked_videos = self.block_list
                else:
                    with open(self.block_list, "r") as f:
                        blocked_videos = [line.rstrip("\n") for line in f]

            # Iterate videos
            if self.subset_name == "train":
                all_videos = train_videos
            elif self.subset_name == "val":
                all_videos = val_videos
            elif self.subset_name == "test":
                all_videos = test_videos
            
            for video_name in all_videos:
                if video_name in blocked_videos:
                    continue

                # Prepare video_info from provided dicts
                total_frames = nframes_dict.get(video_name, 0)
                resolution = resolution_dict.get(
                    video_name, (720, 1280)
                )  # Default to 720p

                # Heuristic for duration/fps
                fps = 25.0 if self.fps <= 0 else self.fps

                # Parse annotations
                anns = []
                video_tubes = gttubes_dict.get(video_name, {})
                # video_tubes structure: { label_idx: [tube1, tube2, ...]}
                # tube is numpy array (N, 5): <frame> <x1> <y1> <x2> <y2>

                for label_idx, tubes_list in video_tubes.items():
                    for tube in tubes_list:
                        if tube.shape[0] > 0:
                            # Frame indices in dataset start from 1
                            frames_1based = tube[:, 0]
                            # x1, y1, x2, y2
                            boxes = tube[:, 1:5]

                            # Normalize boxes to [0,1] using original resolution
                            if resolution[1] > 0 and resolution[0] > 0:
                                w = float(resolution[1])
                                h = float(resolution[0])
                                boxes = boxes.copy()
                                boxes[:, [0, 2]] /= w
                                boxes[:, [1, 3]] /= h
                                boxes = np.clip(boxes, 0.0, 1.0)

                            start_f = np.min(frames_1based) - 1  # Convert to 0-based
                            end_f = np.max(frames_1based) - 1

                            anns.append(
                                {
                                    "segment": [float(start_f), float(end_f)],
                                    "label": int(label_idx),
                                    "boxes": boxes.tolist(),
                                    "frames": (
                                        frames_1based - 1
                                    ).tolist(),  # 0-based frames
                                }
                            )

                video_info = dict(
                    subset=self.subset_name,  # simplified
                    duration=total_frames / fps,
                    frame=total_frames,
                    resolution=resolution,
                    annotations=anns,
                )

                if self.test_mode:
                    video_anno = {}
                else:
                    video_anno = self._parse_gt(video_info, fps)
                    if video_anno is None:
                        continue

                self.data_list.append([video_name, video_info, video_anno])

            self.logger(
                f"{self.subset_name} subset: {len(self.data_list)} videos loaded from PKL."
            )

        else:
            super().get_dataset()

    def _parse_gt(self, video_info, fps, thresh=0.0):
        gt_segment, gt_label, gt_tubes = [], [], []

        if video_info["subset"] != "train":
            # For val/test, keeping gt_tubes is often useful for evaluation
            # But typically 'test' mode might not have annotations.
            pass

        for anno in video_info["annotations"]:
            start_f = anno["segment"][0]
            end_f = anno["segment"][1]
            label = anno["label"]

            if (end_f - start_f) > thresh:
                gt_segment.append([start_f, end_f])
                gt_label.append(label)
                # Store full tube info
                gt_tubes.append(
                    {
                        "boxes": anno.get("boxes", []),
                        "frames": anno.get("frames", []),
                        "label": label,
                    }
                )

        if len(gt_segment) == 0:
            return None

        return dict(
            gt_segments=np.array(gt_segment, dtype=np.float32),
            gt_labels=np.array(gt_label, dtype=np.int32),
            gt_tubes=gt_tubes,
        )

    def __getitem__(self, index):
        video_name, video_info, video_anno = self.data_list[index]

        # Resolve video path, searching subdirectories if necessary
        extensions = [".mp4", ".avi", ".mkv"]
        full_video_path = None

        # 1. Try direct path (standard)
        for ext in extensions:
            cand = os.path.join(self.data_path, video_name + ext)
            if os.path.exists(cand):
                full_video_path = cand
                break

        # 2. If not found, search recursively (heuristic for subfolder organization)
        if full_video_path is None:
            for root, dirs, files in os.walk(self.data_path):
                found = False
                for ext in extensions:
                    if (video_name + ext) in files:
                        full_video_path = os.path.join(root, video_name + ext)
                        found = True
                        break
                if found:
                    break

        if full_video_path is None:
            # Fallback to mp4 if not found (might be missing data)
            full_video_path = os.path.join(self.data_path, video_name + ".mp4")

        results_dict = dict(
            filename=full_video_path,
            video_name=video_name,
            modality="RGB",
            data_path=self.data_path,
            duration=video_info["duration"],
            total_frames=video_info["frame"],
            fps=video_info["frame"] / video_info["duration"],
            snippet_stride=self.snippet_stride,
            **video_anno,
        )

        # Note: pipeline steps like LoadFrames might need 'duration' or 'total_frames'
        # or they read it from file.

        return self.pipeline(results_dict)


@DATASETS.register_module()
class MultiSportsSlidingDataset(SlidingWindowDataset):
    def split_video_to_windows(self, video_name, video_info, video_anno):
        if self.fps > 0:
            num_frames = int(video_info["duration"] * self.fps)
        else:
            num_frames = video_info["frame"]

        video_snippet_centers = np.arange(0, num_frames, self.snippet_stride)
        snippet_num = len(video_snippet_centers)

        data_list = []
        last_window = False

        for idx in range(max(1, snippet_num // self.window_stride)):
            window_start = idx * self.window_stride
            window_end = window_start + self.window_size

            if window_end > snippet_num:
                window_end = snippet_num
                window_start = max(0, window_end - self.window_size)
                last_window = True

            window_snippet_centers = video_snippet_centers[window_start:window_end]
            window_start_frame = window_snippet_centers[0]

            if video_anno != {}:

                gt_segments = video_anno.get("gt_segments", [])
                gt_labels = video_anno.get("gt_labels", [])
                gt_tubes = video_anno.get("gt_tubes", [])

                anchor_start = window_start_frame
                # Approximate end frame coverage
                anchor_end = (
                    window_snippet_centers[-1] + self.snippet_stride
                    if len(window_snippet_centers) > 0
                    else anchor_start
                )
                anchor = np.array([anchor_start, anchor_end])

                scores, truncated_gt = compute_gt_completeness(gt_segments, anchor)

                if self.ioa_thresh > 0:
                    valid_idx = scores > self.ioa_thresh
                else:
                    # Keep all that have any overlap? Or just all?
                    # valid_idx = scores > 0
                    # If we don't filter windows, we take all.
                    valid_idx = np.ones(len(scores), dtype=bool)

                # If filtering is enabled and no valid GT, skip window
                if self.ioa_thresh > 0 and np.sum(valid_idx) == 0:
                    if last_window:
                        break
                    continue

                window_tubes = []
                window_labels = []
                window_segments = []

                indices_to_include = np.where(valid_idx)[0]

                for i in indices_to_include:
                    tube = gt_tubes[i]
                    label = gt_labels[i]
                    seg = truncated_gt[i]  # This is truncated segment [start, end]

                    t_frames = np.array(tube["frames"])
                    t_boxes = np.array(tube["boxes"])

                    # Find frames inside the window [anchor_start, anchor_end)
                    mask = (t_frames >= anchor_start) & (t_frames < anchor_end)

                    if np.sum(mask) > 0:
                        new_frames = t_frames[mask] - anchor_start
                        new_boxes = t_boxes[mask]

                        window_tubes.append(
                            {
                                "frames": new_frames.tolist(),
                                "boxes": new_boxes.tolist(),
                                "label": label,
                            }
                        )
                        window_labels.append(label)
                        window_segments.append(seg)

                if len(window_tubes) > 0 or self.ioa_thresh <= 0:
                    window_anno = dict(
                        gt_segments=(
                            np.array(window_segments, dtype=np.float32)
                            if len(window_segments) > 0
                            else np.zeros((0, 2), dtype=np.float32)
                        ),
                        gt_labels=(
                            np.array(window_labels, dtype=np.int32)
                            if len(window_labels) > 0
                            else np.zeros((0,), dtype=np.int32)
                        ),
                        gt_tubes=window_tubes,
                    )

                    data_list.append(
                        [video_name, video_info, window_anno, window_snippet_centers]
                    )

            else:
                # No GT, just append window
                data_list.append(
                    [video_name, video_info, video_anno, window_snippet_centers]
                )

            if last_window:
                break

        return data_list

    def get_class_map(self, class_map_path):
        # Override to handle pickle annotation file
        if not os.path.exists(class_map_path):
            if self.ann_file.endswith(".pkl"):
                self.logger(f"Generating class map from pickle file: {self.ann_file}")
                with open(self.ann_file, "rb") as f:
                    data = pickle.load(f)

                if "labels" in data:
                    class_map = data["labels"]
                    if class_map is None:
                        class_map = []
                else:
                    class_map = []

                try:
                    with open(class_map_path, "w") as f:
                        for name in class_map:
                            if name:
                                f.write(str(name) + "\\n")
                    self.logger(
                        f"Class map saved to {class_map_path}, total {len(class_map)} classes."
                    )
                except Exception as e:
                    self.logger(
                        f"Warning: Could not save class map to {class_map_path}: {e}"
                    )

                return class_map
            else:
                return super().get_class_map(class_map_path)
        else:
            return super().get_class_map(class_map_path)

    def _parse_gt(self, video_info, fps, thresh=0.0):
        gt_segment, gt_label, gt_tubes = [], [], []

        for anno in video_info["annotations"]:
            start_f = anno["segment"][0]
            end_f = anno["segment"][1]
            label = anno["label"]

            if (end_f - start_f) > thresh:
                gt_segment.append([start_f, end_f])
                gt_label.append(label)
                gt_tubes.append(
                    {
                        "boxes": anno.get("boxes", []),
                        "frames": anno.get("frames", []),
                        "label": label,
                    }
                )

        if len(gt_segment) == 0:
            return None

        return dict(
            gt_segments=np.array(gt_segment, dtype=np.float32),
            gt_labels=np.array(gt_label, dtype=np.int32),
            gt_tubes=gt_tubes,
        )

    def get_dataset(self):
        # Override to load PKL instead of JSON if ann_file ends with .pkl
        # Same as MultiSportsPaddingDataset until calling split_video_to_windows
        if self.ann_file.endswith(".pkl"):
            self.logger(f"Loading annotations from pickle file: {self.ann_file}")
            with open(self.ann_file, "rb") as f:
                data = pickle.load(f)

            self.data_list = []

            # Load keys from pickle structure
            train_videos = set(data.get("train_videos", [])[0])
            test_videos = set(data.get("test_videos", [])[0])
            val_videos = set()
            if "val_videos" in data:
                val_videos = set(data.get("val_videos", [])[0])
            if self.subset_name == "train":
                all_videos = train_videos
            elif self.subset_name == "val":
                all_videos = val_videos
            elif self.subset_name == "test":
                all_videos = test_videos
            nframes_dict = data.get("nframes", {})
            resolution_dict = data.get("resolution", {})
            gttubes_dict = data.get("gttubes", {})

            # Prepare block list
            blocked_videos = []
            if self.block_list:
                if isinstance(self.block_list, list):
                    blocked_videos = self.block_list
                else:
                    with open(self.block_list, "r") as f:
                        blocked_videos = [line.rstrip("\n") for line in f]

            for video_name in all_videos:
                if video_name in blocked_videos:
                    continue

                # Prepare video_info
                total_frames = nframes_dict.get(video_name, 0)
                resolution = resolution_dict.get(video_name, (720, 1280))
                fps = 25.0 if self.fps <= 0 else self.fps

                # Parse annotations
                anns = []
                video_tubes = gttubes_dict.get(video_name, {})
                for label_idx, tubes_list in video_tubes.items():
                    for tube in tubes_list:
                        if tube.shape[0] > 0:
                            frames_1based = tube[:, 0]
                            boxes = tube[:, 1:5]
                            if resolution[1] > 0 and resolution[0] > 0:
                                w = float(resolution[1])
                                h = float(resolution[0])
                                boxes = boxes.copy()
                                boxes[:, [0, 2]] /= w
                                boxes[:, [1, 3]] /= h
                                boxes = np.clip(boxes, 0.0, 1.0)

                            start_f = np.min(frames_1based) - 1
                            end_f = np.max(frames_1based) - 1
                            anns.append(
                                {
                                    "segment": [float(start_f), float(end_f)],
                                    "label": int(label_idx),
                                    "boxes": boxes.tolist(),
                                    "frames": (frames_1based - 1).tolist(),
                                }
                            )

                video_info = dict(
                    subset=(
                        self.subset_name
                        if not isinstance(self.subset_name, (list, tuple))
                        else "train"
                    ),
                    duration=total_frames / fps,
                    frame=total_frames,
                    resolution=resolution,
                    annotations=anns,
                )

                if self.test_mode:
                    video_anno = {}
                else:
                    video_anno = self._parse_gt(video_info, fps)
                    if video_anno is None:
                        continue

                # Split video into windows
                tmp_data_list = self.split_video_to_windows(
                    video_name, video_info, video_anno
                )
                self.data_list.extend(tmp_data_list)

            self.logger(
                f"{self.subset_name} subset: Loaded {len(all_videos)} videos, "
                f"generated {len(self.data_list)} sliding windows."
            )
        else:
            # Fallback to base sliding dataset (JSON)
            SlidingWindowDataset.get_dataset(self)

    def __getitem__(self, index):
        video_name, video_info, video_anno, window_snippet_centers = self.data_list[
            index
        ]
        video_anno = copy.deepcopy(video_anno)

        # NOTE: gt_segments usually normalized by snippet_stride in Thumos.
        # But for End-to-End, we operate on Frames.
        # Check snippet_stride. In config it is 4*1=4.
        # window_snippet_centers are frame indices [0, 4, 8, ...].

        # Resolve path (same as MultiSportsPaddingDataset)
        extensions = [".mp4", ".avi", ".mkv"]
        full_video_path = None
        for ext in extensions:
            cand = os.path.join(self.data_path, video_name + ext)
            if os.path.exists(cand):
                full_video_path = cand
                break
        if full_video_path is None:
            for root, dirs, files in os.walk(self.data_path):
                found = False
                for ext in extensions:
                    if (video_name + ext) in files:
                        full_video_path = os.path.join(root, video_name + ext)
                        found = True
                        break
                if found:
                    break
        if full_video_path is None:
            full_video_path = os.path.join(self.data_path, video_name + ".mp4")

        first_frame = int(window_snippet_centers[0])
        last_frame = int(window_snippet_centers[-1])


        # Pass indices relative to snippets (if LoadFrames expects snippet indices)
        # LoadFrames(sliding_window) expects:
        # frame_stride = snippet_stride / scale_factor.
        # feat_start = results['feature_start_idx']

        results_dict = dict(
            filename=full_video_path,
            video_name=video_name,
            modality="RGB",
            data_path=self.data_path,
            duration=video_info["duration"],
            total_frames=video_info["frame"],
            fps=(
                video_info["frame"] / video_info["duration"]
                if video_info["duration"] > 0
                else 25.0
            ),
            snippet_stride=self.snippet_stride,
            window_size=self.window_size,
            feature_start_idx=int(first_frame / self.snippet_stride),
            feature_end_idx=int(last_frame / self.snippet_stride),
            **video_anno,
        )

        return self.pipeline(results_dict)
