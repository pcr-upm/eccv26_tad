import copy
import os
import pickle
import random
import torch
import random
import pandas as pd
import numpy as np
import cv2

from ..builder import PIPELINES
from torch.nn import functional as F


@PIPELINES.register_module()
class PrepareVideoInfo:
    def __init__(self, format="mp4", modality="RGB", prefix=""):
        self.format = format
        self.modality = modality
        self.prefix = prefix

    def __call__(self, results):
        results["modality"] = self.modality
        results["filename"] = os.path.join(
            results["data_path"],
            self.prefix + results["video_name"] + "." + self.format,
        )
        return results


@PIPELINES.register_module()
class LoadSnippetFrames:
    """Load the snippet frame, the output should follows the format:
    snippet_num x channel x clip_len x height x width
    """

    def __init__(
        self,
        clip_len,
        frame_interval=1,
        method="resize",
        trunc_len=None,
        trunc_thresh=None,
        crop_ratio=None,
    ):
        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.method = method  # resize or padding or sliding window
        # todo: support to  change FPS
        # random_trunc settings
        self.trunc_len = trunc_len
        self.trunc_thresh = trunc_thresh
        self.crop_ratio = crop_ratio

    def random_trunc(
        self, feats, trunc_len, gt_segments, gt_labels, offset=0, max_num_trials=200
    ):
        feat_len = feats.shape[0]
        num_segs = gt_segments.shape[0]

        trunc_len = trunc_len
        if feat_len <= trunc_len:
            if self.crop_ratio == None:  # do nothing
                return feats, gt_segments, gt_labels
            else:  # randomly crop the seq by setting trunc_len to a value in [l, r]
                trunc_len = random.randint(
                    max(round(self.crop_ratio[0] * feat_len), 1),
                    min(round(self.crop_ratio[1] * feat_len), feat_len),
                )
                # corner case
                if feat_len == trunc_len:
                    return feats, gt_segments, gt_labels

        # try a few times till a valid truncation with at least one action
        for _ in range(max_num_trials):
            # sample a random truncation of the video feats
            st = random.randint(0, feat_len - trunc_len)
            ed = st + trunc_len
            window = np.array([st, ed], dtype=np.float32)

            # compute the intersection between the sampled window and all segments
            window = np.repeat(window[None, :], num_segs, axis=0)
            left = np.maximum(window[:, 0] - offset, gt_segments[:, 0])
            right = np.minimum(window[:, 1] + offset, gt_segments[:, 1])
            inter = np.clip(right - left, a_min=0, a_max=None)
            area_segs = np.abs(gt_segments[:, 1] - gt_segments[:, 0])
            inter_ratio = inter / area_segs

            # only select those segments over the thresh
            seg_idx = inter_ratio >= self.trunc_thresh

            # with at least one action
            if seg_idx.sum().item() > 0:
                break

        feats = feats[st:ed]
        gt_segments = np.stack(
            (left[seg_idx], right[seg_idx]), axis=1
        )  # [N,2] in feature grids
        gt_segments = gt_segments - st  # shift the time stamps due to truncation
        gt_labels = gt_labels[seg_idx]  # [N]
        return feats, gt_segments, gt_labels

    def __call__(self, results):
        assert "total_frames" in results.keys(), "should have total_frames as a key"
        total_frames = results["total_frames"]
        fps = results["avg_fps"]

        if self.method == "resize":
            assert (
                "resize_length" in results.keys()
            ), "should have resize_length as a key"
            snippet_num = results["resize_length"]
            snippet_stride = total_frames / snippet_num
            snippet_center = np.arange(
                snippet_stride / 2 - 0.5,
                total_frames + snippet_stride / 2 - 0.5,
                snippet_stride,
            )
            masks = torch.ones(results["resize_length"]).bool()

            # don't forget to resize the ground truth segments
            if "gt_segments" in results.keys():
                # convert gt seconds to feature grid
                results["gt_segments"] = np.clip(
                    results["gt_segments"] / results["duration"], 0.0, 1.0
                )
                results["gt_segments"] *= results["resize_length"]

        elif self.method == "random_trunc":
            snippet_num = self.trunc_len
            snippet_center = np.arange(0, total_frames, results["snippet_stride"])

            # trunc the snippet_center
            snippet_center, gt_segments, gt_labels = self.random_trunc(
                snippet_center,
                trunc_len=snippet_num,
                gt_segments=results["gt_segments"],
                gt_labels=results["gt_labels"],
            )

            # update the gt_segments
            results["gt_segments"] = gt_segments
            results["gt_labels"] = gt_labels

            # pad the snippet_center
            if len(snippet_center) < snippet_num:
                valid_len = len(snippet_center)
                snippet_center = np.pad(
                    snippet_center, (0, snippet_num - valid_len), mode="edge"
                )
                masks = torch.cat(
                    [torch.ones(valid_len), torch.zeros(snippet_num - valid_len)]
                ).bool()
            else:
                masks = torch.ones(snippet_num).bool()

        elif self.method == "sliding_window":
            snippet_num = results["window_size"]
            snippet_center = np.arange(0, total_frames, results["snippet_stride"])

            start_idx = min(results["feature_start_idx"], len(snippet_center))
            end_idx = min((results["feature_end_idx"] + 1), len(snippet_center))

            snippet_center = snippet_center[start_idx:end_idx]

            if len(snippet_center) < snippet_num:
                valid_len = len(snippet_center)
                snippet_center = np.pad(
                    snippet_center, (0, snippet_num - valid_len), mode="edge"
                )
                masks = torch.cat(
                    [torch.ones(valid_len), torch.zeros(snippet_num - valid_len)]
                ).bool()
            else:
                masks = torch.ones(snippet_num).bool()
        elif self.method == "padding":
            raise NotImplementedError

        # extend snippet center to a clip
        clip_idxs = np.arange(-(self.clip_len // 2), self.clip_len // 2)
        frame_idxs = (
            snippet_center[:, None] + self.frame_interval * clip_idxs[None, :]
        )  # [snippet_num, clip_len]

        # truncate to [0, total_frames-1], and round to int
        frame_idxs = np.clip(frame_idxs, 0, total_frames - 1).round()

        assert (
            frame_idxs.shape[0] == snippet_num
        ), "snippet center number should be equal to snippet number"
        assert (
            frame_idxs.shape[1] == self.clip_len
        ), "snippet length should be equal to clip length"

        results["frame_inds"] = frame_idxs.astype(int)
        results["num_clips"] = snippet_num
        results["clip_len"] = self.clip_len
        results["masks"] = masks
        return results


@PIPELINES.register_module()
class LoadFrames:
    def __init__(
        self,
        num_clips=1,
        scale_factor=1,
        method="resize",
        trunc_len=None,
        trunc_thresh=None,
        crop_ratio=None,
    ):
        self.num_clips = num_clips
        self.scale_factor = (
            scale_factor  # multiply by the frame number, if backbone has downsampling
        )
        self.method = method  # resize or padding or random_trunc or sliding_window
        # random_trunc settings
        self.trunc_len = trunc_len
        self.trunc_thresh = trunc_thresh
        self.crop_ratio = crop_ratio

    def random_trunc(
        self,
        feats,
        trunc_len,
        gt_segments,
        gt_labels,
        keypoints=None,
        offset=0,
        max_num_trials=200,
    ):
        feat_len = feats.shape[0]
        num_segs = gt_segments.shape[0]

        if feat_len <= trunc_len:
            if self.crop_ratio is None:
                return (
                    feats,
                    gt_segments,
                    gt_labels,
                    keypoints,
                )
            else:
                trunc_len = random.randint(
                    max(round(self.crop_ratio[0] * feat_len), 1),
                    min(round(self.crop_ratio[1] * feat_len), feat_len),
                )
                if feat_len == trunc_len:
                    return (
                        feats,
                        gt_segments,
                        gt_labels,
                        keypoints,
                    )

        for _ in range(max_num_trials):
            st = random.randint(0, feat_len - trunc_len)
            ed = st + trunc_len
            window = np.array([st, ed], dtype=np.float32)
            window = np.repeat(window[None, :], num_segs, axis=0)
            left = np.maximum(window[:, 0] - offset, gt_segments[:, 0])
            right = np.minimum(window[:, 1] + offset, gt_segments[:, 1])
            inter = np.clip(right - left, a_min=0, a_max=None)
            area_segs = np.abs(gt_segments[:, 1] - gt_segments[:, 0])
            if area_segs.sum() == 0:
                continue
            inter_ratio = inter / (area_segs + 1e-6)
            seg_idx = inter_ratio >= self.trunc_thresh
            if seg_idx.sum().item() > 0:
                break

        feats = feats[st:ed]
        if keypoints is not None:
            if feats.max() > keypoints.shape[0] - 1:
                # repeat last keypoint to get keypoints.shape[0] == feats.max() + 1
                pad_len = int(feats.max() + 1 - keypoints.shape[0])
                if pad_len > 0:
                    last_keypoint = keypoints[-1:, :, :]
                    keypoints = np.concatenate(
                        [keypoints] + [last_keypoint] * pad_len, axis=0
                    )
            keypoints = keypoints[feats, :, :]

        gt_segments = np.stack((left[seg_idx], right[seg_idx]), axis=1)
        gt_segments = gt_segments - st
        gt_labels = gt_labels[seg_idx]

        return feats, gt_segments, gt_labels, keypoints

    def __call__(self, results):
        assert "total_frames" in results.keys(), "should have total_frames as a key"
        total_frames = results["total_frames"]
        if self.method == "resize":
            assert (
                "resize_length" in results.keys()
            ), "should have resize_length as a key"
            frame_num = results["resize_length"] * self.scale_factor
            frame_stride = total_frames / frame_num
            frame_idxs = np.arange(
                frame_stride / 2 - 0.5,
                total_frames + frame_stride / 2 - 0.5,
                frame_stride,
            )
            masks = torch.ones(
                results["resize_length"]
            ).bool()  # should not multiply by scale_factor

            # don't forget to resize the ground truth segments
            if "gt_segments" in results.keys():
                # convert gt seconds to feature grid
                results["gt_segments"] = np.clip(
                    results["gt_segments"] / results["duration"], 0.0, 1.0
                )
                results["gt_segments"] *= results["resize_length"]

        elif self.method == "random_trunc":
            assert (
                results["snippet_stride"] >= self.scale_factor
            ), "snippet_stride should be larger than scale_factor"
            assert (
                results["snippet_stride"] % self.scale_factor == 0
            ), "snippet_stride should be divisible by scale_factor"

            frame_num = self.trunc_len * self.scale_factor
            frame_stride = results["snippet_stride"] // self.scale_factor
            frame_idxs = np.arange(0, total_frames, frame_stride)
            keypoints = results.get("keypoint")
            # trunc the frame_idxs
            frame_idxs, gt_segments, gt_labels, keypoints = self.random_trunc(
                frame_idxs,
                trunc_len=frame_num,
                gt_segments=results["gt_segments"]
                * self.scale_factor,  # gt segment should be mapped to frame level
                gt_labels=results["gt_labels"],
                keypoints=keypoints,
            )
            results["gt_segments"] = (
                gt_segments / self.scale_factor
            )  # convert back to original scale
            results["gt_labels"] = gt_labels
            if keypoints is not None:
                results["keypoint"] = keypoints

            # pad the frame_idxs
            if len(frame_idxs) < frame_num:
                valid_len = len(frame_idxs) // self.scale_factor
                frame_idxs = np.pad(
                    frame_idxs, (0, frame_num - len(frame_idxs)), mode="edge"
                )
                masks = torch.cat(
                    [torch.ones(valid_len), torch.zeros(self.trunc_len - valid_len)]
                ).bool()
            else:
                masks = torch.ones(self.trunc_len).bool()

        elif self.method == "sliding_window":
            assert (
                results["snippet_stride"] >= self.scale_factor
            ), "snippet_stride should be larger than scale_factor"
            assert (
                results["snippet_stride"] % self.scale_factor == 0
            ), "snippet_stride should be divisible by scale_factor"

            window_size = results["window_size"]
            frame_num = window_size * self.scale_factor
            frame_stride = results["snippet_stride"] // self.scale_factor
            frame_idxs = np.arange(0, total_frames, frame_stride)

            start_idx = min(
                results["feature_start_idx"] * self.scale_factor, len(frame_idxs)
            )
            end_idx = min(
                (results["feature_end_idx"] + 1) * self.scale_factor, len(frame_idxs)
            )

            frame_idxs = frame_idxs[start_idx:end_idx]

            if len(frame_idxs) < frame_num:
                valid_len = len(frame_idxs) // self.scale_factor
                frame_idxs = np.pad(
                    frame_idxs, (0, frame_num - len(frame_idxs)), mode="edge"
                )
                masks = torch.cat(
                    [torch.ones(valid_len), torch.zeros(window_size - valid_len)]
                ).bool()
            else:
                masks = torch.ones(window_size).bool()

        elif self.method == "padding":
            raise NotImplementedError

        # truncate to [0, total_frames-1], and round to int
        frame_idxs = np.clip(frame_idxs, 0, total_frames - 1).round()

        assert (
            frame_idxs.shape[0] == frame_num
        ), "snippet center number should be equal to snippet number"

        results["frame_inds"] = frame_idxs.astype(int)
        results["num_clips"] = self.num_clips
        results["clip_len"] = frame_num // self.num_clips
        results["masks"] = masks

        # Shift and Scale GT Tubes to match the local tensor coordinates
        if (
            "gt_tubes" in results
            and len(frame_idxs) > 0
            and self.method
            != "sliding_window"  # Removed enabling shifting for sliding_window too
        ):
            start_frame = frame_idxs[0]
            # Estimate stride from frame_idxs
            if len(frame_idxs) > 1:
                # Assuming constant stride, taking first step
                # Note: frame_idxs might be padded, but usually padding is at end
                # If padding is at start? 'edge' padding.
                # But frame_idxs calculation uses arange.
                frame_stride = frame_idxs[1] - frame_idxs[0]
                if frame_stride == 0:
                    # Should not happen unless stride=0 or repeated frames
                    frame_stride = 1
            else:
                frame_stride = 1

            # Ensure stride is at least 1
            frame_stride = max(int(frame_stride), 1)

            new_tubes = []
            for tube in results["gt_tubes"]:
                frames = np.array(tube.get("frames", []), dtype=np.float32)
                boxes = np.array(tube.get("boxes", []), dtype=np.float32)

                if len(frames) == 0:
                    continue

                # Shift to 0-based relative to crop
                frames = frames - start_frame

                # Scale to feature/tensor index
                # frame_idxs are [0, 4, 8] -> indices [0, 1, 2]
                frames = frames / frame_stride

                # Filter valid frames within the [0, frame_num) window
                # frame_num is the target length (e.g. 384)
                valid = (frames >= 0) & (frames < frame_num)

                if valid.any():
                    # We usually keep the tube structure, but update frames/boxes
                    new_tube = copy.deepcopy(tube)
                    new_tube["frames"] = frames[valid].tolist()
                    new_tube["boxes"] = boxes[valid].tolist()
                    new_tubes.append(new_tube)

            results["gt_tubes"] = new_tubes

            if len(new_tubes) > 0 and False:
                try:
                    import cv2
                    import decord

                    decord.bridge.set_bridge("native")

                    debug_dir = "vis_loadframes"
                    os.makedirs(debug_dir, exist_ok=True)

                    vid_path = results["filename"]
                    vr = decord.VideoReader(vid_path)

                    # We need to visualize frames corresponding to frame_idxs
                    # frame_idxs: [0, 4, 8, ...] absolute indices

                    # Visualize a subset of frames (e.g. every 10th)
                    vis_step = 1
                    # Clip frame_idxs to valid range
                    valid_frame_idxs = np.clip(frame_idxs, 0, len(vr) - 1).astype(int)

                    for t in range(0, len(valid_frame_idxs), vis_step):
                        abs_idx = valid_frame_idxs[t]
                        img = vr[abs_idx].asnumpy()
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        H, W = img.shape[:2]

                        # Draw tubes active at this time 't' (feature index)
                        for tube_idx, tube in enumerate(new_tubes):
                            frames = np.array(
                                tube.get("frames", [])
                            )  # Feature indices (float)
                            boxes = np.array(tube.get("boxes", []))

                            # Find if t is in frames
                            indices = np.where(np.abs(frames - t) < 0.1)[0]
                            for idx in indices:
                                box = boxes[idx]
                                x1, y1, x2, y2 = box
                                # Normalized [0,1] or absolute?
                                # Currently generic 'boxes' usually normalized if floats < 1.0?
                                # Check multisports_GT_local.pkl format or logic?
                                # But let's assume normalized if max < 2.0
                                if np.max(boxes) <= 1.5:
                                    x1, x2 = x1 * W, x2 * W
                                    y1, y2 = y1 * H, y2 * H

                                cv2.rectangle(
                                    img,
                                    (int(x1), int(y1)),
                                    (int(x2), int(y2)),
                                    (0, 255, 0),
                                    2,
                                )
                                cv2.putText(
                                    img,
                                    f"T{tube_idx}",
                                    (int(x1), int(y1) - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (0, 255, 0),
                                    1,
                                )

                        vid_name = os.path.basename(results["filename"])
                        save_path = os.path.join(
                            debug_dir, f"{vid_name}_t{t:03d}_abs{abs_idx}.jpg"
                        )
                        cv2.imwrite(save_path, img)

                except Exception as e:
                    print(f"Debug Vis Error: {e}")

        return results


@PIPELINES.register_module()
class Interpolate:
    def __init__(self, keys, size=128, mode="linear"):
        self.keys = keys
        self.size = size
        self.mode = mode

    def __call__(self, results):
        for key in self.keys:
            feat = results[key]
            if feat.ndim == 5:
                # Handle 5D input (N, C, T, H, W)
                # If size is int, assume it applies to T only, keeping H and W
                t, h, w = feat.shape[2:]
                target_size = self.size
                if isinstance(target_size, int):
                    target_size = (target_size, h, w)

                # Use trilinear for 5D as linear is not supported
                mode = self.mode
                if mode == "linear":
                    mode = "trilinear"

                if (t, h, w) != target_size:
                    orig_dtype = feat.dtype
                    results[key] = F.interpolate(
                        feat.float(),
                        size=target_size,
                        mode=mode,
                        align_corners=(
                            False if mode != "nearest" and mode != "area" else None
                        ),
                    ).to(orig_dtype)
            elif results[key].shape[2:] != self.size:
                feat = results[key]
                orig_dtype = feat.dtype
                results[key] = F.interpolate(
                    feat.float(),
                    size=self.size,
                    mode=self.mode,
                    align_corners=False,
                ).to(orig_dtype)
        return results
