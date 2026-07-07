#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@upm.es'

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
import torch
import numpy as np
from mmengine.dataset import Compose
from opentad.utils import set_seed
from opentad.datasets import ThumosSlidingDataset
from opentad.datasets.builder import collate
from images_framework.src.recognition import Recognition
set_seed(42)
np.random.seed(42)


class ECCV26TAD(Recognition):
    """
    SV-TAD: Native Sparse Convolutions for Efficient Temporal Action Detection
    """
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.model = None
        self.device = None
        self.ckpt = None
        self.thresh = None
        self.topk = None

    def parse_options(self, params):
        super().parse_options(params)
        import argparse
        from mmengine.config import Config
        parser = argparse.ArgumentParser(prog='ECCV26TAD', add_help=False)
        parser.add_argument('--gpu', dest='gpu', type=int, action='append',
                            help='GPU ID (negative value indicates CPU).')
        parser.add_argument('--config', metavar='FILE', type=str, 
                            help='Path to config file.')
        parser.add_argument('--ckpt', type=str, default='none', 
                            help="The checkpoint path")
        parser.add_argument('--thresh', type=float, default=0.0,
                            help='Only show predictions with score above this threshold')
        parser.add_argument('--topk', type=int, default=20,
                            help='Show at most this many predictions (sorted by score)')
        args, unknown = parser.parse_known_args(params)
        print(parser.format_usage())
        self.cfg = Config.fromfile(args.config)
        mode_gpu = torch.cuda.is_available() and -1 not in args.gpu
        self.device = torch.device('cuda' if mode_gpu else 'cpu')
        self.ckpt = args.ckpt
        self.thresh = args.thresh
        self.topk = args.topk

    def train(self, anns_train, anns_valid):
        print('Training model')

    def load(self, mode):
        import torchinfo
        from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
        from images_framework.src.constants import Modes
        from opentad.models import build_detector
        from opentad.utils import remap_legacy_sparse_conv_weights
        # Set up a neural network to train
        model_path = self.path + 'data/'
        print('Loading model from {}'.format(model_path))
        self.cfg.model['backbone']['custom']['pretrain'] = model_path + self.cfg.model['backbone']['custom']['pretrain']
        self.model = build_detector(self.cfg.model)
        # torchinfo.summary(self.model, input_size=(self.batch_size, 3, self.width, self.height), depth=5, device=self.device.type, col_names=['input_size', 'output_size', 'num_params', 'kernel_size'])
        if mode is Modes.TEST:
            self.model = self.model.to(self.device)
            # Load checkpoint (args -> config -> best)
            if self.ckpt != "none":
                checkpoint_path = self.ckpt
            elif "test_epoch" in self.cfg.inference.keys():
                checkpoint_path = os.path.join(self.cfg.work_dir, f"checkpoint/epoch_{self.cfg.inference.test_epoch}.pth")
            else:
                checkpoint_path = os.path.join(self.cfg.work_dir, "checkpoint/best.pth")
            print(f"Loading checkpoint from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            print(f"Checkpoint is epoch {checkpoint.get('epoch', 'unknown')}.")
            # Model EMA
            use_ema = getattr(self.cfg.solver, "ema", False)
            state_dict = checkpoint["state_dict_ema"] if use_ema else checkpoint["state_dict"]
            consume_prefix_in_state_dict_if_present(state_dict, prefix="module.")
            state_dict = remap_legacy_sparse_conv_weights(state_dict, self.model.state_dict())
            self.model.load_state_dict(state_dict)
            if use_ema:
                print("Using Model EMA...")
            self.model.eval()

    def process(self, ann, pred):
        def load_ground_truth(ann_file):
            """
            Read the annotations of a single video directly from the matching JSON file.
            """
            import json
            with open(ann_file, "r", encoding="utf-8") as ifs:
                payload = json.load(ifs)
            video_entries = payload.get("video", [])
            annotations = payload.get("annotations", [])
            class_map = payload.get("class_map", [])
            if isinstance(video_entries, list) and video_entries:
                video_info = video_entries[0]
                video_info = dict(video_info)
                video_info.setdefault("duration", payload.get("duration"))
                video_info.setdefault("frame", payload.get("frame"))
                video_info.setdefault("subset", payload.get("database"))
            else:
                video_info = None
            gt = []
            for anno in annotations:
                if anno.get("label") == "Ambiguous":
                    continue
                gt.append(dict(segment=anno["segment"], label=anno["label"]))
            gt.sort(key=lambda x: x["segment"][0])
            if video_info is None and not gt:
                return None, [], class_map
            return video_info, gt, class_map
        
        def _probe_video_frame_duration(video_path):
            """
            Fallback to read frame count and duration directly from the video file when
            they are missing from the annotation JSON.
            """
            import cv2
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                cap.release()
                raise RuntimeError(f"Cannot open video file: {video_path}")
            frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            cap.release()
            duration = frame / fps if fps > 0 else 0.0
            return frame, duration

        class ExampleSlidingDataset(ThumosSlidingDataset):
            """
            Single-video sliding-window dataset for inference on one example clip.

            It reuses the exact test pipeline / sliding-window splitting / __getitem__ of
            ``ThumosSlidingDataset`` (frame decoding, resize, center crop, normalization,
            NCTHW formatting and the ``metas`` expected by the post-processing), but skips
            reading the whole THUMOS annotation database: the single video info comes
            straight from the matching JSON file (see ``load_ground_truth``).
            """

            def __init__(self, video_name, video_info, data_path, pipeline, class_map,
                        window_size, feature_stride=4, sample_stride=1,
                        window_overlap_ratio=0.5):
                # NOTE: we intentionally do NOT call super().__init__(), because the base
                # SlidingWindowDataset.__init__ reads the full annotation database. Instead
                # we set up only the attributes needed by split_video_to_windows /
                # __getitem__ / the pipeline for this single example video.
                self.data_path = data_path
                self.block_list = None
                self.ann_file = None
                self.subset_name = None
                self.logger = print
                self.class_map = class_map
                self.class_agnostic = False
                self.filter_gt = False
                self.test_mode = True
                self.pipeline = Compose(pipeline)

                # feature settings
                self.feature_stride = int(feature_stride)
                self.sample_stride = int(sample_stride)
                self.offset_frames = 0
                self.snippet_stride = int(feature_stride * sample_stride)
                self.fps = -1

                # window settings
                self.window_size = int(window_size)
                self.window_stride = int(window_size * (1 - window_overlap_ratio))
                self.ioa_thresh = 0.75
                self.video_split_ratio = None

                # thumos-specific attributes (keypoints are unused for the example)
                self.skeleton_data_path_2d = None
                self.preprocessed_skeleton_path = None
                self.skeleton_cache = {}
                self.debug = False
                self._aligned_cache = {}
                self._file_exists_cache = {}

                # build the sliding windows for this single video (test_mode -> no gt)
                self.data_list = self.split_video_to_windows(video_name, video_info, {})

        def build_example_dataset(cfg, input_data, video_info):
            """
            Build a dataset containing only the requested example video, reusing the real
            sliding-window test pipeline defined in the config.
            """
            test_cfg = cfg.dataset.test

            video_name = os.path.splitext(os.path.basename(input_data))[0]
            data_path = os.path.dirname(os.path.abspath(input_data))

            # ensure frame count / duration are available for the sliding-window splitting
            video_info = dict(video_info or {})
            if not video_info.get("frame") or not video_info.get("duration"):
                frame, duration = _probe_video_frame_duration(input_data)
                video_info.setdefault("frame", frame)
                video_info.setdefault("duration", duration)
                if not video_info.get("frame"):
                    video_info["frame"] = frame
                if not video_info.get("duration"):
                    video_info["duration"] = duration

            return ExampleSlidingDataset(
                video_name=video_name,
                video_info=video_info,
                data_path=data_path,
                pipeline=test_cfg.pipeline,
                class_map=[],  # unused in test_mode; external classifier is passed to the model
                window_size=getattr(test_cfg, "window_size", 768),
                feature_stride=getattr(test_cfg, "feature_stride", 4),
                sample_stride=getattr(test_cfg, "sample_stride", 1),
                window_overlap_ratio=getattr(test_cfg, "window_overlap_ratio", 0.5),
            )

        def nms_single_video(predictions, nms_cfg):
            """
            Apply the same NMS used for sliding-window evaluation, but for one video.
            """
            from opentad.models.utils.post_processing import batched_nms
            segments = torch.Tensor([data["segment"] for data in predictions])
            scores = torch.Tensor([data["score"] for data in predictions])
            class_idx = []
            labels = []
            for data in predictions:
                if data["label"] not in class_idx:
                    class_idx.append(data["label"])
                labels.append(class_idx.index(data["label"]))
            labels = torch.Tensor(labels)
            segments, scores, labels = batched_nms(segments, scores, labels, **nms_cfg)
            results = []
            for segment, label, score in zip(segments, labels, scores):
                results.append(dict(segment=[round(seg.item(), 2) for seg in segment], label=class_idx[int(label.item())], score=round(score.item(), 4),))
            return results

        input_data = ann.images[0].filename
        video_info, ground_truth, external_cls = load_ground_truth(os.path.splitext(input_data)[0]+'.json')
        # the external classifier maps predicted class indices -> category names, so it
        # must list ALL training classes in the same (sorted) order used at training time
        num_classes = self.cfg.model["rpn_head"]["num_classes"]
        if len(external_cls) != num_classes:
            raise ValueError(
                f"'class_map' in the example JSON has {len(external_cls)} entries but the "
                f"model predicts {num_classes} classes. It must list all {num_classes} "
                f"training classes in sorted order (see category_idx.txt used at training)."
            )

        # build a dataset containing only the requested example video, reusing the
        # real sliding-window test pipeline from the config
        test_dataset = build_example_dataset(self.cfg, input_data, video_info)
        print(f"Loaded example video '{os.path.basename(input_data)}' as {len(test_dataset)} window(s).")

        use_amp = getattr(self.cfg.solver, "amp", False)

        # this is a sliding window dataset, so NMS is applied after merging windows
        self.cfg.post_processing.sliding_window = True

        # inference, window by window
        # model.eval()
        result_dict = {}
        print("Running inference...")
        for index in range(len(test_dataset)):
            data_dict = collate([test_dataset[index]])
            data_dict["inputs"] = data_dict["inputs"].to(self.device)
            data_dict["masks"] = data_dict["masks"].to(self.device)

            with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_amp):
                with torch.no_grad():
                    results = self.model(
                        **data_dict,
                        return_loss=False,
                        infer_cfg=self.cfg.inference,
                        post_cfg=self.cfg.post_processing,
                        ext_cls=external_cls,
                    )

            for k, v in results.items():
                if k in result_dict:
                    result_dict[k].extend(v)
                else:
                    result_dict[k] = v

        # merge windows with NMS (same as sliding-window evaluation)
        video_name = os.path.splitext(os.path.basename(input_data))[0]
        predictions = result_dict.get(video_name, result_dict.get(os.path.basename(input_data), result_dict.get(input_data, [])))
        if len(predictions) > 0 and self.cfg.post_processing.nms is not None:
            predictions = nms_single_video(predictions, dict(self.cfg.post_processing.nms))
        predictions.sort(key=lambda x: x["score"], reverse=True)

        # ---- report ----
        print("\n" + "=" * 70)
        print(f"VIDEO: {input_data}")
        if video_info is not None:
            print(f"  duration: {video_info.get('duration', '?')} s | frames: {video_info.get('frame', '?')}")
        print("=" * 70)

        print(f"\nGROUND TRUTH ({len(ground_truth)} segments):")
        if len(ground_truth) == 0:
            print("  (no ground truth annotations found for this video)")
        else:
            print(f"  {'start':>8}  {'end':>8}  label")
            print(f"  {'-'*8}  {'-'*8}  {'-'*20}")
            for gt in ground_truth:
                s, e = gt["segment"]
                print(f"  {s:>8.2f}  {e:>8.2f}  {gt['label']}")

        shown = [p for p in predictions if p["score"] >= self.thresh]
        if self.topk >= 0:
            shown = shown[: self.topk]
        print(
            f"\nPREDICTIONS (showing {len(shown)} of {len(predictions)}"
            f"{f', score >= {self.thresh}' if self.thresh > 0 else ''}):"
        )
        if len(shown) == 0:
            print("  (no predictions)")
        else:
            print(f"  {'start':>8}  {'end':>8}  {'score':>7}  label")
            print(f"  {'-'*8}  {'-'*8}  {'-'*7}  {'-'*20}")
            for p in shown:
                s, e = p["segment"]
                print(f"  {s:>8.2f}  {e:>8.2f}  {p['score']:>7.4f}  {p['label']}")
