#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Ricardo Pizarro'
__email__ = 'ricardo.pizarroc@edu.uah.es'

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
import numpy as np
from opentad.utils import set_seed
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
        self.gpu = None
        self.ckpt = None
        self.classes = None
        self.rank = int(os.environ.get('RANK', 0))
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))

    def parse_options(self, params):
        super().parse_options(params)
        import argparse
        from mmengine.config import Config
        from images_framework.categories.actions import Action as Oa
        parser = argparse.ArgumentParser(prog='ECCV26TAD', add_help=False)
        parser.add_argument('--gpu', dest='gpu', type=int, default=-1,
                            help='GPU ID (negative value indicates CPU).')
        parser.add_argument('--config', metavar='FILE', type=str, 
                            help='Path to config file.')
        parser.add_argument('--ckpt', type=str, default='none', 
                            help='The checkpoint path.')
        args, unknown = parser.parse_known_args(params)
        print(parser.format_usage())
        self.cfg = Config.fromfile(args.config)
        self.gpu = args.gpu
        self.ckpt = args.ckpt
        self.classes = {0: "BaseballPitch", 1: "BasketballDunk", 2: "Billiards", 3: "CleanAndJerk", 4: "CliffDiving", 5: "CricketBowling", 6: "CricketShot", 7: "Diving", 8: "FrisbeeCatch", 9: "GolfSwing", 10: "HammerThrow", 11: "HighJump", 12: "JavelinThrow", 13: "LongJump", 14: "PoleVault", 15: "Shotput", 16: "SoccerPenalty", 17: "TennisSwing", 18: "ThrowDiscus", 19: "VolleyballSpiking"}

    def train(self, anns_train, anns_valid):
        print('Training model')

    def load(self, mode):
        import torch
        import torchinfo
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel
        from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
        from images_framework.src.constants import Modes
        from opentad.models import build_detector
        from opentad.utils import remap_legacy_sparse_conv_weights
        # DDP initialization for torchrun execution
        torchrun_mode = 'LOCAL_RANK' in os.environ and 'RANK' in os.environ and 'WORLD_SIZE' in os.environ
        if torchrun_mode:
            print(f'Distributed init (rank {self.rank}/{self.world_size}, local rank {self.local_rank})')
            torch.cuda.set_device(self.local_rank)
            dist.init_process_group('nccl', rank=self.rank, world_size=self.world_size)
            self.device = torch.device(f'cuda:{self.local_rank}')
        # Single GPU or CPU mode
        else:
            self.local_rank = self.gpu
            self.device = torch.device(f'cuda:{self.local_rank}' if torch.cuda.is_available() and self.gpu >= 0 else 'cpu')
            print(f'Using {self.device} device')
        # Set up a neural network to train
        model_path = self.path + 'data/'
        print('Loading model from {}'.format(model_path))
        self.cfg.model['backbone']['custom']['pretrain'] = model_path + self.cfg.model['backbone']['custom']['pretrain']
        self.model = build_detector(self.cfg.model)
        self.model = self.model.to(self.device)
        # DDP
        if torchrun_mode:
            self.model = DistributedDataParallel(self.model, device_ids=[self.local_rank], output_device=self.local_rank)
            print(f'Using DDP with total {self.world_size} GPUS...')
        # [batch, num_clips, channels, T, H, W]
        window_size = self.cfg.window_size
        img_size = self.cfg.model['backbone']['backbone']['img_size']
        torchinfo.summary(self.model.module.backbone if torchrun_mode else self.model.backbone, input_size=(1, 1, 3, window_size, img_size, img_size), depth=5, device=self.device, col_names=['input_size', 'output_size', 'num_params', 'kernel_size'])
        if mode is Modes.TEST:
            # Load checkpoint (args -> config -> best)
            if self.ckpt != 'none':
                checkpoint_path = self.ckpt
            elif 'test_epoch' in self.cfg.inference.keys():
                checkpoint_path = model_path + f'checkpoint/epoch_{self.cfg.inference.test_epoch}.pth'
            else:
                checkpoint_path = model_path + 'checkpoint/best.pth'
            print(f'Loading checkpoint from: {checkpoint_path}')
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            print(f'Checkpoint is epoch {checkpoint.get("epoch", "unknown")}.')
            # Model EMA
            use_ema = getattr(self.cfg.solver, "ema", False)
            state_dict = checkpoint["state_dict_ema"] if use_ema else checkpoint["state_dict"]
            # Older checkpoints may or may not carry the DDP "module." prefix depending on
            # how EMA/the model were wrapped at save time; normalize before comparing/loading.
            consume_prefix_in_state_dict_if_present(state_dict, prefix="module.")
            state_dict = remap_legacy_sparse_conv_weights(state_dict, self.model.module.state_dict() if torchrun_mode else self.model.state_dict())
            self.model.module.load_state_dict(state_dict, strict=False) if torchrun_mode else self.model.load_state_dict(state_dict)
            if use_ema:
                print("Using Model EMA...")
            self.model.eval()

    def process(self, ann, pred):
        # def _probe_video_frame_duration(video_path):
        #     """
        #     Fallback to read frame count and duration directly from the video file when
        #     they are missing from the annotation JSON.
        #     """
        #     import cv2
        #     cap = cv2.VideoCapture(video_path)
        #     if not cap.isOpened():
        #         cap.release()
        #         raise RuntimeError(f"Cannot open video file: {video_path}")
        #     frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        #     fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        #     cap.release()
        #     duration = frame / fps if fps > 0 else 0.0
        #     return frame, duration

        # class ExampleSlidingDataset(ThumosSlidingDataset):
        #     """
        #     Single-video sliding-window dataset for inference on one example clip.

        #     It reuses the exact test pipeline / sliding-window splitting / __getitem__ of
        #     ``ThumosSlidingDataset`` (frame decoding, resize, center crop, normalization,
        #     NCTHW formatting and the ``metas`` expected by the post-processing), but skips
        #     reading the whole THUMOS annotation database: the single video info comes
        #     straight from the matching JSON file (see ``load_ground_truth``).
        #     """

        #     def __init__(self, video_name, video_info, data_path, pipeline, class_map,
        #                 window_size, feature_stride=4, sample_stride=1,
        #                 window_overlap_ratio=0.5):
        #         # NOTE: we intentionally do NOT call super().__init__(), because the base
        #         # SlidingWindowDataset.__init__ reads the full annotation database. Instead
        #         # we set up only the attributes needed by split_video_to_windows /
        #         # __getitem__ / the pipeline for this single example video.
        #         self.data_path = data_path
        #         self.block_list = None
        #         self.ann_file = None
        #         self.subset_name = None
        #         self.logger = print
        #         self.class_map = class_map
        #         self.class_agnostic = False
        #         self.filter_gt = False
        #         self.test_mode = True
        #         self.pipeline = Compose(pipeline)
        #         # feature settings
        #         self.feature_stride = int(feature_stride)
        #         self.sample_stride = int(sample_stride)
        #         self.offset_frames = 0
        #         self.snippet_stride = int(feature_stride * sample_stride)
        #         self.fps = -1
        #         # window settings
        #         self.window_size = int(window_size)
        #         self.window_stride = int(window_size * (1 - window_overlap_ratio))
        #         self.ioa_thresh = 0.75
        #         self.video_split_ratio = None
        #         # thumos-specific attributes (keypoints are unused for the example)
        #         self.skeleton_data_path_2d = None
        #         self.preprocessed_skeleton_path = None
        #         self.skeleton_cache = {}
        #         self.debug = False
        #         self._aligned_cache = {}
        #         self._file_exists_cache = {}
        #         # build the sliding windows for this single video (test_mode -> no gt)
        #         self.data_list = self.split_video_to_windows(video_name, video_info, {})

        # def build_example_dataset(cfg, input_data):
        #     """
        #     Build a dataset containing only the requested example video, reusing the real
        #     sliding-window test pipeline defined in the config.
        #     """
        #     test_cfg = cfg.dataset.test
        #     video_name = os.path.splitext(os.path.basename(input_data))[0]
        #     data_path = os.path.dirname(os.path.abspath(input_data))
        #     # ensure frame count / duration are available for the sliding-window splitting
        #     video_info = dict()
        #     frame, duration = _probe_video_frame_duration(input_data)
        #     video_info.setdefault("frame", frame)
        #     video_info.setdefault("duration", duration)
        #     return ExampleSlidingDataset(
        #         video_name=video_name,
        #         video_info=video_info,
        #         data_path=data_path,
        #         pipeline=test_cfg.pipeline,
        #         class_map=[],  # unused in test_mode; external classifier is passed to the model
        #         window_size=getattr(test_cfg, "window_size", 768),
        #         feature_stride=getattr(test_cfg, "feature_stride", 4),
        #         sample_stride=getattr(test_cfg, "sample_stride", 1),
        #         window_overlap_ratio=getattr(test_cfg, "window_overlap_ratio", 0.5),
        #     )

        # def nms_single_video(predictions, nms_cfg):
        #     """
        #     Apply the same NMS used for sliding-window evaluation, but for one video.
        #     """
        #     from opentad.models.utils.post_processing import batched_nms
        #     segments = torch.Tensor([data["segment"] for data in predictions])
        #     scores = torch.Tensor([data["score"] for data in predictions])
        #     class_idx = []
        #     labels = []
        #     for data in predictions:
        #         if data["label"] not in class_idx:
        #             class_idx.append(data["label"])
        #         labels.append(class_idx.index(data["label"]))
        #     labels = torch.Tensor(labels)
        #     segments, scores, labels = batched_nms(segments, scores, labels, **nms_cfg)
        #     results = []
        #     for segment, label, score in zip(segments, labels, scores):
        #         results.append(dict(segment=[round(seg.item(), 2) for seg in segment], label=class_idx[int(label.item())], score=round(score.item(), 4),))
        #     return results

        import torch
        from mmengine.dataset import Compose
        from opentad.datasets import ThumosSlidingDataset, build_dataset, build_dataloader
        from opentad.datasets.builder import collate
        from opentad.cores import eval_one_epoch
        from images_framework.src.annotations import TemporalCategory

        # build dataset
        test_dataset = build_dataset(self.cfg.dataset.test)
        test_loader = build_dataloader(test_dataset, rank=self.rank, world_size=self.world_size, shuffle=False, drop_last=False, **self.cfg.solver.test)

        # # build a dataset containing only the requested example video, reusing the
        # # real sliding-window test pipeline from the config
        # test_dataset = build_example_dataset(self.cfg, pred.filename)
        # print(f"Loaded example video '{os.path.basename(pred.filename)}' as {len(test_dataset)} window(s).")

        # AMP: automatic mixed precision
        use_amp = getattr(self.cfg.solver, "amp", False)
        if use_amp:
            print("Using Automatic Mixed Precision...")

        print("Testing Starts...\n")
        eval_one_epoch(test_loader, self.model, self.cfg, print, self.rank, model_ema=None, use_amp=use_amp, world_size=self.world_size, not_eval=False)
        print("Testing Over...\n")

        # # this is a sliding window dataset, so NMS is applied after merging windows
        # self.cfg.post_processing.sliding_window = True

        # # inference, window by window
        # # model.eval()
        # result_dict = {}
        # print("Running inference...")
        # for index in range(len(test_dataset)):
        #     data_dict = collate([test_dataset[index]])
        #     data_dict["inputs"] = data_dict["inputs"].to(self.device)
        #     data_dict["masks"] = data_dict["masks"].to(self.device)

        #     with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_amp):
        #         with torch.no_grad():
        #             results = self.model(
        #                 **data_dict,
        #                 return_loss=False,
        #                 infer_cfg=self.cfg.inference,
        #                 post_cfg=self.cfg.post_processing,
        #                 ext_cls=list(self.classes.values()),
        #             )

        #     for k, v in results.items():
        #         if k in result_dict:
        #             result_dict[k].extend(v)
        #         else:
        #             result_dict[k] = v

        # # merge windows with NMS (same as sliding-window evaluation)
        # video_name = os.path.splitext(os.path.basename(pred.filename))[0]
        # predictions = result_dict.get(video_name, result_dict.get(os.path.basename(pred.filename), result_dict.get(pred.filename, [])))
        # if len(predictions) > 0 and self.cfg.post_processing.nms is not None:
        #     predictions = nms_single_video(predictions, dict(self.cfg.post_processing.nms))
        # predictions.sort(key=lambda x: x["score"], reverse=True)
        # for action in predictions:
        #     pred.add_action(TemporalCategory(label=action['label'], score=action['score'], segment=tuple(action['segment'])))
