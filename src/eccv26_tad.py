#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Ricardo Pizarro'
__email__ = 'ricardo.pizarroc@edu.uah.es'

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
import numpy as np
from mmengine.dataset import Compose
from opentad.utils import set_seed
from opentad.datasets import ThumosSlidingDataset
from images_framework.src.recognition import Recognition
set_seed(42)
np.random.seed(42)


class ExampleSlidingDataset(ThumosSlidingDataset):
    """
    Single-video sliding-window dataset, reuses real test pipeline
    """
    def __init__(self, video_name, video_info, data_path, pipeline, class_map, window_size, feature_stride=4, sample_stride=1, window_overlap_ratio=0.5):
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
        self.feature_stride = int(feature_stride)
        self.sample_stride = int(sample_stride)
        self.offset_frames = 0
        self.snippet_stride = int(feature_stride * sample_stride)
        self.fps = -1
        self.window_size = int(window_size)
        self.window_stride = int(window_size * (1 - window_overlap_ratio))
        self.ioa_thresh = 0.75
        self.video_split_ratio = None
        self.skeleton_data_path_2d = None
        self.preprocessed_skeleton_path = None
        self.skeleton_cache = {}
        self.debug = False
        self._aligned_cache = {}
        self._file_exists_cache = {}
        self.data_list = self.split_video_to_windows(video_name, video_info, {})


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
        self.cfg.work_dir = self.path
        self.cfg.post_processing.save_dict = True
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
        import cv2
        import json
        import torch.utils.data
        from opentad.cores import eval_one_epoch
        from opentad.datasets.builder import collate as default_collate
        from images_framework.src.annotations import TemporalCategory

        def _probe_video_frame_duration(video_path):
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                cap.release()
                raise RuntimeError(f"Cannot open video file: {video_path}")
            frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            cap.release()
            return frame, (frame / fps if fps > 0 else 0.0)

        def collate_with_device(batch):
            data_dict = default_collate(batch)
            if isinstance(data_dict["inputs"], torch.Tensor):
                data_dict["inputs"] = data_dict["inputs"].to(self.device)
            if isinstance(data_dict["masks"], torch.Tensor):
                data_dict["masks"] = data_dict["masks"].to(self.device)
            return data_dict

        # Build dataset with only the video from pred.filename
        test_cfg = self.cfg.dataset.test
        video_name = os.path.splitext(os.path.basename(pred.filename))[0]
        data_path = os.path.dirname(os.path.abspath(pred.filename))
        video_info = {}
        frame, duration = _probe_video_frame_duration(pred.filename)
        video_info["frame"] = frame
        video_info["duration"] = duration
        test_dataset = ExampleSlidingDataset(video_name=video_name, video_info=video_info, data_path=data_path, pipeline=test_cfg.pipeline, class_map=list(self.classes.values()), window_size=getattr(test_cfg, "window_size", 768), feature_stride=getattr(test_cfg, "feature_stride", 4), sample_stride=getattr(test_cfg, "sample_stride", 1), window_overlap_ratio=getattr(test_cfg, "window_overlap_ratio", 0.5))
        # Build dataloader with custom collate that moves to device
        sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=False, drop_last=False)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.cfg.solver.test.get("batch_size", 1) // self.world_size, collate_fn=collate_with_device, sampler=sampler, num_workers=0, pin_memory=False)

        # AMP: automatic mixed precision
        use_amp = getattr(self.cfg.solver, "amp", False)
        eval_one_epoch(test_loader, self.model, self.cfg, print, self.rank, model_ema=None, use_amp=use_amp, world_size=self.world_size, not_eval=True)

        # Save prediction
        with open(os.path.join(self.cfg.work_dir, 'result_detection.json'), 'r') as ifs:
            data = json.load(ifs)
        for video_name, actions in data['results'].items():
            for action in actions:
                pred.add_action(TemporalCategory(label=action["label"], segment=tuple(action["segment"]), score=action["score"]))
