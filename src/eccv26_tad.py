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
        from opentad.datasets import build_dataset, build_dataloader
        from opentad.cores import eval_one_epoch

        # build dataset
        test_dataset = build_dataset(self.cfg.dataset.test)
        test_loader = build_dataloader(test_dataset, rank=self.rank, world_size=self.world_size, shuffle=False, drop_last=False, **self.cfg.solver.test)
        print(f"Loaded video '{os.path.basename(os.path.basename(pred.filename))}' natively as {len(test_dataset)} window(s).")

        # AMP: automatic mixed precision
        use_amp = getattr(self.cfg.solver, "amp", False)
        if use_amp:
            print("Using Automatic Mixed Precision...")

        print("Testing Starts...\n")
        eval_one_epoch(test_loader, self.model, self.cfg, print, self.rank, model_ema=None, use_amp=use_amp, world_size=self.world_size, not_eval=True)
        print("Testing Over...\n")
