#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Ricardo Pizarro'
__email__ = 'ricardo.pizarroc@edu.uah.es'

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
import torch
import numpy as np
from mmengine.dataset import Compose
from opentad.utils import set_seed
from opentad.datasets.builder import DATASETS
from pcr_framework.src.recognition import Recognition
set_seed(42)
np.random.seed(42)


class ExampleDataset:
    """
    Factory that builds a single-video dataset by reusing the real registered dataset class (e.g. ThumosSlidingDataset, AnetResizeDataset)
    """
    def __new__(cls, video_name, video_info, data_path, pipeline, class_map, cfg_test):
        dataset_type = cfg_test.get('type', 'ThumosSlidingDataset')
        real_cls = DATASETS.get(dataset_type)
        if real_cls is None:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        # Create the real instance without running its __init__
        self = real_cls.__new__(real_cls)
        # Basic settings expected by the base classes / __getitem__
        self.data_path = data_path
        self.class_map = class_map
        self.block_list = None
        self.ann_file = None
        self.subset_name = None
        self.logger = print
        self.class_agnostic = False
        self.filter_gt = False
        self.test_mode = bool(getattr(cfg_test, 'test_mode', False))
        self.pipeline = Compose(pipeline)
        self.debug = False
        self.fps = -1
        self.video_split_ratio = None
        # Feature settings
        self.feature_stride = int(getattr(cfg_test, 'feature_stride', 4))
        self.sample_stride = int(getattr(cfg_test, 'sample_stride', 1))
        self.offset_frames = int(getattr(cfg_test, 'offset_frames', 0))
        self.snippet_stride = int(self.feature_stride * self.sample_stride)
        # Sliding window settings
        self.window_size = int(getattr(cfg_test, 'window_size', 768))
        self.window_stride = int(self.window_size * (1 - getattr(cfg_test, 'window_overlap_ratio', 0.5)))
        self.ioa_thresh = getattr(cfg_test, 'ioa_thresh', 0.75)
        # Resize settings
        self.resize_length = getattr(cfg_test, 'resize_length', 224)
        # Skeleton / keypoint settings (used by some __getitem__ implementations)
        self.skeleton_data_path_2d = getattr(cfg_test, 'skeleton_data_path_2d', None)
        self.preprocessed_skeleton_path = getattr(cfg_test, 'preprocessed_skeleton_path', None)
        self.skeleton_cache = {}
        self.skeleton_cache_2d = {}
        self.filenames_cache = {}
        self._aligned_cache = {}
        self._file_exists_cache = {}
        # Build the single-video data_list in the shape the class' __getitem__ expects.
        empty_gt = dict(gt_segments=np.empty((0, 2), dtype=np.float32), gt_labels=np.empty((0,), dtype=np.int32))
        video_anno = {} if self.test_mode else (self.get_gt(video_info) or empty_gt)
        is_sliding = 'Sliding' in dataset_type or 'Padding' in dataset_type
        if is_sliding:
            self.data_list = self.split_video_to_windows(video_name, video_info, video_anno)
        else:
            self.data_list = [(video_name, video_info, video_anno)]
        return self


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
        self.class_map = None
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
        parser.add_argument('--ckpt', type=str, default=None, 
                            help='The checkpoint path.')
        args, unknown = parser.parse_known_args(params)
        print(parser.format_usage())
        self.cfg = Config.fromfile(args.config)
        self.gpu = args.gpu
        self.ckpt = args.ckpt
        if self.database == 'thumos':
            self.class_map = ["BaseballPitch", "BasketballDunk", "Billiards", "CleanAndJerk", "CliffDiving", "CricketBowling", "CricketShot", "Diving", "FrisbeeCatch", "GolfSwing", "HammerThrow", "HighJump", "JavelinThrow", "LongJump", "PoleVault", "Shotput", "SoccerPenalty", "TennisSwing", "ThrowDiscus", "VolleyballSpiking"]
        elif self.database == 'anet':
            self.class_map = ["Applying sunscreen", "Archery", "Arm wrestling", "Assembling bicycle", "BMX", "Baking cookies", "Ballet", "Bathing dog", "Baton twirling", "Beach soccer", "Beer pong", "Belly dance", "Blow-drying hair", "Blowing leaves", "Braiding hair", "Breakdancing", "Brushing hair", "Brushing teeth", "Building sandcastles", "Bullfighting", "Bungee jumping", "Calf roping", "Camel ride", "Canoeing", "Capoeira", "Carving jack-o-lanterns", "Changing car wheel", "Cheerleading", "Chopping wood", "Clean and jerk", "Cleaning shoes", "Cleaning sink", "Cleaning windows", "Clipping cat claws", "Cricket", "Croquet", "Cumbia", "Curling", "Cutting the grass", "Decorating the Christmas tree", "Disc dog", "Discus throw", "Dodgeball", "Doing a powerbomb", "Doing crunches", "Doing fencing", "Doing karate", "Doing kickboxing", "Doing motocross", "Doing nails", "Doing step aerobics", "Drinking beer", "Drinking coffee", "Drum corps", "Elliptical trainer", "Fixing bicycle", "Fixing the roof", "Fun sliding down", "Futsal", "Gargling mouthwash", "Getting a haircut", "Getting a piercing", "Getting a tattoo", "Grooming dog", "Grooming horse", "Hammer throw", "Hand car wash", "Hand washing clothes", "Hanging wallpaper", "Having an ice cream", "High jump", "Hitting a pinata", "Hopscotch", "Horseback riding", "Hula hoop", "Hurling", "Ice fishing", "Installing carpet", "Ironing clothes", "Javelin throw", "Kayaking", "Kite flying", "Kneeling", "Knitting", "Laying tile", "Layup drill in basketball", "Long jump", "Longboarding", "Making a cake", "Making a lemonade", "Making a sandwich", "Making an omelette", "Mixing drinks", "Mooping floor", "Mowing the lawn", "Paintball", "Painting", "Painting fence", "Painting furniture", "Peeling potatoes", "Ping-pong", "Plastering", "Plataform diving", "Playing accordion", "Playing badminton", "Playing bagpipes", "Playing beach volleyball", "Playing blackjack", "Playing congas", "Playing drums", "Playing field hockey", "Playing flauta", "Playing guitarra", "Playing harmonica", "Playing ice hockey", "Playing kickball", "Playing lacrosse", "Playing piano", "Playing polo", "Playing pool", "Playing racquetball", "Playing rubik cube", "Playing saxophone", "Playing squash", "Playing ten pins", "Playing violin", "Playing water polo", "Pole vault", "Polishing forniture", "Polishing shoes", "Powerbocking", "Preparing pasta", "Preparing salad", "Putting in contact lenses", "Putting on makeup", "Putting on shoes", "Rafting", "Raking leaves", "Removing curlers", "Removing ice from car", "Riding bumper cars", "River tubing", "Rock climbing", "Rock-paper-scissors", "Rollerblading", "Roof shingle removal", "Rope skipping", "Running a marathon", "Sailing", "Scuba diving", "Sharpening knives", "Shaving", "Shaving legs", "Shot put", "Shoveling snow", "Shuffleboard", "Skateboarding", "Skiing", "Slacklining", "Smoking a cigarette", "Smoking hookah", "Snatch", "Snow tubing", "Snowboarding", "Spinning", "Spread mulch", "Springboard diving", "Starting a campfire", "Sumo", "Surfing", "Swimming", "Swinging at the playground", "Table soccer", "Tai chi", "Tango", "Tennis serve with ball bouncing", "Throwing darts", "Trimming branches or hedges", "Triple jump", "Tug of war", "Tumbling", "Using parallel bars", "Using the balance beam", "Using the monkey bar", "Using the pommel horse", "Using the rowing machine", "Using uneven bars", "Vacuuming floor", "Volleyball", "Wakeboarding", "Walking the dog", "Washing dishes", "Washing face", "Washing hands", "Waterskiing", "Waxing skis", "Welding", "Windsurfing", "Wrapping presents", "Zumba"]
        elif self.database == 'attach':
            self.class_map = ["board__hold__both", "board__hold__left", "board__hold__right", "board__lift__both", "board__lift__left", "board__lift__right", "board__move__both", "board__move__left", "board__move__right", "board__place__both", "board__place__left", "board__place__right", "board__plug__both", "board__plug__left", "board__plug__right", "board__rotate__both", "board__rotate__left", "board__rotate__right", "object__attach_hand__both", "object__attach_hand__left", "object__attach_hand__right", "object__attach_skrewdriver__both", "object__attach_skrewdriver__left", "object__attach_skrewdriver__right", "object__attach_wrench__both", "object__attach_wrench__left", "object__attach_wrench__right", "object__hold__both", "object__hold__left", "object__hold__right", "object__lift__left", "object__lift__right", "object__place__both", "object__place__left", "object__place__right", "object__plug__both", "object__plug__left", "object__plug__right", "other__browse__instructions__undefined", "other__read__instructions__undefined", "tool__hold__left", "tool__hold__right", "tool__lift__left", "tool__lift__right", "tool__place__left", "tool__place__right", "workpiece__hold__both", "workpiece__move__both", "workpiece__press_hammer__both", "workpiece__press_hand__both", "workpiece__rotate__both"]
        else:
            raise ValueError('Database is not implemented')

    def train(self, anns_train, anns_valid):
        from opentad.datasets import build_dataset, build_dataloader
        from opentad.cores import train_one_epoch, val_one_epoch, eval_one_epoch, build_optimizer, build_scheduler
        from opentad.utils import setup_logger, save_checkpoint, save_best_checkpoint

        # setup logger
        logger = setup_logger("Train", save_dir=self.cfg.work_dir, distributed_rank=self.rank)
        logger.info(f"Using torch version: {torch.__version__}, CUDA version: {torch.version.cuda}")
        logger.info(f"Config: \n{self.cfg.pretty_text}")

        # build dataset
        train_dataset = build_dataset(self.cfg.dataset.train, default_args=dict(logger=logger))
        train_loader = build_dataloader(train_dataset, rank=self.rank, world_size=self.world_size, shuffle=True, drop_last=True, **self.cfg.solver.train)

        val_dataset = build_dataset(self.cfg.dataset.val, default_args=dict(logger=logger))
        val_loader = build_dataloader(val_dataset, rank=self.rank, world_size=self.world_size, shuffle=False, drop_last=False, **self.cfg.solver.val)

        test_dataset = build_dataset(self.cfg.dataset.test, default_args=dict(logger=logger))
        test_loader = build_dataloader(test_dataset, rank=self.rank, world_size=self.world_size, shuffle=False, drop_last=False, **self.cfg.solver.test)

        # AMP: automatic mixed precision
        use_amp = getattr(self.cfg.solver, "amp", False)
        if use_amp:
            logger.info("Using Automatic Mixed Precision...")
            # GradScaler is only needed for float16 AMP, not bfloat16.
            # bfloat16 has the same dynamic range as float32, so loss scaling
            # is unnecessary and can introduce NaN from rounding errors.
            scaler = None
        else:
            scaler = None

        # build optimizer and scheduler
        optimizer = build_optimizer(self.cfg.optimizer, self.model, logger)
        scheduler, max_epoch = build_scheduler(self.cfg.scheduler, optimizer, len(train_loader))

        # override the max_epoch
        max_epoch = self.cfg.workflow.get("end_epoch", max_epoch)

        # --- Early Stopping Parameters Initialization ---
        val_loss_best = 1e6  # Initialize best validation loss
        epochs_no_improve = 0  # Counter for epochs without improvement
        # Get early stopping patience and min_delta from config, with defaults
        early_stopping_patience = self.cfg.workflow.get("early_stopping_patience", -1)  # -1 means disabled
        early_stopping_min_delta = self.cfg.workflow.get("early_stopping_min_delta", 0.0)  # 0.0 means any improvement counts
        # measure gflops

        # resume: reset epoch, load checkpoint / best rmse
        if self.ckpt is not None:
            logger.info("Resume training from: {}".format(self.ckpt))
            device = f"cuda:{self.local_rank}"
            checkpoint = torch.load(self.ckpt, map_location=device)
            resume_epoch = checkpoint["epoch"]
            logger.info("Resume epoch is {}".format(resume_epoch))
            self.model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            if self.model_ema is not None:
                self.model_ema.module.load_state_dict(checkpoint["state_dict_ema"])

            # If resuming, ideally load previous best_val_loss and epochs_no_improve
            # For simplicity here, we re-initialize them, meaning early stopping
            # will start fresh from the resume point. The loaded model state
            # will still reflect the best from prior training.
            # val_loss_best = checkpoint.get("val_loss_best", val_loss_best)
            # epochs_no_improve = checkpoint.get("epochs_no_improve", epochs_no_improve)

            del checkpoint  #  save memory if the model is very large such as ViT-g
            torch.cuda.empty_cache()
        else:
            resume_epoch = -1

        # train the detector
        logger.info("Training Starts...\n")
        val_start_epoch = self.cfg.workflow.get("val_start_epoch", 0)  # Already defined, just keeping it here for clarity
        for epoch in range(resume_epoch + 1, max_epoch):
            train_loader.sampler.set_epoch(epoch)
            # train for one epoch
            train_one_epoch(train_loader, self.model, optimizer, scheduler, epoch, logger, rank=self.rank, model_ema=self.model_ema, clip_grad_l2norm=self.cfg.solver.clip_grad_norm, logging_interval=self.cfg.workflow.logging_interval, scaler=scaler, use_amp=use_amp)

            # save checkpoint
            if (epoch == max_epoch - 1) or ((epoch + 1) % self.cfg.workflow.checkpoint_interval == 0):
                if self.rank == 0:
                    save_checkpoint(self.model, self.model_ema, optimizer, scheduler, epoch, work_dir=self.cfg.work_dir)

            # val for one epoch and early stopping check
            # None = no val loss computed this epoch, so eval falls back to its interval
            val_loss_improved = None
            if epoch >= val_start_epoch:
                if (self.cfg.workflow.val_loss_interval > 0) and ((epoch + 1) % self.cfg.workflow.val_loss_interval == 0):
                    val_loss = val_one_epoch(val_loader, self.model, logger, self.rank, epoch, model_ema=self.model_ema, use_amp=use_amp)

                    # --- Early Stopping and Best Checkpoint Saving Logic ---
                    # Only activate early stopping if patience is set (i.e., not -1)
                    if early_stopping_patience > 0:
                        # Check for significant improvement
                        if val_loss < val_loss_best - early_stopping_min_delta:
                            logger.info(f"Validation loss improved from {val_loss_best:.4f} to {val_loss:.4f}. Resetting early stopping counter.")
                            val_loss_best = val_loss  # Update the best loss
                            epochs_no_improve = 0  # Reset counter
                            val_loss_improved = True
                            if self.rank == 0:
                                # Save the best model only when significant improvement is observed
                                save_best_checkpoint(self.model, self.model_ema, epoch, work_dir=self.cfg.work_dir)
                        else:
                            # No significant improvement, increment counter
                            epochs_no_improve += 1
                            val_loss_improved = False
                            logger.info(f"Validation loss did not improve significantly. Early stopping counter: {epochs_no_improve}/{early_stopping_patience}.")

                        # Check if early stopping condition is met
                        if epochs_no_improve >= early_stopping_patience:
                            logger.info(f"Early stopping triggered after {epochs_no_improve} epochs without significant improvement. Training will stop.")
                            break  # Exit the training loop
                    else:
                        # If early stopping is not enabled, use the original logic for saving best checkpoint
                        if val_loss < val_loss_best:
                            logger.info(f"New best epoch {epoch} based on val_loss: {val_loss:.4f}.")
                            val_loss_best = val_loss
                            val_loss_improved = True
                            if self.rank == 0:
                                save_best_checkpoint(self.model, self.model_ema, epoch, work_dir=self.cfg.work_dir)
                        else:
                            val_loss_improved = False

            # eval for one epoch (evaluation metrics, not for early stopping loss)
            # skipped when the val loss did not improve this epoch
            if epoch >= val_start_epoch and val_loss_improved is not False:
                if (self.cfg.workflow.val_eval_interval > 0) and ((epoch + 1) % self.cfg.workflow.val_eval_interval == 0):
                    eval_one_epoch(test_loader, self.model, self.cfg, logger, self.rank, model_ema=self.model_ema, use_amp=use_amp, world_size=self.world_size, not_eval=True, training=True)
            elif val_loss_improved is False and (self.cfg.workflow.val_eval_interval > 0) and ((epoch + 1) % self.cfg.workflow.val_eval_interval == 0):
                logger.info(f"Skipping evaluation at epoch {epoch}: val_loss did not improve.")
        logger.info("Training Over...\n")

        # Load best model if exists
        best_checkpoint_path = os.path.join(self.cfg.work_dir, "checkpoint", "best.pth")
        if os.path.exists(best_checkpoint_path):
            logger.info(f"Loading best checkpoint from {best_checkpoint_path} for final evaluation...")
            checkpoint = torch.load(best_checkpoint_path, map_location=f"cuda:{self.local_rank}")
            self.model.load_state_dict(checkpoint["state_dict"])
            if self.model_ema is not None and "state_dict_ema" in checkpoint:
                self.model_ema.module.load_state_dict(checkpoint["state_dict_ema"])
        else:
            logger.info("Best checkpoint not found. Using the last model for final evaluation.")
        eval_one_epoch(test_loader, self.model, self.cfg, logger, self.rank, model_ema=self.model_ema, use_amp=use_amp, world_size=self.world_size, not_eval=True, training=True)

    def load(self, mode):
        import torchinfo
        import torch.distributed as dist
        from torch.distributed.algorithms.ddp_comm_hooks import default as comm_hooks
        from torch.nn.parallel import DistributedDataParallel
        from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
        from opentad.models import build_detector
        from opentad.utils import remap_legacy_sparse_conv_weights, ModelEma
        from pcr_framework.src.constants import Modes

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
            use_static_graph = getattr(self.cfg.solver, 'static_graph', False)
            self.model = DistributedDataParallel(self.model, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=False if use_static_graph else True, static_graph=use_static_graph)  # default is False, should be true when use activation checkpointing in E2E
            print(f'Using DDP with total {self.world_size} GPUS...')
        # [batch, num_clips, channels, T, H, W]
        window_size = getattr(self.cfg, 'window_size', 768)
        img_size = self.cfg.model['backbone']['backbone']['img_size']
        torchinfo.summary(self.model.module.backbone if torchrun_mode else self.model.backbone, input_size=(1, 1, 3, window_size, img_size, img_size), depth=5, device=self.device, col_names=['input_size', 'output_size', 'num_params', 'kernel_size'])
        # FP16 compression
        use_fp16_compress = getattr(self.cfg.solver, 'fp16_compress', False)
        if use_fp16_compress:
            print('Using FP16 compression ...')
            self.model.register_comm_hook(state=None, hook=comm_hooks.fp16_compress_hook)
        # Model EMA
        self.model_ema = None
        use_ema = getattr(self.cfg.solver, 'ema', False)
        if use_ema:
            print('Using Model EMA ...')
            if mode is Modes.TRAIN:
                self.model_ema = ModelEma(self.model.module)
        if mode is Modes.TEST:
            # Load checkpoint
            if self.ckpt is not None:
                checkpoint_path = self.ckpt
            elif 'test_epoch' in self.cfg.inference.keys():
                checkpoint_path = model_path + f'checkpoint/epoch_{self.cfg.inference.test_epoch}.pth'
            else:
                checkpoint_path = model_path + 'checkpoint/best.pth'
            print(f'Loading checkpoint from: {checkpoint_path}')
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            print(f'Checkpoint is epoch {checkpoint.get("epoch", "unknown")}.')
            state_dict = checkpoint['state_dict_ema'] if use_ema else checkpoint['state_dict']
            # Older checkpoints may or may not carry the DDP "module." prefix depending on
            # how EMA/the model were wrapped at save time; normalize before comparing/loading.
            consume_prefix_in_state_dict_if_present(state_dict, prefix='module.')
            state_dict = remap_legacy_sparse_conv_weights(state_dict, self.model.module.state_dict() if torchrun_mode else self.model.state_dict())
            self.model.module.load_state_dict(state_dict, strict=False) if torchrun_mode else self.model.load_state_dict(state_dict)

    def get_video_id(self, filename):
        if self.database == 'attach':
            video_name = os.path.basename(os.path.dirname(filename))
        else:
            video_name = os.path.splitext(os.path.basename(filename))[0]
            prepare_video_info_cfg = next((p for p in self.cfg.dataset.test.pipeline if p.get('type') == 'PrepareVideoInfo'), {})
            video_prefix = prepare_video_info_cfg.get('prefix', '')
            if video_prefix and video_name.startswith(video_prefix):
                video_name = video_name[len(video_prefix):]
        return video_name

    def process(self, ann, pred):
        import cv2
        import json
        from opentad.cores import eval_one_epoch
        from opentad.datasets.builder import collate as default_collate
        from pcr_framework.src.annotations import TemporalCategory
        from pcr_framework.src.datasets import Database

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

        datasets = [subclass().get_names() for subclass in Database.__subclasses__()]
        idx = [datasets.index(subset) for subset in datasets if self.database in subset]
        categories = Database.__subclasses__()[idx[0]]().get_categories()
        # Build dataset with only the video from pred.filename
        test_cfg = self.cfg.dataset.test
        video_name = self.get_video_id(pred.filename)
        data_path = test_cfg.data_path
        video_info = {}
        frame, duration = _probe_video_frame_duration(pred.filename)
        video_info["frame"] = frame
        video_info["duration"] = duration
        # Needed by get_gt() when the dataset's pipeline requires real gt_segments/gt_labels even at test time (e.g. ATTACH, whose config doesn't set test_mode=True)
        video_info["annotations"] = [{'segment': list(action.segment), 'label': self.class_map[list(categories.values()).index(action.label)]} for action in ann.actions]
        test_dataset = ExampleDataset(video_name=video_name, video_info=video_info, data_path=data_path, pipeline=test_cfg.pipeline, class_map=self.class_map, cfg_test=test_cfg)
        # Build dataloader with custom collate that moves to device
        sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=False, drop_last=False)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.cfg.solver.test.get("batch_size", 1) // self.world_size, collate_fn=collate_with_device, sampler=sampler, num_workers=0, pin_memory=False)

        # AMP: automatic mixed precision
        use_amp = getattr(self.cfg.solver, "amp", False)

        # eval_one_epoch combines all GPU results with gather_ddp_results() and saves to a single JSON
        eval_one_epoch(test_loader, self.model, self.cfg, print, self.rank, model_ema=None, use_amp=use_amp, world_size=self.world_size, not_eval=True)

        # Only rank 0 reads the combined result_detection.json (which contains all predictions from all GPUs)
        if self.rank == 0:
            result_file = os.path.join(self.cfg.work_dir, 'result_detection.json')
            with open(result_file, 'r') as ifs:
                data = json.load(ifs)
            for video_name, actions in data['results'].items():
                for action in actions:
                    label = categories[self.class_map.index(action["label"])]
                    pred.add_action(TemporalCategory(label=label, segment=tuple(action["segment"]), score=action["score"]))
