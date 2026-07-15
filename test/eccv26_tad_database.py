#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Ricardo Pizarro'
__email__ = 'ricardo.pizarroc@edu.uah.es'

import os
import sys
sys.path.append(os.getcwd())
import cv2
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from mmengine.config import Config, DictAction
from opentad.models import build_detector
from opentad.datasets import build_dataset, build_dataloader
from opentad.cores import eval_one_epoch
from opentad.utils import (update_workdir, set_seed, create_folder, setup_logger, remap_legacy_sparse_conv_weights, override_dataset_paths)


def parse_options():
    """
    Parse options from command line.
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE", type=str, 
                        help="path to config file")
    parser.add_argument("--checkpoint", type=str, default="none", 
                        help="the checkpoint path")
    parser.add_argument("--seed", type=int, default=42, 
                        help="random seed")
    parser.add_argument("--id", type=int, default=0, 
                        help="repeat experiment id")
    parser.add_argument("--not_eval", action="store_true", 
                        help="whether to not to eval, only do inference")
    parser.add_argument("--cfg-options", nargs="+", action=DictAction, 
                        help="override settings")
    parser.add_argument("--ann-file", type=str, default=None,
                        help="override the annotation file path for all dataset splits and evaluation")
    parser.add_argument("--class-map", type=str, default=None,
                        help="override the class map / category index file path for all dataset splits")
    parser.add_argument("--data-root", type=str, default=None,
                        help="override the raw video / feature data root path for all dataset splits")
    parser.add_argument("--block-list", type=str, default=None,
                        help="override the block list file path for all dataset splits")
    parser.add_argument("--external-cls-path", type=str, default=None,
                        help="override the external classifier (post_processing.external_cls) path")
    args, unknown = parser.parse_known_args()
    print(parser.format_usage())
    config = args.config
    checkpoint = args.checkpoint
    seed = args.seed
    id = args.id
    not_eval = args.not_eval
    cfg_options = args.cfg_options
    ann_file = args.ann_file
    class_map = args.class_map
    data_root = args.data_root
    block_list = args.block_list
    external_cls_path = args.external_cls_path
    return unknown, config, checkpoint, seed, id, not_eval, cfg_options, ann_file, class_map, data_root, block_list, external_cls_path


def main():
    """
    SV-TAD: Native Sparse Convolutions for Efficient Temporal Action Detection test database script.
    """
    print('OpenCV ' + cv2.__version__)
    unknown, config, checkpoint, seed, id, not_eval, cfg_options, ann_file, class_map, data_root, block_list, external_cls_path = parse_options()

    # load config
    cfg = Config.fromfile(config)
    if cfg_options is not None:
        cfg.merge_from_dict(cfg_options)
    cfg = override_dataset_paths(cfg, ann_file=ann_file, class_map=class_map, data_path=data_root, block_list=block_list, external_cls_path=external_cls_path)

    # DDP init
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    print(f"Distributed init (rank {rank}/{world_size}, local rank {local_rank})")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)

    # set random seed, create work_dir
    set_seed(seed)
    cfg = update_workdir(cfg, id, torch.cuda.device_count())
    if rank == 0:
        create_folder(cfg.work_dir)

    # setup logger
    logger = setup_logger("Test", save_dir=cfg.work_dir, distributed_rank=rank)
    logger.info(f"Using torch version: {torch.__version__}, CUDA version: {torch.version.cuda}")
    logger.info(f"Config: \n{cfg.pretty_text}")

    # build dataset
    test_dataset = build_dataset(cfg.dataset.test, default_args=dict(logger=logger))
    test_loader = build_dataloader(test_dataset, rank=rank, world_size=world_size, shuffle=False, drop_last=False, **cfg.solver.test)

    # build model
    cfg.model['backbone']['custom']['pretrain'] = 'data/' + cfg.model['backbone']['custom']['pretrain']
    model = build_detector(cfg.model)

    # DDP
    model = model.to(local_rank)
    model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    logger.info(f"Using DDP with total {world_size} GPUS...")

    if cfg.inference.load_from_raw_predictions:  # if load with saved predictions, no need to load checkpoint
        logger.info(f"Loading from raw predictions: {cfg.inference.fuse_list}")
    else:  # load checkpoint: args -> config -> best
        if checkpoint != "none":
            checkpoint_path = checkpoint
        elif "test_epoch" in cfg.inference.keys():
            checkpoint_path = os.path.join(cfg.work_dir, f"checkpoint/epoch_{cfg.inference.test_epoch}.pth")
        else:
            checkpoint_path = os.path.join(cfg.work_dir, "checkpoint/best.pth")
        logger.info("Loading checkpoint from: {}".format(checkpoint_path))
        device = f"cuda:{rank % torch.cuda.device_count()}"
        checkpoint = torch.load(checkpoint_path, map_location=device)
        logger.info("Checkpoint is epoch {}.".format(checkpoint["epoch"]))

        # Model EMA
        use_ema = getattr(cfg.solver, "ema", False)
        state_dict = checkpoint["state_dict_ema"] if use_ema else checkpoint["state_dict"]
        # Older checkpoints may or may not carry the DDP "module." prefix depending on
        # how EMA/the model were wrapped at save time; normalize before comparing/loading.
        consume_prefix_in_state_dict_if_present(state_dict, prefix="module.")
        state_dict = remap_legacy_sparse_conv_weights(state_dict, model.module.state_dict())
        missing, unexpected = model.module.load_state_dict(state_dict, strict=False)
        if missing:
            logger.info(f"Missing keys in checkpoint: {len(missing)} keys")
            for k in missing:
                logger.info(f"  - {k}")
        if unexpected:
            logger.info(f"Unexpected keys in checkpoint: {len(unexpected)} keys")
            for k in unexpected:
                logger.info(f"  - {k}")
        if use_ema:
            logger.info("Using Model EMA...")

    # AMP: automatic mixed precision
    use_amp = getattr(cfg.solver, "amp", False)
    if use_amp:
        logger.info("Using Automatic Mixed Precision...")

    # test the detector
    logger.info("Testing Starts...\n")
    eval_one_epoch(test_loader, model, cfg, logger, rank, model_ema=None, use_amp=use_amp, world_size=world_size, not_eval=not_eval)
    logger.info("Testing Over...\n")


if __name__ == "__main__":
    main()
