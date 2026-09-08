#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@upm.es'

import os
import sys
sys.path.append(os.getcwd())
import cv2
import wandb
import random
import string
from pathlib import Path
from opentad.utils import override_dataset_paths
from images_framework.src.constants import Modes
from images_framework.src.composite import Composite
from src.eccv26_tad import ECCV26TAD


def parse_options():
    """
    Parse options from command line.
    """
    import argparse
    from mmengine.config import DictAction
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb", action="store_true", 
                        help="whether to use wandb for logging.")
    parser.add_argument("--project", type=str, default="thumos", 
                        help="wandb project name.")
    parser.add_argument("--ann-file", type=str, default=None,
                        help="override the annotation file path for all dataset splits and evaluation.")
    parser.add_argument("--class-map", type=str, default=None,
                        help="override the class map / category index file path for all dataset splits.")
    parser.add_argument("--data-root", type=str, default=None,
                        help="override the raw video / feature data root path for all dataset splits.")
    parser.add_argument("--block-list", type=str, default=None,
                        help="override the block list file path for all dataset splits.")
    parser.add_argument("--external-cls-path", type=str, default=None,
                        help="override the external classifier (post_processing.external_cls) path.")
    parser.add_argument("--cfg-options", nargs="+", action=DictAction, default=None,
                        help="override settings in the config (e.g., --cfg-options model.backbone.backbone.n_landmarks=32)")
    args, unknown = parser.parse_known_args()
    print(parser.format_usage())
    use_wandb = args.wandb
    project = args.project
    ann_file = args.ann_file
    class_map = args.class_map
    data_root = args.data_root
    block_list = args.block_list
    external_cls_path = args.external_cls_path
    cfg_options = args.cfg_options or {}
    return unknown, use_wandb, project, ann_file, class_map, data_root, block_list, external_cls_path, cfg_options


def main():
    """
    SV-TAD: Native Sparse Convolutions for Efficient Temporal Action Detection train database script.
    """
    print('OpenCV ' + cv2.__version__)
    unknown, use_wandb, project, ann_file, class_map, data_root, block_list, external_cls_path, cfg_options = parse_options()

    # Load vision components
    composite = Composite()
    sr = ECCV26TAD('')
    composite.add(sr)
    composite.parse_options(unknown)
    if ann_file or class_map or data_root or block_list or external_cls_path:
        sr.cfg = override_dataset_paths(sr.cfg, ann_file=ann_file, class_map=class_map, data_path=data_root, block_list=block_list, external_cls_path=external_cls_path)
    if cfg_options:
        sr.cfg.merge_from_dict(cfg_options)
    print(f"Config: \n{sr.cfg.pretty_text}")
    composite.load(Modes.TRAIN)
    # Generate random three letter run id
    run_id = "".join([random.choice(string.ascii_lowercase) for _ in range(3)])
    sr.cfg.work_dir = os.path.join(sr.cfg.work_dir, f"gpu{sr.world_size}_id{run_id}/")
    if sr.rank == 0:
        Path(sr.cfg.work_dir).mkdir(parents=True, exist_ok=True)

    # Load annotations
    anns_train, anns_valid = [], []

    # Train model
    if use_wandb and sr.rank == 0:
        p = Path(sr.cfg.work_dir).parts
        run_name = sr.database + '_' + p[3].split('_', 2)[2] + '_' + p[4].split('_', 1)[1]
        wandb.init(project=project, name=run_name, config=sr.cfg.to_dict())
    composite.train(anns_train, anns_valid)
    if use_wandb and sr.rank == 0:
        wandb.finish()


if __name__ == "__main__":
    main()
