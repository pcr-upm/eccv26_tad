#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@upm.es'

import os
import sys
sys.path.append(os.getcwd())
import cv2
import json
import wandb
import importlib.util
from tqdm import tqdm
from pathlib import Path
from opentad.utils import override_dataset_paths
from opentad.evaluations import build_evaluator
from images_framework.src.constants import Modes
from images_framework.src.composite import Composite
from images_framework.src.annotations import GenericVideo
from images_framework.src.viewer import Viewer
from src.eccv26_tad import ECCV26TAD


def parse_options():
    """
    Parse options from command line.
    """
    import argparse
    parser = argparse.ArgumentParser()
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
    ann_file = args.ann_file
    class_map = args.class_map
    data_root = args.data_root
    block_list = args.block_list
    external_cls_path = args.external_cls_path
    return unknown, ann_file, class_map, data_root, block_list, external_cls_path


def main():
    """
    SV-TAD: Native Sparse Convolutions for Efficient Temporal Action Detection test database script.
    """
    print('OpenCV ' + cv2.__version__)
    unknown, ann_file, class_map, data_root, block_list, external_cls_path = parse_options()

    # Load vision components
    #composite = Composite()
    sr = ECCV26TAD('')
    #composite.add(sr)
    sr.parse_options(unknown)
    sr.load(Modes.TEST)
    sr.cfg = override_dataset_paths(sr.cfg, ann_file=ann_file, class_map=class_map, data_path=data_root, block_list=block_list, external_cls_path=external_cls_path)
    sr.cfg.work_dir = sr.path
    sr.cfg.post_processing.save_dict = True
    spec = importlib.util.find_spec('images_framework')
    output_path = os.path.join('images_framework' if spec is None else os.path.dirname(spec.origin), 'output')
    viewer = Viewer('eccv26_tad_database')
    dirname = os.path.join(output_path, 'images/')
    Path(dirname).mkdir(parents=True, exist_ok=True)

    # Load annotations
    # anns = load_annotations(ann_file)

    # Process database
    # for i in tqdm(range(len(anns)), file=sys.stdout):
    #     sr.process(anns[i], pred)
    ann, pred = GenericVideo(filename=''), GenericVideo(filename='')
    sr.process(ann, pred)

    # Compute metrics
    print('Evaluation starts...')
    wandb.init(project=sr.cfg.get('project_name', 'opentad'), config=sr.cfg)
    with open(os.path.join(sr.cfg.work_dir, 'result_detection.json'), 'r') as ifs:
        result_dict = json.load(ifs)
    evaluator = build_evaluator(dict(prediction_filename=result_dict, **sr.cfg.evaluation))
    metrics_dict = evaluator.evaluate()
    evaluator.logging()
    wandb.log(metrics_dict)
    columns = ['video-id', 'segment', 'label', 'score']
    data = []
    for video_id, predictions in result_dict['results'].items():
        for pred in predictions:
            data.append([video_id, str(pred['segment']), pred['label'], pred['score']])
    wandb.log({'evaluation_results': wandb.Table(data=data, columns=columns)})
    print('End of eccv26_tad_database')


if __name__ == "__main__":
    main()
