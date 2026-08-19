#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@upm.es'

import os
import sys
sys.path.append(os.getcwd())
import cv2
import copy
import importlib.util
from tqdm import tqdm
from pathlib import Path
from opentad.utils import override_dataset_paths
from opentad.datasets import build_dataset
from opentad.evaluations import build_evaluator
from images_framework.src.constants import Modes
from images_framework.src.datasets import Database
from images_framework.src.composite import Composite
from images_framework.src.viewer import Viewer
from src.eccv26_tad import ECCV26TAD


def parse_options():
    """
    Parse options from command line.
    """
    import argparse
    from mmengine.config import DictAction
    parser = argparse.ArgumentParser()
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
    parser.add_argument('--save-video', '-v', dest='save_video', action="store_true",
                        help='Save processed video.')
    args, unknown = parser.parse_known_args()
    print(parser.format_usage())
    ann_file = args.ann_file
    class_map = args.class_map
    data_root = args.data_root
    block_list = args.block_list
    external_cls_path = args.external_cls_path
    cfg_options = args.cfg_options or {}
    save_video = args.save_video
    return unknown, ann_file, class_map, data_root, block_list, external_cls_path, cfg_options, save_video


def load_annotations(config, load_images=True):
    """
    Load ground truth annotations from test dataset.
    """
    test_dataset = build_dataset(config)
    db = 'thumos' if config.type.startswith('Thumos') else 'anet' if config.type.startswith('Anet') else 'attach' if config.type.startswith('Attach') else None
    datasets = [subclass().get_names() for subclass in Database.__subclasses__()]
    idx = [datasets.index(subset) for subset in datasets if db in subset]
    if len(idx) != 1:
        raise ValueError('Database does not exist')
    database = Database.__subclasses__()[idx[0]]()
    # A sliding dataset stores one entry per window, so several entries share the same video
    anns, seen = [], set()
    for entry in tqdm(test_dataset.data_list, file=sys.stdout):
        video_name, video_info = entry[0], entry[1]
        if video_name in seen:
            continue
        seen.add(video_name)
        anns.append(database.load_filename(test_dataset.data_path, db, (video_name, video_info), load_images))
    return anns


def main():
    """
    SV-TAD: Native Sparse Convolutions for Efficient Temporal Action Detection test database script.
    """
    print('OpenCV ' + cv2.__version__)
    unknown, ann_file, class_map, data_root, block_list, external_cls_path, cfg_options, save_video = parse_options()

    # Load vision components
    composite = Composite()
    sr = ECCV26TAD('')
    composite.add(sr)
    sr.parse_options(unknown)
    sr.load(Modes.TEST)
    # Apply config overrides from --cfg-options before path overrides
    if cfg_options:
        sr.cfg.merge_from_dict(cfg_options)
    sr.cfg = override_dataset_paths(sr.cfg, ann_file=ann_file, class_map=class_map, data_path=data_root, block_list=block_list, external_cls_path=external_cls_path)
    sr.cfg.work_dir = sr.path
    sr.cfg.post_processing.save_dict = True
    if save_video:
        viewer = Viewer('eccv26_tad_test')
        spec = importlib.util.find_spec('images_framework')
        output_path = os.path.join('images_framework' if spec is None else os.path.dirname(spec.origin), 'output')
        dirname = os.path.join(output_path, 'images/')
        Path(dirname).mkdir(parents=True, exist_ok=True)

    # Load annotations from test dataset
    anns = load_annotations(sr.cfg.dataset.test, load_images=save_video)

    # Process database
    label_names = {enum_obj: sr.class_map[idx] for idx, enum_obj in sr.classes.items()}
    result_dict = {'results': {}}
    for i in tqdm(range(len(anns)), file=sys.stdout):
        pred = copy.deepcopy(anns[i])
        pred.categories.clear()
        pred.actions.clear()
        sr.process(anns[i], pred)
        video_name = sr.get_video_id(pred.filename)
        result_dict['results'].setdefault(video_name, [])
        for action in pred.actions:
            result_dict["results"][video_name].append({'segment': list(action.segment), 'label': label_names[action.label], 'score': float(action.score)})
        if save_video:
            for img_pred in pred.images:
                viewer.set_image(img_pred)
            composite.show(viewer, anns[i], pred)
            viewer.save(dirname, as_video=True, format='avi', fps=30, codec='XVID')

    # Compute metrics
    import wandb
    print('Evaluation starts...')
    wandb.init(project=sr.cfg.get('project_name', 'opentad'), config=sr.cfg)
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
