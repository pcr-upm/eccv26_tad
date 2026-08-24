#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@upm.es'

import os
import sys
sys.path.append(os.getcwd())
import cv2
import copy
import tempfile
import numpy as np
import importlib.util
from pathlib import Path
from opentad.utils import override_dataset_paths
from images_framework.src.constants import Modes
from images_framework.src.datasets import Database
from images_framework.src.composite import Composite
from images_framework.src.annotations import GenericVideo, GenericImage, TemporalCategory
from images_framework.src.viewer import Viewer
from src.eccv26_tad import ECCV26TAD


def parse_options():
    """
    Parse options from command line.
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-data', '-d', dest='input_data', required=True, default='',
                        help='Input as image video file.')
    parser.add_argument('--thresh', type=float, default=0.0,
                        help='Only show predictions with score above this threshold.')
    parser.add_argument('--topk', type=int, default=20,
                        help='Show at most this many predictions (sorted by score).')
    parser.add_argument('--data-root', type=str, default=None,
                        help='override the raw video / feature data root path.')
    parser.add_argument('--save-video', '-v', dest='save_video', action="store_true",
                        help='Save processed video.')
    args, unknown = parser.parse_known_args()
    print(parser.format_usage())
    input_data = args.input_data
    thresh = args.thresh
    topk = args.topk
    data_root = args.data_root
    save_video = args.save_video
    return unknown, input_data, thresh, topk, data_root, save_video


def load_annotations(filename):
    """
    Read the annotations of a single video directly from the matching JSON file.
    """
    import json
    with open(filename, 'r', encoding='utf-8') as ifs:
        payload = json.load(ifs)
    annotations = payload.get('annotations', [])
    gt = []
    for anno in annotations:
        gt.append(dict(segment=anno['segment'], label=anno['label']))
    gt.sort(key=lambda x: x['segment'][0])
    return gt


def main():
    """
    SV-TAD: Native Sparse Convolutions for Efficient Temporal Action Detection test video script.
    """
    print('OpenCV ' + cv2.__version__)
    unknown, input_data, thresh, topk, data_root, save_video = parse_options()

    # Load vision components
    composite = Composite()
    sr = ECCV26TAD('')
    composite.add(sr)
    composite.parse_options(unknown)
    composite.load(Modes.TEST)
    if data_root:
        sr.cfg = override_dataset_paths(sr.cfg, data_path=data_root)
    if save_video:
        viewer = Viewer('eccv26_tad_test')
        spec = importlib.util.find_spec('images_framework')
        output_path = os.path.join('images_framework' if spec is None else os.path.dirname(spec.origin), 'output')
        dirname = os.path.join(output_path, 'images/')
        Path(dirname).mkdir(parents=True, exist_ok=True)

    # Load annotations
    datasets = [subclass().get_names() for subclass in Database.__subclasses__()]
    db = Database.__subclasses__()[next((idx for idx, subset in enumerate(datasets) if 'thumos' in subset), None)]()
    categories = db.get_categories()
    ground_truth = load_annotations(os.path.splitext(input_data)[0]+'.json')
    ann = GenericVideo(filename=input_data)
    cap = cv2.VideoCapture(input_data)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video file: {input_data}')
    ann.frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    ann.duration = ann.frames / fps if fps > 0 else 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    for frame_idx in range(ann.frames):
        ret, frame = cap.read()
        if not ret:
            break
        tmp_filename = os.path.join(tempfile.gettempdir(), f'{frame_idx:06d}.jpg')
        cv2.imwrite(tmp_filename, frame)
        image = GenericImage(tmp_filename)
        image.tile = np.array([0, 0, width, height])
        ann.add_image(image)
    cap.release()
    pred = copy.deepcopy(ann)
    for action in ground_truth:
        ann.add_action(TemporalCategory(label=categories[int(action['label'])], segment=tuple(action['segment'])))

    # Process video
    ticks = cv2.getTickCount()
    composite.process(ann, pred)
    ticks = cv2.getTickCount() - ticks
    if save_video:
        for img_pred in pred.images:
            viewer.set_image(img_pred)
        composite.show(viewer, ann, pred)
        fps = 'FPS = ' + "{0:.3f}".format(pred.frames / (ticks / cv2.getTickFrequency()))
        viewer.text(pred.images[0], fps, (20, np.shape(viewer.get_image(pred.images[0]))[0] - 20), 0.5, (0, 255, 0))
        viewer.save(dirname, as_video=True, format='avi', fps=30, codec='XVID')

    # Print the best K results
    print("\n" + "=" * 70)
    print(f"VIDEO: {input_data}")
    print(f"  duration: {ann.duration} s | frames: {ann.frames}")
    print("=" * 70)
    print(f"\nGROUND TRUTH ({len(ann.actions)} segments):")
    if len(ann.actions) == 0:
        print("  (no ground truth annotations found for this video)")
    else:
        print(f"  {'start':>8}  {'end':>8}  label")
        print(f"  {'-'*8}  {'-'*8}  {'-'*20}")
        for action in ann.actions:
            s, e = action.segment
            print(f"  {s:>8.2f}  {e:>8.2f}  {action.label.name}")
    shown = [action for action in pred.actions if action.score >= thresh]
    if topk >= 0:
        shown = shown[: topk]
    print(
        f"\nPREDICTIONS (showing {len(shown)} of {len(pred.actions)}"
        f"{f', score >= {thresh}' if thresh > 0 else ''}):"
    )
    if len(shown) == 0:
        print("  (no predictions)")
    else:
        print(f"  {'start':>8}  {'end':>8}  {'score':>7}  label")
        print(f"  {'-'*8}  {'-'*8}  {'-'*7}  {'-'*20}")
        for action in shown:
            s, e = action.segment
            print(f"  {s:>8.2f}  {e:>8.2f}  {action.score:>7.4f}  {action.label.name}")
    print('End of eccv26_tad_test')


if __name__ == '__main__':
    main()
