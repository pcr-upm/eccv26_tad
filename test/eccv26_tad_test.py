#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@upm.es'

import os
import sys
sys.path.append(os.getcwd())
import cv2
import numpy as np
import importlib.util
from pathlib import Path
from images_framework.src.constants import Modes
from images_framework.src.composite import Composite
from images_framework.src.annotations import GenericGroup, GenericImage
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
    parser.add_argument('--show-viewer', '-v', dest='show_viewer', action="store_true",
                        help='Show results visually.')
    parser.add_argument('--save-image', '-i', dest='save_image', action="store_true",
                        help='Save processed images.')
    args, unknown = parser.parse_known_args()
    print(parser.format_usage())
    input_data = args.input_data
    show_viewer = args.show_viewer
    save_image = args.save_image
    return unknown, input_data, show_viewer, save_image


def main():
    """
    SV-TAD: Native Sparse Convolutions for Efficient Temporal Action Detection test script.
    """
    print('OpenCV ' + cv2.__version__)
    unknown, input_data, show_viewer, save_image = parse_options()

    # Load vision components
    composite = Composite()
    sr = ECCV26TAD('')
    composite.add(sr)
    composite.parse_options(unknown)
    composite.load(Modes.TEST)
    spec = importlib.util.find_spec('images_framework')
    output_path = os.path.join('images_framework' if spec is None else os.path.dirname(spec.origin), 'output')
    viewer = Viewer('eccv26_tad_test')
    dirname = os.path.join(output_path, 'images/')
    Path(dirname).mkdir(parents=True, exist_ok=True)

    # Process video and show results
    ann, pred = GenericGroup(), GenericGroup()
    img_ann = GenericImage(input_data)
    ann.add_image(img_ann)
    ticks = cv2.getTickCount()
    composite.process(ann, pred)
    ticks = cv2.getTickCount() - ticks
    if show_viewer:
        for img_pred in pred.images:
            viewer.set_image(img_pred)
        composite.show(viewer, ann, pred)
        fps = 'FPS = ' + "{0:.3f}".format(cv2.getTickFrequency() / ticks)
        viewer.text(pred.images[0], fps, (20, np.shape(viewer.get_image(pred.images[0]))[0] - 20), 0.5, (0, 255, 0))
        viewer.show(1)
    if save_image:
        for img_pred in pred.images:
            viewer.set_image(img_pred)
        composite.show(viewer, ann, pred)
        viewer.save(dirname)
        composite.save(dirname, pred)
    print('End of eccv26_tad_test')


if __name__ == '__main__':
    main()
