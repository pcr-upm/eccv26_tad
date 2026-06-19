#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Ricardo Pizarro'
__email__ = 'ricardo.pizarroc@edu.uah.es'

import os
import sys
sys.path.append(os.getcwd())
import cv2
import json
import copy
import numpy as np
import importlib.util
from pathlib import Path
from images_framework.src.constants import Modes
from images_framework.src.datasets import Database
from images_framework.src.composite import Composite
from images_framework.src.annotations import GenericGroup, GenericImage, PersonObject, GenericLandmark, GenericCategory
from images_framework.src.viewer import Viewer
from images_framework.src.utils import load_geoimage
from images_framework.regression.alignment.landmarks import lps
from src.eccv26_tad import ECCV26TAD

image_extensions = ('bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff')
video_extensions = ('mp4', 'avi', 'mkv')


def parse_options():
    """
    Parse options from command line.
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-data', '-d', dest='input_data', required=False, default='',
                        help='Input as image, directory, camera or video file.')
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


def process_frame(composite, filename, show_viewer, save_image, viewer, delay, dirname):
    """
    Process frame and show results.
    """
    datasets = [subclass().get_names() for subclass in Database.__subclasses__()]
    # Read annotations
    ann, pred = GenericGroup(), GenericGroup()
    img_ann = GenericImage(filename)
    img, _ = load_geoimage(img_ann.filename)
    img_ann.tile = np.array([0, 0, img.shape[1], img.shape[0]])
    pred.add_image(copy.deepcopy(img_ann))
    ifs = os.path.splitext(filename)[0]+'.json'
    if os.path.exists(ifs):
        db = Database.__subclasses__()[next((idx for idx, subset in enumerate(datasets) if json.load(open(ifs))['database'] in subset), None)]()
        landmarks = db.get_landmarks()
        gender = db._gender
        race = db._race
        age = db._age
        categories = db.get_categories()
        for line in json.load(open(ifs))['annotations']:
            obj = PersonObject()
            bbox = line['bbox']
            obj.bb = (float(bbox['pt1_x']), float(bbox['pt1_y']), float(bbox['pt2_x']), float(bbox['pt2_y']))
            for lnd in line['landmarks']:
                name = list(landmarks.keys())[next((ids for ids, xs in enumerate(landmarks.values()) for x in xs if x == lnd['label']), None)].value 
                lp = next((elem for part in lps.keys() for elem in part if elem.value == name), None)
                obj.add_landmark(GenericLandmark(lnd['label'], lp, lnd['pos'], lnd['visible'], lnd['confidence']), lps[type(lp)])
            attributes = line['attributes']
            obj.add_attribute(GenericCategory(gender[int(attributes['gender'])]))
            obj.add_attribute(GenericCategory(race[int(attributes['race'])]))
            obj.add_attribute(GenericCategory(age[int(attributes['age'])]))
            pred.images[-1].add_object(copy.deepcopy(obj))
            obj.add_category(GenericCategory(categories[int(attributes['emotion'])]))
            img_ann.add_object(obj)
    else:
        from images_framework.detection.ssd16_detection.src.ssd16_detection import SSD16Detection
        sd = SSD16Detection('images_framework/detection/ssd16_detection/')
        sd.parse_options(['--database', 'coco'])
        sd.load(Modes.TEST)
        sd.process(ann, pred)
    ann.add_image(img_ann)
    # Process frame and show results
    ticks = cv2.getTickCount()
    composite.process(ann, pred)
    ticks = cv2.getTickCount() - ticks
    if show_viewer:
        for img_pred in pred.images:
            viewer.set_image(img_pred)
        composite.show(viewer, ann, pred)
        fps = 'FPS = ' + "{0:.3f}".format(cv2.getTickFrequency() / ticks)
        viewer.text(pred.images[0], fps, (20, np.shape(viewer.get_image(pred.images[0]))[0] - 20), 0.5, (0, 255, 0))
        viewer.show(delay)
    if save_image:
        for img_pred in pred.images:
            viewer.set_image(img_pred)
        composite.show(viewer, ann, pred)
        viewer.save(dirname)
        composite.save(dirname, pred)


def main():
    """
    SV-TAD: Native Sparse Convolutions for Efficient Temporal Action Detection test script.
    """
    print('OpenCV ' + cv2.__version__)
    unknown, input_data, show_viewer, save_image = parse_options()

    # Determine if we get the images from a camera, video or directory
    process_image, process_directory, process_video = False, False, False
    if os.path.isfile(input_data):
        if input_data.lower().endswith(image_extensions):
            print('Processing an image file ...')
            process_image = True
        elif input_data.lower().endswith(video_extensions):
            print('Processing from a video file ...')
            process_video = True
            capture = cv2.VideoCapture(input_data)
            if not capture.isOpened():
                raise ValueError('Could not grab images from video')
        else:
            raise ValueError('Unknown input file extension')
    else:
        if os.path.isdir(input_data):
            print('Processing a directory ...')
            process_directory = True
        else:
            print('Processing from camera ...')
            process_video = True
            capture = cv2.VideoCapture(0)
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            if not capture.isOpened():
                raise ValueError('Could not grab images from camera')

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

    # Process frame and show results
    if process_image:
        process_frame(composite, input_data, show_viewer, save_image, viewer, 0, dirname)
    elif process_directory:
        for filename in os.listdir(input_data):
            filename = input_data + '/' + filename
            if not filename.lower().endswith(image_extensions):
                continue
            process_frame(composite, filename, show_viewer, save_image, viewer, 0, dirname)
    elif process_video:
        idx = 0
        while True:
            success, frame = capture.read()
            if not success:
                break
            # Save camera frame into disk to continue processing
            filename = dirname + str(idx).zfill(5) + '.png'
            cv2.imwrite(filename, frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            idx += 1
            process_frame(composite, filename, show_viewer, save_image, viewer, 1, dirname)
        capture.release()
    print('End of eccv26_tad_test')


if __name__ == '__main__':
    main()
