#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Ricardo Pizarro'
__email__ = 'ricardo.pizarroc@edu.uah.es'

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
import torch
import numpy as np
from images_framework.src.annotations import GenericCategory
from images_framework.src.utils import load_geoimage, DepthMode, ChannelsMode
from images_framework.src.recognition import Recognition
os.environ['PYTHONHASHSEED'] = '0'
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
        self.batch_size = None
        self.epoch = None
        self.patience = None
        self.width = 224
        self.height = 224
        self.classes = None
        self.depth = None
        self.channels = None

    def parse_options(self, params):
        super().parse_options(params)
        import argparse
        from images_framework.categories.emotions import Emotion as Oe
        parser = argparse.ArgumentParser(prog='ECCV26TAD', add_help=False)
        parser.add_argument('--gpu', dest='gpu', type=int, action='append',
                            help='GPU ID (negative value indicates CPU).')
        parser.add_argument('--batch-size', dest='batch_size', type=int, default=16,
                            help='Number of images in each mini-batch.')
        parser.add_argument('--epoch', dest='epoch', type=int, default=1000,
                            help='Number of sweeps over the dataset to train.')
        parser.add_argument('--patience', dest='patience', type=int, default=40,
                            help='Number of epochs with no improvement after which training will be stopped.')
        args, unknown = parser.parse_known_args(params)
        print(parser.format_usage())
        mode_gpu = torch.cuda.is_available() and -1 not in args.gpu
        self.gpus = args.gpu
        self.device = torch.device('cuda' if mode_gpu else 'cpu')
        self.batch_size = args.batch_size
        self.epoch = args.epoch
        self.patience = args.patience
        if self.database == 'fer2013':
            self.classes = {0: Oe.FACE.ANGER, 1: Oe.FACE.DISGUST, 2: Oe.FACE.FEAR, 3: Oe.FACE.HAPPINESS, 4: Oe.FACE.NEUTRAL, 5: Oe.FACE.SADNESS, 6: Oe.FACE.SURPRISE}
        elif self.database == 'raf':
            self.classes = {0: Oe.FACE.SURPRISE, 1: Oe.FACE.FEAR, 2: Oe.FACE.DISGUST, 3: Oe.FACE.HAPPINESS, 4: Oe.FACE.SADNESS, 5: Oe.FACE.ANGER, 6: Oe.FACE.NEUTRAL}
        elif self.database == 'affectnet':
            self.classes = {0: Oe.FACE.NEUTRAL, 1: Oe.FACE.HAPPINESS, 2: Oe.FACE.SADNESS, 3: Oe.FACE.SURPRISE, 4: Oe.FACE.FEAR, 5: Oe.FACE.DISGUST, 6: Oe.FACE.ANGER, 7: Oe.FACE.CONTEMPT}
        elif self.database == 'affwild2':
            self.classes = {0: Oe.FACE.NEUTRAL, 1: Oe.FACE.ANGER, 2: Oe.FACE.DISGUST, 3: Oe.FACE.FEAR, 4: Oe.FACE.HAPPINESS, 5: Oe.FACE.SADNESS, 6: Oe.FACE.SURPRISE, 7: Oe.FACE.CONTEMPT}
        elif self.database == 'multipie':
            self.classes = {0: Oe.FACE.NEUTRAL, 1: Oe.FACE.HAPPINESS, 2: Oe.FACE.SURPRISE, 3: Oe.FACE.OTHER, 4: Oe.FACE.DISGUST, 5: Oe.FACE.FEAR}
        self.depth = DepthMode.UBYTE
        self.channels = ChannelsMode.THREE

    def train(self, anns_train, anns_valid):
        print('Training model')

    def load(self, mode):
        import torchinfo
        from images_framework.src.constants import Modes
        from src.models_fer import FERBaselineNet
        from src.checkpoint_loader import load_submodel_state_dict
        # Set up a neural network to train
        print('Load model')
        self.model = FERBaselineNet(self.database, num_expr=len(self.classes), pretrained_backbone=False)
        torchinfo.summary(self.model, input_size=(self.batch_size, 3, self.width, self.height), depth=5, device=self.device.type, col_names=['input_size', 'output_size', 'num_params', 'kernel_size'])
        if mode is Modes.TEST:
            model_path = self.path + 'data/' + self.database + '/'
            print('Loading model from {}'.format(model_path))
            self.model.load_state_dict(load_submodel_state_dict(str(model_path+'last.ckpt'), 'model'), strict=True)            
            self.model.to(self.device)
            self.model.eval()

    def process(self, ann, pred):
        from PIL import Image
        from torchvision import transforms
        from torchvision.transforms import InterpolationMode
        transform = transforms.Compose([
                transforms.Resize((100, 100), interpolation=InterpolationMode.BILINEAR),
                transforms.CenterCrop(96),
                transforms.ToTensor(),
                transforms.Normalize((0.5,) * 3, (0.5,) * 3),
        ])
        with torch.no_grad():
            for img_pred in pred.images:
                # Load image
                image, _ = load_geoimage(img_pred.filename, self.depth, self.channels)
                for obj_pred in img_pred.objects:
                    image_tensor = transform(Image.fromarray(image))
                    # Add batch dimension if needed
                    if image_tensor.ndim == 3:
                        image_tensor = image_tensor.unsqueeze(0)
                    image_tensor = image_tensor.to(self.device)
                    # Generate prediction
                    logits = self.model(image_tensor)
                    idx = logits.argmax(dim=1).item()
                    score = torch.softmax(logits, dim=1)[0, idx].item()
                    # Save prediction
                    obj_pred.add_category(GenericCategory(self.classes[idx], score))
