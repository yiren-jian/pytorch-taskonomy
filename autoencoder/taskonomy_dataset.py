from __future__ import print_function

from PIL import Image
from matplotlib import pyplot as plt
# plt.switch_backend('agg')

import argparse
import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

import sys
import os
import tqdm


def data_statistics(loader):
    mean = 0.
    std = 0.
    nb_samples = 0.
    step = 0
    for data, labels, idxs in loader:
        step += 1
        print('step: %d' %step)
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples
    print('mean:', mean, 'std: ', std)

class TaskonomyDatasetAutoencoder(torch.utils.data.Dataset):
    def __init__(self,
                 path2images,                        # '../data/tiny_taskonomy_rgb_xxx.csv'
                 task_folder,                        # '/segment_semantic' or '/class_object', etc.
                 task_extension,                     # 'segmentsemantic.png' or 'class_1000.npy', etc.
                 resize256 = False,
                 transform=transforms.ToTensor(),
                 brenta = False):
        """
        Args:
            path2images (string): A csv file with path to images (RGB) and labels (semantic).
            resize256 (boolen): Resize images to 256x256 before torch.vision.transforms.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.images_frame = pd.read_csv(path2images, delimiter = ' ')
        self.task_folder = task_folder
        self.task_extension = task_extension
        self.resize256 = resize256
        self.transform = transform
        self.brenta = brenta

    def __getitem__(self, idx):
        rgb_name = self.images_frame.iloc[idx, 0]
        if self.brenta == True:
            rgb_name = rgb_name.replace('/home/yiren/datasets', '/home/brenta/scratch/data')
        image = Image.open(rgb_name)
        if self.resize256 == True:
            image.thumbnail((256,256))
        if self.transform:
            image = self.transform(image)
        target = image
        return image, target, idx

    def __len__(self):
        return len(self.images_frame)


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_batch', default=32, type=int)
    parser.add_argument('--test_batch', default=32, type=int)
    parser.add_argument('--statistics', action='store_true')
    parser.add_argument('--brenta', action='store_true')
    args = parser.parse_args()

    taskonomy_transform = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize((0.5456, 0.5176, 0.4863),
                                                                   (0.1825, 0.1965, 0.2172))])
    taskonomy_trainset = TaskonomyDatasetAutoencoder('../data/tiny_taskonomy_rgb_train.csv',
                                                     '/rgb',
                                                     'rgb.png',
                                                     resize256 = True,
                                                     transform=taskonomy_transform,
                                                     brenta = args.brenta)
    taskonomy_trainloader = torch.utils.data.DataLoader(taskonomy_trainset,
                                                        batch_size=args.train_batch,
                                                        shuffle=True,
                                                        num_workers=8)
    taskonomy_testset = TaskonomyDatasetAutoencoder('../data/tiny_taskonomy_rgb_test.csv',
                                                '/rgb',
                                                'rgb.png',
                                                resize256 = True,
                                                transform=taskonomy_transform,
                                                brenta = args.brenta)
    taskonomy_testloader = torch.utils.data.DataLoader(taskonomy_testset,
                                                       batch_size=args.test_batch,
                                                       shuffle=False,
                                                       num_workers=8)
