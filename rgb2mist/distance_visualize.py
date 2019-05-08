from __future__ import print_function

import argparse
import numpy as np
import pandas as pd
import random
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import SubsetRandomSampler

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

from models.encoder import encoder8x16x16
from models.decoder import decoder256x256
from models.encoder_decoder import EncoderDecoder

from taskonomy_dataset import TaskonomyDatasetEuclidean

from PIL import Image
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from skimage.color.colorlabel import label2rgb
plt.switch_backend('agg')

import os
import time
from tqdm import tqdm

import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")


parser = argparse.ArgumentParser()
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--train_batch', default=32, type=int)
parser.add_argument('--test_batch', default=4, type=int)
parser.add_argument('--epochs', default=30, type=int)
parser.add_argument('--lr', default=0.0005, type=float)
parser.add_argument('--resume', action='store_true')
parser.add_argument('--brenta', action='store_true')
parser.add_argument('--shuffle', action='store_true', default=False)
args = parser.parse_args()

num_classes = 1
ckpt = './checkpoints/rgb2depth-ckpt.pth'
best = './checkpoints/rgb2depth-best.pth'

def main():
    taskonomy_transform = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize((0.5456, 0.5176, 0.4863),
                                                                   (0.1825, 0.1965, 0.2172))])
    taskonomy_testset = TaskonomyDatasetEuclidean('../data/tiny_taskonomy_rgb_test.csv',
                                                  '/depth_euclidean',
                                                  'depth_euclidean.png',
                                                  resize256 = True,
                                                  transform=taskonomy_transform,
                                                  brenta = args.brenta)
    taskonomy_testloader = torch.utils.data.DataLoader(taskonomy_testset,
                                                       batch_size=args.test_batch,
                                                       shuffle=True,
                                                       num_workers=8)


    taskonomy_testframe = pd.read_csv('../data/tiny_taskonomy_rgb_test.csv', delimiter=' ')
    # Define the loss function for different tasks.
    criterion = nn.L1Loss()

    device = torch.device(args.device)
    model = EncoderDecoder(encoder8x16x16(), decoder256x256(num_output_channels=num_classes))
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=2e-6)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20], gamma=0.1)
    best_loss = float('inf')

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoints'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(best)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state'])
        model = torch.nn.DataParallel(model).to(device)
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        best_loss = checkpoint['best_loss']
    else:
        start_epoch = 0
        model = torch.nn.DataParallel(model).to(device)
        cudnn.benchmark = True
    print('  Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    model.eval()

    dataiters = iter(taskonomy_testloader)
    images, labels, idxs = next(dataiters)
    idxs = idxs.numpy()
    with torch.no_grad():
        images = images.to(device)
        outputs = model(images)

    images = images.cpu().numpy()
    labels = labels.squeeze(dim=1).numpy()
    pred = outputs.data.cpu().squeeze(dim=1).numpy()

    plt.figure(1)
    for i in range(args.test_batch):
        # Original image
        plt.subplot(3,args.test_batch,i+1)
        img = Image.open(taskonomy_testframe.iloc[idxs[i],0])
        plt.imshow(img)
        plt.axis('off')

        # Ground truth
        plt.subplot(3,args.test_batch,i+args.test_batch+1)
        label = labels[i]
        plt.imshow(label, cmap=plt.get_cmap('hot'), vmin=0, vmax=1)
        plt.colorbar()
        plt.axis('off')

        # Predicted depth
        plt.subplot(3,args.test_batch,i+2*args.test_batch+1)
        plt.imshow(pred[i], cmap=plt.get_cmap('hot'), vmin=0, vmax=1)
        plt.colorbar()
        plt.axis('off')

    plt.show()
    plt.savefig('demo.png', dpi=300)



if __name__ == '__main__':
    main()
