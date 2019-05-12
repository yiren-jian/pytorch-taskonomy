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
from models.decoder import decoder1000
from models.encoder_decoder import EncoderDecoder

from taskonomy_dataset import TaskonomyDatasetVanish

import os
import time
from tqdm import tqdm

from PIL import Image
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")


parser = argparse.ArgumentParser()
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--train_batch', default=32, type=int)
parser.add_argument('--test_batch', default=4, type=int)
parser.add_argument('--epochs', default=15, type=int)
parser.add_argument('--lr', default=0.0005, type=float)
parser.add_argument('--resume', action='store_true')
parser.add_argument('--shuffle', action='store_true')
parser.add_argument('--brenta', action='store_true')
args = parser.parse_args()

num_classes = 9
if os.path.isdir('./checkpoints') == False:
    os.makedirs('./checkpoints')
ckpt = './checkpoints/Vanish-ckpt.pth'
best = './checkpoints/Vanish-best.pth'

def main():
    taskonomy_transform = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize((0.5456, 0.5176, 0.4863),
                                                                   (0.1825, 0.1965, 0.2172))])
    taskonomy_testset = TaskonomyDatasetVanish('../data/tiny_taskonomy_rgb_test.csv',
                                                  '/point_info',
                                                  'point_info.json',
                                                  resize256 = True,
                                                  transform=taskonomy_transform,
                                                  brenta = args.brenta)
    taskonomy_testloader = torch.utils.data.DataLoader(taskonomy_testset,
                                                       batch_size=args.test_batch,
                                                       shuffle=args.shuffle,
                                                       num_workers=8)

    criterion = nn.MSELoss()

    taskonomy_testframe = pd.read_csv('../data/tiny_taskonomy_rgb_test.csv', delimiter=' ')
    device = torch.device(args.device)
    model = EncoderDecoder(encoder8x16x16(), decoder1000(num_output_channels=num_classes))
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=2e-6)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5,10], gamma=0.1)
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
    images, targets, idxs = next(dataiters)
    idxs = idxs.numpy()
    with torch.no_grad():
        images = images.to(device)
        outputs = model(images)

    images = images.cpu().numpy()
    targets = targets.numpy()
    preds = outputs.data.cpu().numpy()

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(50,10))
    for i in range(4):
        ax = fig.add_subplot(1,4,i+1, projection='3d')
        A,B,C,D,E,F,H,I,J = targets[i]
        a,b,c,d,e,f,h,i,j = preds[i]
        ax.quiver(0,0,0,A,B,C,length=1.0, colors='red')
        ax.quiver(0,0,0,D,E,F,length=1.0, colors='red')
        ax.quiver(0,0,0,H,I,J,length=1.0, colors='red')
        ax.quiver(0,0,0,a,b,c,length=1.0, colors='blue')
        ax.quiver(0,0,0,d,e,f,length=1.0, colors='blue')
        ax.quiver(0,0,0,h,i,j,length=1.0, colors='blue')
        ax.set_xlim(-1,1)
        ax.set_ylim(-1,1)
        ax.set_zlim(-1,1)

    plt.savefig("demo.png")

    plt.figure(2)
    for i in range(args.test_batch):
        # Original image
        plt.subplot(1,args.test_batch,i+1)
        img = Image.open(taskonomy_testframe.iloc[idxs[i],0])
        img.thumbnail((256,256))
        plt.imshow(img)
        plt.axis('off')

    plt.show()
    plt.savefig('images.png', dpi=400)

if __name__ == '__main__':
    main()
