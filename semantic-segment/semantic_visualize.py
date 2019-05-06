from __future__ import print_function

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

from models.encoder import encoder8x16x16
from models.decoder import decoder256x256
from models.encoder_decoder import EncoderDecoder

from taskonomy_dataset import TaskonomyDatasetSemSeg
from utils.metrics import runningScore

from PIL import Image
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from skimage.color.colorlabel import label2rgb
plt.switch_backend('agg')

import sys
import os
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--train_batch', default=32, type=int)
parser.add_argument('--test_batch', default=4, type=int)
parser.add_argument('--epochs', default=20, type=int)
parser.add_argument('--shuffle', action='store_true', default=False)
parser.add_argument('--brenta', action='store_true', default=False)
args = parser.parse_args()

colors = ['grey',
          'orange',
          'green',
          'red',
          'lime',
          'yellow',
          'blue',
          'olive',
          'cyan',
          'brown',
          'firebrick',
          'greenyellow',
          'gold',
          'indigo',
          'slateblue',
          'lightpink',
          'purple']

classes = ['background',
           'bottle',
           'chair',
           'couch',
           'potted plant',
           'bed',
           'dining table',
           'toilet',
           'tv',
           'microwave',
           'oven',
           'toaster',
           'sink',
           'refrigerator',
           'book',
           'clock',
           'vase']

def colored(code):
    colored = []
    for c in code:
        colored.append(colors[c])
    return colored


def main():
    num_classes = 17
    taskonomy_transform = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize((0.5456, 0.5176, 0.4863),
                                                                   (0.1825, 0.1965, 0.2172))])
    taskonomy_trainset = TaskonomyDatasetSemSeg('../data/tiny_taskonomy_rgb_train.csv',
                                                '/segment_semantic',
                                                'segmentsemantic.png',
                                                resize256 = True,
                                                transform=taskonomy_transform,
                                                brenta = args.brenta)
    taskonomy_trainloader = torch.utils.data.DataLoader(taskonomy_trainset,
                                                        batch_size=args.train_batch,
                                                        shuffle=True,
                                                        num_workers=8)
    taskonomy_testset = TaskonomyDatasetSemSeg('../data/tiny_taskonomy_rgb_test.csv',
                                               '/segment_semantic',
                                               'segmentsemantic.png',
                                               resize256 = True,
                                               transform=taskonomy_transform,
                                               brenta = args.brenta)
    taskonomy_testloader = torch.utils.data.DataLoader(taskonomy_testset,
                                                       batch_size=args.test_batch,
                                                       shuffle=True,
                                                       num_workers=8)

    taskonomy_testframe = pd.read_csv('../data/tiny_taskonomy_rgb_test.csv', delimiter=' ')
    device = torch.device(args.device)
    model = EncoderDecoder(encoder8x16x16(), decoder256x256(num_output_channels=num_classes))
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoints'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoints/semantic-segment-best.pth')
    model.load_state_dict(checkpoint['model_state'])
    model = torch.nn.DataParallel(model).to(device)
    print('  Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    model.eval()

    dataiters = iter(taskonomy_testloader)
    images, labels, idxs = next(dataiters)
    idxs = idxs.numpy()
    labels = labels.squeeze(dim=1).numpy()
    images = images.to(device)
    outputs = model(images)

    images = images.cpu().numpy()
    pred = outputs.data.max(1)[1].cpu().numpy()

    plt.figure(1)
    for i in range(args.test_batch):
        # ground truth
        plt.subplot(2,args.test_batch,i+1)
        img = Image.open(taskonomy_testframe.iloc[idxs[i],0])
        img.thumbnail((256,256))
        npimg = np.array(img) / 255.0
        nplabel = labels[i]
        C1 = np.unique(nplabel)
        res1 = label2rgb(nplabel, npimg, colors=colored(C1))
        plt.imshow(res1)
        for c in range(len(C1)):
            print_text = str(classes[C1[c]]) + '-' + str(colors[C1[c]])
            plt.text(0,275+c*25, print_text, fontsize=5)
        plt.axis('off')

        # predicted
        plt.subplot(2,args.test_batch,i+1*args.test_batch+1)
        nppred = pred[i]
        C2 = np.unique(nppred)
        res2 = label2rgb(nppred, npimg, colors=colored(C2))
        plt.imshow(res2)
        for c in range(len(C2)):
            print_text = str(classes[C2[c]]) + '-' + str(colors[C2[c]])
            plt.text(0,275+c*25, print_text, fontsize=5)
        plt.axis('off')

    plt.show()
    plt.savefig('demo.png', dpi=300)


if __name__ == '__main__':
    main()
