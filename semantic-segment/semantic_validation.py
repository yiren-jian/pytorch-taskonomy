from __future__ import print_function

import argparse
import numpy as np
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

from taskonomy_dataset import TaskonomyDatasetSemSeg, TaskonomyDatasetSemSegAmir
from utils.metrics import runningScore

import os
import time
from tqdm import tqdm

import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")


parser = argparse.ArgumentParser()
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--mode', default='my', type=str)
parser.add_argument('--train_batch', default=32, type=int)
parser.add_argument('--test_batch', default=32, type=int)
parser.add_argument('--epochs', default=30, type=int)
parser.add_argument('--lr', default=0.0005, type=float)
parser.add_argument('--resume', action='store_true')
parser.add_argument('--brenta', action='store_true', default=False)
args = parser.parse_args()

num_classes = 17
ckpt = './checkpoints/semantic-segment-ckpt.pth'
best = './checkpoints/semantic-segment-best.pth'

def main():
    assert args.mode=='my' or args.mode=='Amir', 'mode has to be my or Amir!'
    taskonomy_transform = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize((0.5456, 0.5176, 0.4863),
                                                                   (0.1825, 0.1965, 0.2172))])
    if args.mode == 'my':
        taskonomy_testset = TaskonomyDatasetSemSeg('../data/tiny5k_taskonomy_rgb_test.csv',
                                                   '/segment_semantic',
                                                   'segmentsemantic.png',
                                                   resize256 = True,
                                                   transform=taskonomy_transform,
                                                   brenta = args.brenta)
    if args.mode == 'Amir':
        taskonomy_testset = TaskonomyDatasetSemSegAmir('../data/tiny5k_taskonomy_rgb_test.csv',
                                                       '/segment_semantic',
                                                       'segmentsemantic.png',
                                                       resize256 = True,
                                                       transform=taskonomy_transform,
                                                       brenta = args.brenta)
    taskonomy_testloader = torch.utils.data.DataLoader(taskonomy_testset,
                                                       batch_size=args.test_batch,
                                                       shuffle=False,
                                                       num_workers=8)

    criterion = nn.NLLLoss()

    device = torch.device(args.device)
    model = EncoderDecoder(encoder8x16x16(), decoder256x256(num_output_channels=num_classes))
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=2e-6)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20], gamma=0.1)
    best_IoU = 0

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoints'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(ckpt)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state'])
        model = torch.nn.DataParallel(model).to(device)
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        best_IoU = checkpoint['best_IoU']
    else:
        start_epoch = 0
        model = torch.nn.DataParallel(model).to(device)
        cudnn.benchmark = True
    print('  Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    if args.mode == 'my':
        MyValidation(taskonomy_testloader, model)
    elif args.mode == 'Amir':
        AmirValidation(taskonomy_testloader)


def MyValidation(testloader, model):
    model.eval()
    running_metrics = runningScore(num_classes)
    end = time.time()

    with torch.no_grad():
        with tqdm(total=len(testloader), unit=' batch(s)') as bar:
            bar.set_description('Testing:')
            for i, (images, labels, idxs) in enumerate(testloader):
                images = images.to('cuda')

                outputs = model(images)
                pred = outputs.data.max(1)[1].cpu().numpy()

                gt = labels.numpy()
                running_metrics.update(gt, pred)

                score, class_iou = running_metrics.get_scores()
                mean_IoU = score['Mean IoU : \t']
                mean_Acc = score["Mean Acc : \t"]

                # measure elapsed time
                batch_time = (time.time() - end)

                # progress bar in tqdm
                bar.set_postfix(time='{:03.1f}'.format(batch_time),
                                IoU='{:05.3f}'.format(mean_IoU),
                                Acc='{:05.3f}'.format(mean_Acc))
                bar.update()

    score, class_iou = running_metrics.get_scores()
    mean_IoU = score['Mean IoU : \t']

    for k, v in score.items():
        print(k, v)

    for i in range(num_classes):
        print(i, class_iou[i])

    return mean_IoU


def AmirValidation(testloader):
    running_metrics = runningScore(num_classes)
    end = time.time()

    with torch.no_grad():
        with tqdm(total=len(testloader), unit=' batch(s)') as bar:
            bar.set_description('Testing:')
            for i, (labels, preds, idxs) in enumerate(testloader):

                pred = preds.data.max(1)[1].cpu().numpy()
                gt = labels.numpy()
                running_metrics.update(gt, pred)

                score, class_iou = running_metrics.get_scores()
                mean_IoU = score['Mean IoU : \t']
                mean_Acc = score["Mean Acc : \t"]

                # measure elapsed time
                batch_time = (time.time() - end)

                # progress bar in tqdm
                bar.set_postfix(time='{:03.1f}'.format(batch_time),
                                IoU='{:05.3f}'.format(mean_IoU),
                                Acc='{:05.3f}'.format(mean_Acc))
                bar.update()

    score, class_iou = running_metrics.get_scores()
    mean_IoU = score['Mean IoU : \t']

    for k, v in score.items():
        print(k, v)

    for i in range(num_classes):
        print(i, class_iou[i])

    return mean_IoU

if __name__ == '__main__':
    main()
