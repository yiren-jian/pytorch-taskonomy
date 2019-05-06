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

from taskonomy_dataset import TaskonomyDatasetSemSeg
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
parser.add_argument('--train_batch', default=32, type=int)
parser.add_argument('--test_batch', default=32, type=int)
parser.add_argument('--epochs', default=15, type=int)
parser.add_argument('--lr', default=0.0005, type=float)
parser.add_argument('--resume', action='store_true')
parser.add_argument('--brenta', action='store_true', default=False)
parser.add_argument('--validate_only', action='store_true')
args = parser.parse_args()

num_classes = 17
if os.path.isdir('./checkpoints') == False:
    os.makedirs('./checkpoints')
ckpt = './checkpoints/semantic-segment-ckpt.pth'
best = './checkpoints/semantic-segment-best.pth'

def main():
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

    criterion = nn.NLLLoss()

    device = torch.device(args.device)
    model = EncoderDecoder(encoder8x16x16(), decoder256x256(num_output_channels=num_classes))
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=2e-6)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5,10], gamma=0.1)
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

    if args.validate_only:
        MyValidation(taskonomy_testloader, model)
    else:
        for epoch in range(start_epoch, args.epochs):
            train(epoch, taskonomy_trainloader, model, optimizer, criterion)
            IoU = validation(taskonomy_testloader, model)
            scheduler.step()
            if IoU > best_IoU:
                best_IoU = IoU
                print('best model is found with IoU: %.3f' %(best_IoU))
                state = {
                            'epoch': epoch + 1,
                            'model_state': model.module.state_dict(),
                            'optimizer_state': optimizer.state_dict(),
                            'scheduler_state': scheduler.state_dict(),
                            'best_IoU': best_IoU,
                        }
                save_path = best
                torch.save(state, save_path)

            state = {
                        'epoch': epoch + 1,
                        'model_state': model.module.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'scheduler_state': scheduler.state_dict(),
                        'best_IoU': best_IoU,
                    }
            save_path = ckpt
            torch.save(state, save_path)


def train(epoch, trainloader, model, optimizer, criterion):
    model.train()
    running_loss = 0.0
    total = 0
    end = time.time()

    for param_group in optimizer.param_groups:
        LR = param_group['lr']

    with tqdm(total=len(trainloader), unit=' batch(s)') as bar:
        bar.set_description('Training epoch: %d' % epoch)
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            images, targets, idxs = data
            targets = targets.squeeze(dim=1)
            images, targets = images.to('cuda'), targets.to('cuda')


            # forward + backward + optimize
            outputs = model(images)
            logprobs = nn.LogSoftmax(dim=1)(outputs)
            loss = criterion(logprobs, targets)

            # zero the parameter gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            total += 1

            # measure elapsed time
            batch_time = (time.time() - end)

            # progress bar in tqdm
            bar.set_postfix(time='{:03.1f}'.format(batch_time),
                            loss='{:05.3f}'.format(running_loss/total),
                            lr='{:03.1e}'.format(LR))
            bar.update()

    return running_loss / total


def validation(testloader, model):
    model.eval()
    running_metrics = runningScore(num_classes)
    end = time.time()

    with torch.no_grad():
        with tqdm(total=len(testloader), unit=' batch(s)') as bar:
            bar.set_description('Testing:')
            for i, (images, target, idxs) in enumerate(testloader):
                images = images.to('cuda')

                outputs = model(images)
                pred = outputs.data.max(1)[1].cpu().numpy()

                gt = target.numpy()
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

    return mean_IoU


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
