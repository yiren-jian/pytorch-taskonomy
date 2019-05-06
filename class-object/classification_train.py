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
from models.decoder import decoder1000
from models.encoder_decoder import EncoderDecoder

from taskonomy_dataset import TaskonomyDataset
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
args = parser.parse_args()

num_classes = 1000
ckpt = './checkpoints/class_1000-ckpt.pth'
best = './checkpoints/class_1000-best.pth'

def main():
    taskonomy_transform = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize((0.5456, 0.5176, 0.4863),
                                                                   (0.1825, 0.1965, 0.2172))])
    taskonomy_trainset = TaskonomyDataset('./dataloader_csv/tiny_taskonomy_class_1000_train.csv',
                                          resize256 = True,
                                          transform=taskonomy_transform)
    taskonomy_trainloader = torch.utils.data.DataLoader(taskonomy_trainset,
                                                        batch_size=args.train_batch,
                                                        shuffle=True,
                                                        num_workers=8)
    taskonomy_testset = TaskonomyDataset('./dataloader_csv/tiny_taskonomy_class_1000_test.csv',
                                         resize256 = True,
                                         transform=taskonomy_transform)
    taskonomy_testloader = torch.utils.data.DataLoader(taskonomy_testset,
                                                       batch_size=args.test_batch,
                                                       shuffle=False,
                                                       num_workers=8)

    criterion = nn.KLDivLoss(reduction='batchmean')

    device = torch.device(args.device)
    model = EncoderDecoder(encoder8x16x16(), decoder1000(num_output_channels=num_classes))
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=2e-6)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5,10], gamma=0.1)
    best_loss = float('inf')

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
        best_loss = checkpoint['best_loss']
    else:
        start_epoch = 0
        model = torch.nn.DataParallel(model).to(device)
        cudnn.benchmark = True
    print('  Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))


    for epoch in range(start_epoch, args.epochs):
        train(epoch, taskonomy_trainloader, model, optimizer, criterion)
        loss = validation(taskonomy_testloader, model, criterion)
        scheduler.step()
        if loss < best_loss:
            best_loss = loss
            print('best model is found with loss: %.3f' %(best_loss))
            state = {
                        'epoch': epoch + 1,
                        'model_state': model.module.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'scheduler_state': scheduler.state_dict(),
                        'best_loss': best_loss,
                    }
            save_path = best
            torch.save(state, save_path)

        state = {
                    'epoch': epoch + 1,
                    'model_state': model.module.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'scheduler_state': scheduler.state_dict(),
                    'best_loss': best_loss,
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
                            loss='{:05.4f}'.format(running_loss/total),
                            lr='{:03.1e}'.format(LR))
            bar.update()

    return running_loss / total


def validation(testloader, model, criterion):
    model.eval()
    running_loss = 0.0
    total = 0.0
    end = time.time()

    with torch.no_grad():
        with tqdm(total=len(testloader), unit=' batch(s)') as bar:
            bar.set_description('Testing:')
            for i, (images, labels, idxs) in enumerate(testloader):
                images = images.to('cuda')
                targets = labels.to('cuda')

                outputs = model(images)
                logprobs = nn.LogSoftmax(dim=1)(outputs)
                loss = criterion(logprobs, targets)

                running_loss += loss.item()
                total += 1

                # measure elapsed time
                batch_time = (time.time() - end)

                # progress bar in tqdm
                bar.set_postfix(time='{:03.1f}'.format(batch_time),
                                loss='{:05.4f}'.format(running_loss/total))
                bar.update()


    return running_loss/total

if __name__ == '__main__':
    main()
