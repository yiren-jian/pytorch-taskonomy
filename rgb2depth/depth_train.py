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

from taskonomy_dataset import TaskonomyDatasetZbuffer
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
parser.add_argument('--brenta', action='store_true')
args = parser.parse_args()

num_classes = 1
if os.path.isdir('./checkpoints') == False:
    os.makedirs('./checkpoints')
ckpt = './checkpoints/rgb2depth-ckpt.pth'
best = './checkpoints/rgb2depth-best.pth'

def main():
    taskonomy_transform = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize((0.5456, 0.5176, 0.4863),
                                                                   (0.1825, 0.1965, 0.2172))])
    taskonomy_trainset = TaskonomyDatasetZbuffer('../data/tiny_taskonomy_rgb_train.csv',
                                                 '/depth_zbuffer',
                                                 'depth_zbuffer.png',
                                                 resize256 = True,
                                                 transform=taskonomy_transform,
                                                 brenta = args.brenta)
    taskonomy_trainloader = torch.utils.data.DataLoader(taskonomy_trainset,
                                                        batch_size=args.train_batch,
                                                        shuffle=True,
                                                        num_workers=8)
    taskonomy_testset = TaskonomyDatasetZbuffer('../data/tiny_taskonomy_rgb_test.csv',
                                                '/depth_zbuffer',
                                                'depth_zbuffer.png',
                                                resize256 = True,
                                                transform=taskonomy_transform,
                                                brenta = args.brenta)
    taskonomy_testloader = torch.utils.data.DataLoader(taskonomy_testset,
                                                       batch_size=args.test_batch,
                                                       shuffle=True,
                                                       num_workers=8)

    # Define the loss function for different tasks.
    criterion = nn.L1Loss()

    device = torch.device(args.device)
    model = EncoderDecoder(encoder8x16x16(), decoder256x256(num_output_channels=num_classes))
    # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=2e-6)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
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
            # keep targets with shape [N,1,h,w]
            targets = targets.unsqueeze(dim=1)
            images, targets = images.to('cuda'), targets.to('cuda')


            # forward + backward + optimize
            outputs = model(images)
            loss = criterion(outputs, targets)

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
                            loss='{:06.4f}'.format(running_loss/total),
                            lr='{:03.1e}'.format(LR))
            bar.update()

    return running_loss / total


def train_with_cGAN(epoch, trainloader, model_G, optimizer_G, model_D, optimizer_D, criterionL1, criterionGAN):
    model_G.train()
    model_D.train()
    running_loss = 0.0
    total = 0
    end = time.time()

    for param_group in optimizer_G.param_groups:
        G_LR = param_group['lr']
    for param_group in optimizer_D.param_groups:
        D_LR = param_group['lr']

    with tqdm(total=len(trainloader), unit=' batch(s)') as bar:
        bar.set_description('Training epoch: %d' % epoch)
        for i, data in enumerate(trainloader, 0):
            images, targets, idxs = data         # get the inputs
            targets = targets.unsqueeze(dim=1)   # keep targets with shape [N,1,h,w]
            images, targets = images.to('cuda'), targets.to('cuda')

            real_A, real_B = images, targets                    # use the convention in pix2pix

            # Create the labels of True and False for BCEWithLogitsLoss.
            b_size = real_A.size(0)
            False_label = torch.full((b_size,1), 0, device='cuda')
            True_label = torch.full((b_size,1), 1, device='cuda')

            fake_B = model_G(real_A)                            # forward pass, compute fake images: G(A)

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            set_requires_grad([model_D], requires_grad=True)   # enable backprop for D
            optimizer_D.zero_grad()                             # Set D's gradients to zero
            # Fake
            fake_AB = torch.cat((real_A, fake_B), 1)            # we use conditional GANs; we need to feed both input and output to the discriminator
            pred_fake = model_D(fake_AB.detach())
            loss_D_fake = criterionGAN(pred_fake, False_label)
            # Real
            real_AB = torch.cat((real_A, real_B), 1)
            pred_real = model_D(real_AB)
            loss_D_real = criterionGAN(pred_real, True_label)
            # Combine loss and calculate gradients
            loss_D = (loss_D_fake + loss_D_real) * 0.5
            loss_D.backward()                                  # calculate gradients for D
            optimizer_D.step()                                 # update D's weights

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            set_requires_grad([model_D], False)                # D requires no gradients when optimizing G
            optimizer_G.zero_grad()                            # set G's gradients to zero
            # First, G(A) should fake the discriminator
            fake_AB = torch.cat((real_A, fake_B), 1)
            pred_fake = model_D(fake_AB)
            loss_G_GAN = criterionGAN(pred_fake, True_label)
            # Second, G(A) = B
            loss_G_L1 = criterionL1(fake_B, real_B)
            # combine loss and calculate gradients
            loss_G = 0.004*loss_G_GAN + 0.996*loss_G_L1
            loss_G.backward()                                  # calculate graidents for G
            optimizer_G.step()                                 # udpate G's weights

            # print statistics
            running_loss += loss_G_L1.item()
            total += 1

            # measure elapsed time
            batch_time = (time.time() - end)

            # progress bar in tqdm
            bar.set_postfix(time='{:03.1f}'.format(batch_time),
                            G_loss='{:06.4f}'.format(running_loss/total),
                            G_lr='{:03.1e}'.format(G_LR))
            bar.update()

    return running_loss / total


def validation(testloader, model, criterion):
    model.eval()
    running_loss = 0.0
    total = 0
    end = time.time()

    with torch.no_grad():
        with tqdm(total=len(testloader), unit=' batch(s)') as bar:
            bar.set_description('Testing:')
            for i, (images, targets, idxs) in enumerate(testloader):
                images = images.to('cuda')
                # keep targets with shape [N,1,h,w]
                targets = targets.unsqueeze(dim=1)
                targets = targets.to('cuda')

                outputs = model(images)
                loss = criterion(outputs, targets)

                # print statistics
                running_loss += loss.item()
                total += 1

                # measure elapsed time
                batch_time = (time.time() - end)

                # progress bar in tqdm
                bar.set_postfix(time='{:03.1f}'.format(batch_time),
                                loss='{:06.4f}'.format(running_loss/total))
                bar.update()

    mean_loss = running_loss/total

    return mean_loss

if __name__ == '__main__':
    main()
