# Ported from StanfordVL/Taskonomy (tensorflow version)
# https://github.com/StanfordVL/taskonomy
# https://github.com/StanfordVL/taskonomy/blob/master/taskbank/lib/models/sample_models.py#L599

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


def conv3x3(in_planes, out_planes, stride=4, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Discriminator(nn.Module):

    def __init__(self, num_input_channels=4, num_output_channels=1, output_activation='sigmoid', norm_layer=None):
        super(Discriminator, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        # input of decoder is 8x16x16
        self.conv1 = conv3x3(in_planes=num_input_channels, out_planes=64, stride=4)
        self.bn1 = norm_layer(64)
        self.conv2 = conv3x3(in_planes=64, out_planes=128, stride=4)
        self.bn2 = norm_layer(128)
        self.conv3 = conv3x3(in_planes=128, out_planes=256, stride=4)
        self.bn3 = norm_layer(256)
        self.conv4 = conv3x3(in_planes=256, out_planes=512, stride=1)
        self.bn4 = norm_layer(512)
        self.conv5 = conv3x3(in_planes=512, out_planes=1, stride=1)
        self.bn5 = norm_layer(1)

        self.conv6 = nn.Conv2d(1, 1, kernel_size=4, stride=1, padding=0, bias=False)

        if output_activation == 'sigmoid':
            self.out = nn.Sigmoid()


    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))

        x = self.conv6(x)
        x = x.view(x.size()[0],-1)
        return x


def discriminator(**kwargs):
    model = Discriminator(**kwargs)

    return model
