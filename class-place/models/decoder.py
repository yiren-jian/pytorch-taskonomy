# Ported from StanfordVL/Taskonomy (tensorflow version)
# https://github.com/StanfordVL/taskonomy
# https://github.com/StanfordVL/taskonomy/blob/master/taskbank/lib/models/sample_models.py#L599

import torch.nn as nn
import torch.utils.model_zoo as model_zoo


def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def deconv3x3(in_planes, out_planes, stride=2):
    """3x3 transpose convolution with padding"""
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class Decoder(nn.Module):

    def __init__(self, num_output_channels=3, output_activation='softmax', norm_layer=None):
        super(Decoder, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        # input of decoder is 8x16x16
        self.fc1 = nn.Linear(2048,2048)
        self.fc2 = nn.Linear(2048,num_output_channels)

        if output_activation == 'softmax':
            self.out = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, 2048)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x


def decoder1000(**kwargs):
    model = Decoder(**kwargs)

    return model
