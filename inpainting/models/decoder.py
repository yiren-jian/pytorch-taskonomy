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
            norm_layer = nn.BatchNorm2d
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        # input of decoder is 8x16x16
        self.conv1 = conv3x3(in_planes=8, out_planes=1024)
        self.bn1 = norm_layer(1024)
        self.conv2 = conv3x3(in_planes=1024, out_planes=1024)
        self.bn2 = norm_layer(1024)
        self.conv3 = conv3x3(in_planes=1024, out_planes=1024)
        self.bn3 = norm_layer(1024)
        self.conv4 = conv3x3(in_planes=1024, out_planes=512)
        self.bn4 = norm_layer(512)
        self.conv5 = conv3x3(in_planes=512, out_planes=256)
        self.bn5 = norm_layer(256)
        self.conv6 = conv3x3(in_planes=256, out_planes=256)
        self.bn6 = norm_layer(256)
        self.conv7 = conv3x3(in_planes=256, out_planes=128)
        self.bn7 = norm_layer(128)
        self.deconv8 = deconv3x3(in_planes=128, out_planes=64, stride=2)
        self.bn8 = norm_layer(64)
        self.conv9 = conv3x3(in_planes=64, out_planes=64)
        self.bn9 = norm_layer(64)
        self.deconv10 = deconv3x3(in_planes=64, out_planes=32, stride=2)
        self.bn10 = norm_layer(32)
        self.conv11 = conv3x3(in_planes=32, out_planes=32)
        self.bn11 = norm_layer(32)
        self.deconv12 = deconv3x3(in_planes=32, out_planes=16, stride=2)
        self.bn12 = norm_layer(16)
        self.conv13 = conv3x3(in_planes=16, out_planes=32)
        self.bn13 = norm_layer(32)
        self.deconv14 = deconv3x3(in_planes=32, out_planes=16, stride=2)
        self.bn14 = norm_layer(16)
        self.conv15 = conv3x3(in_planes=16, out_planes=num_output_channels)
        self.bn15 = norm_layer(num_output_channels)

        if output_activation == 'softmax':
            self.out = nn.Softmax(dim=1)


    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.relu(self.bn6(self.conv6(x)))
        x = self.relu(self.bn7(self.conv7(x)))
        x = self.relu(self.bn8(self.deconv8(x, output_size=[32,32]))) # hard-code output_size
                                                                      # otherwise it will be 31x31
        x = self.relu(self.bn9(self.conv9(x)))
        x = self.relu(self.bn10(self.deconv10(x, output_size=[64,64])))
        x = self.relu(self.bn11(self.conv11(x)))
        x = self.relu(self.bn12(self.deconv12(x, output_size=[128,128])))
        x = self.relu(self.bn13(self.conv13(x)))
        x = self.relu(self.bn14(self.deconv14(x, output_size=[256,256])))
        x = self.conv15(x)

        return x


def decoder256x256(**kwargs):
    model = Decoder(**kwargs)

    return model
