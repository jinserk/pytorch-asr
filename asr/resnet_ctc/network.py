#!python
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.autograd import Variable


class View(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, *args):
        return x.view(*self.dim)


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = x.size()
        return x.view(x.size(0), -1)


class Swish(nn.Module):

    def forward(self, x):
        return x * torch.sigmoid(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        #self.relu = nn.ReLU(inplace=True)
        self.relu = Swish()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        #self.relu = nn.ReLU(inplace=True)
        self.relu = Swish()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 32
        super(ResNet, self).__init__()

        #self.conv1 = nn.Conv2d(2, self.inplanes, kernel_size=7, stride=(2, 1), padding=3, bias=False)
        #self.bn1 = nn.BatchNorm2d(self.inplanes)
        ##self.relu = nn.ReLU(inplace=True)
        #self.relu = Swish()
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
            nn.BatchNorm2d(32),
            #nn.Hardtanh(0, 20, inplace=True),
            Swish(),
            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(32),
            #nn.Hardtanh(0, 20, inplace=True)
            Swish(),
            nn.Conv2d(32, 32, kernel_size=(11, 11), stride=(2, 1), padding=(5, 5)),
            nn.BatchNorm2d(32),
            #nn.Hardtanh(0, 20, inplace=True)
            Swish(),
        )
        self.layer1 = self._make_layer(block, self.inplanes, layers[0])
        self.layer2 = self._make_layer(block, 64, layers[1], stride=(2, 1))
        self.layer3 = self._make_layer(block, 128, layers[2], stride=(2, 1))
        self.layer4 = self._make_layer(block, 256, layers[3], stride=(2, 1))
        self.avgpool = nn.AvgPool2d(3, stride=1, padding=(0, 1))
        self.fc1 = nn.Linear(256 * block.expansion, 512)
        self.do1 = nn.Dropout(p=0.5, inplace=True)
        self.fc2 = nn.Linear(512, 512)
        self.do2 = nn.Dropout(p=0.5, inplace=True)
        self.fc3 = nn.Linear(512, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x, softmax=False):
        x = self.conv(x)
        #x = self.bn1(x)
        #x = self.relu(x)
        #x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        # BxCxWxH -> BxHxCxW -> BxTxH
        x = x.transpose(2, 3).transpose(1, 2)
        x = self.fc1(x.view(x.size(0), x.size(1), -1))
        x = self.do1(x)
        x = self.fc2(x)
        x = self.do2(x)
        x = self.fc3(x)
        if softmax:
            return nn.Softmax(dim=2)(x)
        else:
            return x


def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


def resnet152(**kwargs):
    return ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)


if __name__ == "__main__":
    from ..utils import params as p
    print("resnet")
    net = resnet152(num_classes=p.NUM_CTC_LABELS)
