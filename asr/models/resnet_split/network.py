#!python
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from asr.utils.misc import Swish, InferenceBatchSoftmax


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        #self.relu1 = nn.ReLU(inplace=True)
        self.relu1 = Swish(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        #self.relu2 = nn.ReLU(inplace=True)
        self.relu2 = Swish(inplace=True)
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu2(out)
        return out


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        #self.relu1 = nn.ReLU(inplace=True)
        self.relu1 = Swish(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        #self.relu2 = nn.ReLU(inplace=True)
        self.relu2 = Swish(inplace=True)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.downsample = downsample
        #self.relu3 = nn.ReLU(inplace=True)
        self.relu3 = Swish(inplace=True)
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu3(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 32
        super(ResNet, self).__init__()

        #self.conv1 = nn.Conv2d(2, self.inplanes, kernel_size=7, stride=(2, 1), padding=3, bias=False)
        #self.bn1 = nn.BatchNorm2d(self.inplanes)
        ##self.relu = nn.ReLU(inplace=True)
        #self.relu = Swish(inplace=True)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv = nn.Sequential(
            nn.Conv2d(2, self.inplanes, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
            nn.BatchNorm2d(self.inplanes),
            #nn.ReLU(inplace=True),
            #nn.Hardtanh(0, 20, inplace=True),
            Swish(inplace=True),
            nn.Conv2d(self.inplanes, self.inplanes, kernel_size=(21, 11), stride=(2, 2), padding=(10, 5)),
            nn.BatchNorm2d(self.inplanes),
            #nn.ReLU(inplace=True),
            #nn.Hardtanh(0, 20, inplace=True)
            Swish(inplace=True),
        )

        # Based on the conv formula (W - F + 2P) // S + 1
        freq_size = np.array([129, 51])
        freq_size = (freq_size - np.array([41, 11]) + 2 * np.array([20, 5])) // 2 + 1
        freq_size = (freq_size - np.array([21, 11]) + 2 * np.array([10, 5])) // 2 + 1

        self.layer1 = self._make_layer(block, self.inplanes, layers[0])
        self.layer2 = self._make_layer(block, self.inplanes, layers[1], stride=(2, 2))
        self.layer3 = self._make_layer(block, self.inplanes, layers[2], stride=(2, 2))
        self.layer4 = self._make_layer(block, self.inplanes, layers[3], stride=(2, 2))

        #self.avgpool = nn.AvgPool2d(3, stride=1, padding=(1, 1))

        freq_size = (freq_size - 3 + 2) // 2 + 1
        freq_size = (freq_size - 3 + 2) // 2 + 1
        freq_size = (freq_size - 3 + 2) // 2 + 1

        self.fc1 = nn.Linear(self.inplanes * np.prod(freq_size), 1024)
        self.do1 = nn.Dropout(p=0.5, inplace=True)
        self.fc2 = nn.Linear(1024, num_classes)

        self.softmax = InferenceBatchSoftmax()

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

    def forward(self, x):
        x = self.conv(x)
        #x = self.bn1(x)
        #x = self.relu(x)
        #x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        #x = self.avgpool(x)
        # BxCxHxW -> BxCxWxH -> BxWxCxH -> TxH
        #x.transpose(2, 3).transpose(1, 2)
        x = self.fc1(x.view(x.size(0), -1))
        x = self.do1(x)
        x = self.fc2(x)
        x = self.softmax(x)
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
