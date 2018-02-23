#!python
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.models.densenet import _DenseLayer, _DenseBlock

from pyro.nn import ClippedSoftmax, ClippedSigmoid

import utils.params as p


class View(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, *args):
        return x.view(*self.dim)


class MultiOut(nn.ModuleList):

    def __init__(self, modules):
        super().__init__(modules)

    def forward(self, *args, **kwargs):
        return (m.forward(*args, **kwargs) for m in self)


class Swish(nn.Module):

    def forward(self, x):
        return x * nn.functional.sigmoid(x)


class MlpEncoderY(nn.Module):

    def __init__(self, x_dim=p.NUM_PIXELS, y_dim=p.NUM_LABELS,
                 h_dims=p.NUM_HIDDEN, eps=p.EPS):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        # network
        layers = [
            nn.Linear(x_dim, h_dims[0]),
            Swish(),
        ]
        for l in range(1, len(h_dims) - 1):
            layers += [
                nn.Linear(h_dims[l], h_dims[l + 1]),
                Swish(),
            ]
        layers += [
            nn.Linear(h_dims[-1], y_dim),
            ClippedSoftmax(eps, dim=1),
        ]
        self.hidden = nn.Sequential(*layers)

    def forward(self, xs, *args, **kwargs):
        ys = self.hidden.forward(xs, *args, **kwargs)
        return ys


class MlpEncoderZ(nn.Module):

    def __init__(self, x_dim=p.NUM_PIXELS, y_dim=p.NUM_LABELS, z_dim=p.NUM_STYLE,
                 h_dims=p.NUM_HIDDEN, eps=p.EPS):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
        # network
        layers = [
            nn.Linear(x_dim + y_dim, h_dims[0]),
            Swish(),
        ]
        for l in range(1, len(h_dims) - 1):
            layers += [
                nn.Linear(h_dims[l], h_dims[l + 1]),
                Swish(),
            ]
        layers += [
            MultiOut([
                nn.Linear(h_dims[-1], z_dim),   # for z mean
                nn.Linear(h_dims[-1], z_dim),   # for z log std
            ]),
        ]
        self.hidden = nn.Sequential(*layers)

    def forward(self, xs, ys, *args, **kwargs):
        z_mean, z_log_std = self.hidden.forward(torch.cat([xs, ys], -1), *args, **kwargs)
        return z_mean, torch.exp(z_log_std)


class MlpDecoder(nn.Module):

    def __init__(self, x_dim=p.NUM_PIXELS, y_dim=p.NUM_LABELS, z_dim=p.NUM_STYLE,
                 h_dims=p.NUM_HIDDEN, eps=p.EPS):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
        # network
        layers = [
            nn.Linear(z_dim + y_dim, h_dims[0]),
            Swish(),
        ]
        for l in range(1, len(h_dims) - 1):
            layers += [
                nn.Linear(h_dims[l], h_dims[l + 1]),
                Swish(),
            ]
        layers += [
            nn.Linear(h_dims[-1], x_dim),
            ClippedSigmoid(eps),
        ]
        self.hidden = nn.Sequential(*layers)

    def forward(self, zs, ys, *args, **kwargs):
        xs = self.hidden.forward(torch.cat([zs, ys], -1), *args, **kwargs)
        return xs


class ConvEncoderY(nn.Module):

    def __init__(self, x_dim=p.NUM_PIXELS, y_dim=p.NUM_LABELS, softmax=True, eps=p.EPS):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        # network
        layers = [
            View(dim=(-1, p.CHANNEL, p.WIDTH, p.HEIGHT)),
            nn.Conv2d(p.CHANNEL, 16, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),  # 2x129x21 -> 16x65x11
            nn.BatchNorm2d(16),
            Swish(),
            nn.Conv2d(16, 32, (5, 5), (2, 2), (2, 2)),  # 16x65x11 -> 32x33x6
            nn.BatchNorm2d(32),
            Swish(),
            nn.Conv2d(32, 64, (5, 5), (2, 2), (2, 2)),  # 32x33x6 -> 64x17x3
            nn.BatchNorm2d(64),
            Swish(),
            nn.Conv2d(64, 128, (5, 5), (2, 2), (2, 2)),  # 64x17x3 -> 128x9x2
            nn.BatchNorm2d(128),
            Swish(),
            nn.Conv2d(128, 256, (5, 5), (2, 2), (2, 2)),  # 128x9x2 -> 256x5x1
            nn.BatchNorm2d(256),
            Swish(),
            View(dim=(-1, 256 * 5 * 1)),
            nn.Linear(256 * 5 * 1, y_dim),
            nn.BatchNorm2d(y_dim),
        ]
        if softmax:
            layers.append(ClippedSoftmax(eps, dim=1))
        self.hidden = nn.Sequential(*layers)

    def forward(self, xs, *args, **kwargs):
        ys = self.hidden.forward(xs, *args, **kwargs)
        return ys

    def test(self):
        xs = torch.randn(10, self.x_dim)
        print(xs.shape)
        xs = Variable(xs)
        for h in self.hidden:
            xs = h.forward(xs)
            print(xs.shape)


class ConvDecoder(nn.Module):

    def __init__(self, x_dim=p.NUM_PIXELS, y_dim=p.NUM_LABELS, z_dim=p.NUM_STYLE, eps=p.EPS):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
        # network
        layers = [
            nn.Linear(z_dim + y_dim, 256 * 5 * 1),
            Swish(),
            View(dim=(-1, 256, 5, 1)),
            nn.ConvTranspose2d(256, 128, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), output_padding=(0, 1)),  # 256x5x1 -> 128x9x2
            Swish(),
            nn.ConvTranspose2d(128, 64, (5, 5), (2, 2), (2, 2)),  # 128x9x2 -> 64x17x3
            Swish(),
            nn.ConvTranspose2d(64, 32, (5, 5), (2, 2), (2, 2), output_padding=(0, 1)),  # 64x17x3 -> 32x33x6
            Swish(),
            nn.ConvTranspose2d(32, 16, (5, 5), (2, 2), (2, 2)),  # 32x33x6 -> 16x65x11
            Swish(),
            nn.ConvTranspose2d(16, p.CHANNEL, (5, 5), (2, 2), (2, 2)),  # 16x65x11 -> 2x129x21
            Swish(),
            View(dim=(-1, p.CHANNEL * p.WIDTH * p.HEIGHT)),
            nn.Linear(p.CHANNEL * p.WIDTH * p.HEIGHT, x_dim),
            ClippedSigmoid(eps)
        ]
        self.hidden = nn.Sequential(*layers)

    def forward(self, zs, ys, *args, **kwargs):
        xs = self.hidden.forward(torch.cat([zs, ys], -1), *args, **kwargs)
        return xs

    def test(self):
        zs = torch.randn(10, self.z_dim)
        ys = torch.randn(10, self.y_dim)
        zs, ys = Variable(zs), Variable(ys)
        xs = torch.cat([zs, ys], -1)
        for h in self.hidden:
            xs = h.forward(xs)
            print(xs.shape)


class _DenseLayer(nn.Sequential):

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module("norm.1", nn.BatchNorm2d(num_input_features)),
        self.add_module("swish.1", Swish()),
        self.add_module("conv.1", nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module("norm.2", nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module("swish.2", Swish()),
        self.add_module("conv.2", nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super().forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module(f"denselayer{i+1}", layer)


class _Transition(nn.Sequential):

    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module("norm", nn.BatchNorm2d(num_input_features))
        self.add_module("swish", Swish())
        self.add_module("conv", nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module("pool", nn.AvgPool2d(kernel_size=3, stride=2, padding=1))


class DenseEncoderY(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
    """
    def __init__(self, x_dim=p.NUM_PIXELS, y_dim=p.NUM_LABELS, growth_rate=4, block_config=(6, 12, 24, 48, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, softmax=True, eps=p.EPS):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        # First convolution
        self.hidden = nn.Sequential(OrderedDict([
            ("view.i", View(dim=(-1, 2, 129, 21))),
            ("conv.i", nn.Conv2d(2, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
            ("norm.i", nn.BatchNorm2d(num_init_features)),
            ("swish.i", Swish()),
            ("pool.i", nn.MaxPool2d(kernel_size=3, stride=(2, 1), padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.hidden.add_module(f"denseblock{i+1}", block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.hidden.add_module(f"transition{i+1}", trans)
                num_features = num_features // 2

        # Final layer
        self.hidden.add_module("norm.f", nn.BatchNorm2d(num_features))
        self.hidden.add_module("swish.f", Swish())
        self.hidden.add_module("pool.f", nn.AvgPool2d(kernel_size=2, stride=1))
        self.hidden.add_module("view.f", View(dim=(-1, 195 * 4)))
        self.hidden.add_module("class.f", nn.Linear(195 * 4, y_dim))
        if softmax:
            self.hidden.add_module("softmax.f", ClippedSoftmax(eps, dim=1))

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.hidden(x)
        return out

    def test(self):
        xs = torch.randn(10, self.x_dim)
        print(xs.shape)
        xs = Variable(xs)
        for h in self.hidden:
            xs = h.forward(xs)
            print(xs.shape)


if __name__ == "__main__":
    #print("enc")
    #enc = ConvEncoderY(x_dim=p.NUM_PIXELS, y_dim=p.NUM_LABELS)
    #enc.test()

    #print("dec")
    #dec = ConvDecoder(x_dim=p.NUM_PIXELS, y_dim=p.NUM_LABELS)
    #dec.test()

    print("dense")
    dense = DenseEncoderY(x_dim=p.NUM_PIXELS, y_dim=p.NUM_LABELS)
    dense.test()
