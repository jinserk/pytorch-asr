#!python
import math

import torch
import torch.nn as nn
from torch.autograd import Variable

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


if __name__ == "__main__":
    print("enc")
    enc = ConvEncoderY(x_dim=p.NUM_PIXELS, y_dim=p.NUM_LABELS)
    enc.test()

    print("dec")
    dec = ConvDecoder(x_dim=p.NUM_PIXELS, y_dim=p.NUM_LABELS)
    dec.test()

