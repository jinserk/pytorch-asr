#!python
import math

import torch
import torch.nn as nn
from torch.autograd import Variable

from pyro.nn import ClippedSoftmax, ClippedSigmoid

NUM_PIXELS = 2 * 129 * 9
NUM_DIGITS = 187
NUM_HIDDEN = [256, 256]
NUM_STYLE = 200
EPS = 1e-9


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

    def __init__(self, x_dim=NUM_PIXELS, y_dim=NUM_DIGITS,
                 h_dims=NUM_HIDDEN, eps=EPS):
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

    def __init__(self, x_dim=NUM_PIXELS, y_dim=NUM_DIGITS, z_dim=NUM_STYLE,
                 h_dims=NUM_HIDDEN, eps=EPS):
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

    def __init__(self, x_dim=NUM_PIXELS, y_dim=NUM_DIGITS, z_dim=NUM_STYLE,
                 h_dims=NUM_HIDDEN, eps=EPS):
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

    def __init__(self, x_dim=NUM_PIXELS, y_dim=NUM_DIGITS, softmax=True, eps=EPS):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        # network
        layers = [
            View(dim=(-1, 2, 129, 9)),
            nn.Conv2d(2, 16, kernel_size=(16, 8), stride=(4, 2), padding=(6, 3)),  # 2x129x9 -> 16x32x4
            nn.BatchNorm2d(16),
            Swish(),
            nn.Conv2d(16, 32, (16, 8), (4, 2), (6, 3)),  # 16x32x4 -> 32x8x2
            nn.BatchNorm2d(32),
            Swish(),
            nn.Conv2d(32, 64, (16, 8), (4, 2), (6, 3)),  # 32x8x2 -> 64x2x1
            nn.BatchNorm2d(64),
            Swish(),
            View(dim=(-1, 64 * 2)),
            nn.Linear(64 * 2, y_dim),
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

    def __init__(self, x_dim=NUM_PIXELS, y_dim=NUM_DIGITS, z_dim=NUM_STYLE, eps=EPS):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
        # network
        layers = [
            nn.Linear(z_dim + y_dim, 64 * 2),
            Swish(),
            View(dim=(-1, 64, 2, 1)),
            nn.ConvTranspose2d(64, 32, kernel_size=(16, 8), stride=(4, 2), padding=(6, 3)),  # 64x2x1 -> 32x8x2
            Swish(),
            nn.ConvTranspose2d(32, 16, (16, 8), (4, 2), (6, 3)),  # 32x8x2 -> 16x32x4
            Swish(),
            nn.ConvTranspose2d(16, 2, (16, 8), (4, 2), (6, 3)),  # 16x32x4 -> 2x128x8
            Swish(),
            View(dim=(-1, 2 * 128 * 8)),
            nn.Linear(2 * 128 * 8, x_dim),
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
    enc = ConvEncoderY()
    enc.test()

    print("dec")
    dec = ConvDecoder()
    dec.test()

