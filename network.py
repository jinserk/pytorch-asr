#!python
import math

import torch
import torch.nn as nn
from torch.nn import Parameter

from pyro.nn import ClippedSoftmax, ClippedSigmoid

NUM_PIXELS = 784
NUM_DIGITS = 10
NUM_HIDDEN = [256, 256]
NUM_STYLE = 7 * 7 * 2
EPS = 1e-9


class View(nn.Module):

    def __init__(self, dim):
        super(self.__class__, self).__init__()
        self.dim = dim

    def forward(self, x, *args):
        return x.view(*self.dim)


class Tuple(nn.Module):

    def __init__(self, nets):
        super(self.__class__, self).__init__()
        self.nets = nets

    def forward(self, *args, **kwargs):
        return (n.forward(*args, **kwargs) for n in self.nets)


class Swish(nn.Module):

    def forward(self, x):
        return x * nn.functional.sigmoid(x)


class MlpEncoderY(nn.Module):

    def __init__(self, x_dim=NUM_PIXELS, y_dim=NUM_DIGITS, h_dims=NUM_HIDDEN,
                 eps=EPS):
        super(self.__class__, self).__init__()
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

    def __init__(self, x_dim=NUM_PIXELS, y_dim=NUM_DIGITS, h_dims=NUM_HIDDEN, z_dim=NUM_STYLE,
                 eps=EPS):
        super(self.__class__, self).__init__()
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
            Tuple(nets=(
                nn.Linear(h_dims[-1], z_dim),   # for z mean
                nn.Linear(h_dims[-1], z_dim),   # for z log std
            )),
        ]
        self.hidden = nn.Sequential(*layers)

    def forward(self, xs, ys, *args, **kwargs):
        z_mean, z_log_std = self.hidden.forward(torch.cat([xs, ys], -1), *args, **kwargs)
        return z_mean, torch.exp(z_log_std)


class MlpDecoder(nn.Module):

    def __init__(self, x_dim=NUM_PIXELS, y_dim=NUM_DIGITS, h_dims=NUM_HIDDEN, z_dim=NUM_STYLE,
                 eps=EPS):
        super(self.__class__, self).__init__()
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

    def __init__(self, x_dim=NUM_PIXELS, y_dim=NUM_DIGITS, eps=EPS):
        super(self.__class__, self).__init__()
        # network
        layers = [
            View(dim=(-1, 1, 28, 28)),
            nn.Conv2d(1, 16, kernel_size=8, stride=2, padding=3),  # 1x28x28 -> 16x14x14
            nn.BatchNorm2d(16),
            Swish(),
            nn.Conv2d(16, 32, 8, 2, 3),  # 16x14x14 -> 32x7x7
            nn.BatchNorm2d(32),
            Swish(),
            View(dim=(-1, 32 * 7 * 7)),
            nn.Linear(32 * 7 * 7, y_dim),
            ClippedSoftmax(eps, dim=1),
        ]
        self.hidden = nn.Sequential(*layers)

    def forward(self, xs, *args, **kwargs):
        ys = self.hidden.forward(xs.view(-1, 1, 28, 28), *args, **kwargs)
        return ys


class ConvDecoder(nn.Module):

    def __init__(self, x_dim=NUM_PIXELS, y_dim=NUM_DIGITS, z_dim=NUM_STYLE,
                 eps=EPS):
        super(self.__class__, self).__init__()
        # network
        layers = [
            nn.Linear(y_dim + z_dim, 32 * 7 * 7),
            Swish(),
            View(dim=(-1, 32, 7, 7)),
            nn.ConvTranspose2d(32, 16, kernel_size=8, stride=2, padding=3),  # 32x7x7 -> 16x14x14
            Swish(),
            nn.ConvTranspose2d(16, 1, 8, 2, 3),  # 16x14x14 -> 1x28x28
            Swish(),
            View(dim=(-1, 28*28)),
            nn.Linear(x_dim, x_dim),
            ClippedSigmoid(eps)
        ]
        self.hidden = nn.Sequential(*layers)

    def forward(self, zs, ys, *args, **kwargs):
        xs = self.hidden.forward(torch.cat([zs, ys], -1), *args, **kwargs)
        return xs


#class ConvEncoder(nn.Module):
#
#    def __init__(self, num_pixels=NUM_PIXELS, num_hidden=NUM_HIDDEN,
#                 num_digits=NUM_DIGITS, num_style=NUM_STYLE):
#        super(self.__class__, self).__init__()
#
#        self.enc_hidden = nn.Sequential(
#            nn.Conv2d(1, 16, kernel_size=8, stride=2, padding=3),  # 1x28x28 -> 16x14x14
#            nn.BatchNorm2d(16),
#            Swish(),
#            nn.Conv2d(16, 32, 8, 2, 3),  # 16x14x14 -> 32x7x7
#            nn.BatchNorm2d(32),
#            Swish()
#        )
#
#        # W2 = (W1 - F + 2P) / S + 1
#        num_out = int(math.floor((14 - 8 + 2 * 3) / 2) + 1)
#        num_out *= 32
#
#        self.digit_log_weights = nn.Sequential(
#            nn.BatchNorm1d(num_out),
#            nn.Linear(num_out, num_digits)
#        )
#
#        self.digit_temp = 0.66
#        self.z_mean = nn.Linear(num_hidden + num_digits, num_style)
#        self.z_log_std = nn.Linear(num_hidden + num_digits, num_style)
#
#    def forward(self, x, labels=None, num_samples=None):
#        q = probtorch.Trace()
#        h = self.enc_hidden(x)
#        y = q.concrete(self.digit_log_weights(h), self.digit_temp, value=labels, name='y')
#        h2 = torch.cat((y, h), dim=1)
#        z_mean = self.z_mean(h2)
#        z_std = torch.exp(self.z_log_std(h2))
#        z = q.normal(z_mean, z_std, name='z')
#        return q


#class ConvDecoder(nn.Module):
#
#    def __init__(self, num_pixels=NUM_PIXELS, num_hidden=NUM_HIDDEN,
#                 num_digits=NUM_DIGITS, num_style=NUM_STYLE):
#        super(self.__class__, self).__init__()
#        self.num_digits = num_digits
#        self.digit_log_weights = Parameter(torch.zeros(num_digits))
#        self.digit_temp = 0.66
#        #self.z_mean = Parameter(torch.zeros(num_style))
#        #self.z_log_std = Parameter(torch.zeros(num_style))
#
#        self.dec_hidden = nn.Sequential(
#            nn.ConvTranspose2d(32, 16, kernel_size=8, stride=2, padding=3),  # 32x7x7 -> 16x14x14
#            Swish(),
#            nn.ConvTranspose2d(16, 1, 8, 2, 3),  # 16x14x14 -> 1x28x28
#            Swish()
#        )
#
#        self.dec_image = nn.Sequential(
#            nn.Linear(num_hidden, num_pixels),
#            nn.Sigmoid()
#        )
#
#    def forward(self, x, q=None, num_samples=None):
#        p = probtorch.Trace()
#        y = p.concrete(self.digit_log_weights, self.digit_temp, value=q['y'], name='y')
#        z = p.normal(0.0, 1.0, value=q['z'], name='z')
#        h = self.dec_hidden(torch.cat([y, z], -1))
#        x_mean = self.dec_image(h)
#        p.loss(binary_cross_entropy, x_mean, x, name='x')
#        return p


if __name__ == "__main__":
    pass
