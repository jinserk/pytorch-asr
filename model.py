#!python

import math

import torch
import torch.nn as nn
from torch.nn import Parameter

import probtorch

from functools import wraps


# TODO: move this into probtorch.util
def expand_inputs(f):
    @wraps(f)
    def g(*args, num_samples=None, **kwargs):
        if not num_samples is None:
            new_args = []
            new_kwargs = {}
            for arg in args:
                if hasattr(arg, 'expand'):
                    new_args.append(arg.expand(num_samples, *arg.size()))
                else:
                    new_args.append(arg)
            for k in kwargs:
                arg = kwargs[k]
                if hasattr(arg, 'expand'):
                    new_args.append(arg.expand(num_samples, *arg.size()))
                else:
                    new_args.append(arg)
            return f(*new_args, num_samples=num_samples, **new_kwargs)
        else:
            return f(*args, num_samples=None, **kwargs)
    return g


# global parameters
NUM_PIXELS = 784
NUM_HIDDEN = 256
NUM_DIGITS = 10
NUM_STYLE = 50
EPS = 1e-9


class Swish(nn.Module):
    def forward(self, x):
        return x * nn.functional.sigmoid(x)


def binary_cross_entropy(x_mean, x):
    return -(torch.log(x_mean + EPS) * x + torch.log(1 - x_mean + EPS) * (1 - x)).sum(-1)


class LinearEncoder(nn.Module):

    def __init__(self, num_pixels=NUM_PIXELS, num_hidden=NUM_HIDDEN,
                 num_digits=NUM_DIGITS, num_style=NUM_STYLE):
        super(self.__class__, self).__init__()
        self.enc_hidden = nn.Sequential(nn.Linear(num_pixels, num_hidden),
                                        Swish())
        self.digit_log_weights = nn.Linear(num_hidden, num_digits)
        self.digit_temp = 0.66
        self.z_mean = nn.Linear(num_hidden + num_digits, num_style)
        self.z_log_std = nn.Linear(num_hidden + num_digits, num_style)

    @expand_inputs
    def forward(self, x, labels=None, num_samples=None):
        q = probtorch.Trace()
        h = self.enc_hidden(x)
        y = q.concrete(self.digit_log_weights(h), self.digit_temp, value=labels, name='y')
        h2 = torch.cat([y, h], -1)
        z_mean = self.z_mean(h2)
        z_std = torch.exp(self.z_log_std(h2))
        z = q.normal(z_mean, z_std, name='z')
        return q


class LinearDecoder(nn.Module):

    def __init__(self, num_pixels=NUM_PIXELS, num_hidden=NUM_HIDDEN,
                 num_digits=NUM_DIGITS, num_style=NUM_STYLE):
        super(self.__class__, self).__init__()
        self.num_digits = num_digits
        self.digit_log_weights = Parameter(torch.zeros(num_digits))
        self.digit_temp = 0.66
        #self.z_mean = Parameter(torch.zeros(num_style))
        #self.z_log_std = Parameter(torch.zeros(num_style))
        self.dec_hidden = nn.Sequential(nn.Linear(num_style + num_digits, num_hidden),
                                        Swish())
        self.dec_image = nn.Sequential(nn.Linear(num_hidden, num_pixels),
                                       nn.Sigmoid())

    def forward(self, x, q=None, num_samples=None):
        p = probtorch.Trace()
        y = p.concrete(self.digit_log_weights, self.digit_temp, value=q['y'], name='y')
        z = p.normal(0.0, 1.0, value=q['z'], name='z')
        h = self.dec_hidden(torch.cat([y, z], -1))
        x_mean = self.dec_image(h)
        p.loss(binary_cross_entropy, x_mean, x, name='x')
        return p


class ConvEncoder(nn.Module):

    def __init__(self, num_pixels=NUM_PIXELS, num_hidden=NUM_HIDDEN,
                 num_digits=NUM_DIGITS, num_style=NUM_STYLE):
        super(self.__class__, self).__init__()

        self.enc_hidden = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=8, stride=2, padding=3),  # 1x28x28 -> 16x14x14
            nn.BatchNorm2d(16),
            Swish(),
            nn.Conv2d(16, 32, 8, 2, 3),  # 16x14x14 -> 32x7x7
            nn.BatchNorm2d(32),
            Swish()
        )

        # W2 = (W1 - F + 2P) / S + 1
        num_out = int(math.floor((14 - 8 + 2 * 3) / 2) + 1)
        num_out *= 32

        self.digit_log_weights = nn.Sequential(
            nn.BatchNorm1d(num_out),
            nn.Linear(num_out, num_digits)
        )

        self.digit_temp = 0.66
        self.z_mean = nn.Linear(num_hidden + num_digits, num_style)
        self.z_log_std = nn.Linear(num_hidden + num_digits, num_style)

    @expand_inputs
    def forward(self, x, labels=None, num_samples=None):
        q = probtorch.Trace()
        h = self.enc_hidden(x)
        y = q.concrete(self.digit_log_weights(h), self.digit_temp, value=labels, name='y')
        h2 = torch.cat((y, h), dim=1)
        z_mean = self.z_mean(h2)
        z_std = torch.exp(self.z_log_std(h2))
        z = q.normal(z_mean, z_std, name='z')
        return q


class ConvDecoder(nn.Module):

    def __init__(self, num_pixels=NUM_PIXELS, num_hidden=NUM_HIDDEN,
                 num_digits=NUM_DIGITS, num_style=NUM_STYLE):
        super(self.__class__, self).__init__()
        self.num_digits = num_digits
        self.digit_log_weights = Parameter(torch.zeros(num_digits))
        self.digit_temp = 0.66
        #self.z_mean = Parameter(torch.zeros(num_style))
        #self.z_log_std = Parameter(torch.zeros(num_style))

        self.dec_hidden = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=8, stride=2, padding=3),  # 32x7x7 -> 16x14x14
            Swish(),
            nn.ConvTranspose2d(16, 1, 8, 2, 3),  # 16x14x14 -> 1x28x28
            Swish()
        )

        self.dec_image = nn.Sequential(
            nn.Linear(num_hidden, num_pixels),
            nn.Sigmoid()
        )

    def forward(self, x, q=None, num_samples=None):
        p = probtorch.Trace()
        y = p.concrete(self.digit_log_weights, self.digit_temp, value=q['y'], name='y')
        z = p.normal(0.0, 1.0, value=q['z'], name='z')
        h = self.dec_hidden(torch.cat([y, z], -1))
        x_mean = self.dec_image(h)
        p.loss(binary_cross_entropy, x_mean, x, name='x')
        return p


if __name__ == "__main__":
    enc = LinearEncoder()
    dec = LinearDecoder()
