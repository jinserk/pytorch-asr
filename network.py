#!python
import math

import torch
import torch.nn as nn
from torch.nn import Parameter

from pyro.nn import ClippedSoftmax, ClippedSigmoid

NUM_PIXELS = 784
NUM_DIGITS = 10
NUM_HIDDEN = [256, ]
NUM_STYLE = 7 * 7 * 2
EPS = 1e-9


class Swish(nn.Module):
    def forward(self, x):
        return x * nn.functional.sigmoid(x)


class MlpEncoderY(nn.Module):
    def __init__(self, x_dim=NUM_PIXELS, y_dim=NUM_DIGITS, h_dims=NUM_HIDDEN,
                 eps=EPS):
        super(self.__class__, self).__init__()
        # network
        in_layers = [ nn.Linear(x_dim, h_dims[0]), Swish() ]
        hid_layers = []
        for l in range(1, len(h_dims) - 1):
            hid_layers.extend([ nn.Linear(h_dims[l], h_dims[l + 1]), Swish() ])
        out_layers = [ nn.Linear(h_dims[-1], y_dim), ClippedSoftmax(eps, dim=1) ]
        # parallelize
        layers = in_layers + hid_layers + out_layers
        self.hidden = nn.Sequential(*layers)

    def forward(self, *args, **kwargs):
        return self.hidden.forward(*args, **kwargs)


class MlpEncoderZ(nn.Module):
    def __init__(self, x_dim=NUM_PIXELS, y_dim=NUM_DIGITS, h_dims=NUM_HIDDEN, z_dim=NUM_STYLE,
                 eps=EPS):
        super(self.__class__, self).__init__()
        # network
        in_layers = [ nn.Linear(x_dim + y_dim, h_dims[0]), Swish() ]
        hid_layers = []
        for l in range(1, len(h_dims) - 1):
            hid_layers.append([ nn.Linear(h_dims[l], h_dims[l + 1]), Swish() ])
        # parallelize
        layers = in_layers + hid_layers
        self.hidden = nn.Sequential(*layers)
        # output: z_mean and z_std
        self.z_mean_hidden = nn.Linear(h_dims[-1], z_dim)
        self.z_log_std_hidden = nn.Linear(h_dims[-1], z_dim)

    def forward(self, xs, ys, *args, **kwargs):
        h = self.hidden.forward(torch.cat([xs, ys], -1), *args, **kwargs)
        z_mean = self.z_mean_hidden.forward(h, *args, **kwargs)
        z_std = torch.exp(self.z_log_std_hidden.forward(h, *args, **kwargs))
        return z_mean, z_std


class MlpDecoder(nn.Module):

    def __init__(self, x_dim=NUM_PIXELS, y_dim=NUM_DIGITS, h_dims=NUM_HIDDEN, z_dim=NUM_STYLE,
                 eps=EPS):
        super(self.__class__, self).__init__()
        # network
        in_layers = [ nn.Linear(z_dim + y_dim, h_dims[0]), Swish() ]
        hid_layers = []
        for l in range(1, len(h_dims) - 1):
            hid_layers.append([ nn.Linear(h_dims[l], h_dims[l + 1]), Swish() ])
        out_layers = [ nn.Linear(h_dims[-1], x_dim), ClippedSigmoid(eps) ]
        # parallelize
        layers = in_layers + hid_layers + out_layers
        self.hidden = nn.Sequential(*layers)

    def forward(self, zs, ys, *args, **kwargs):
        return self.hidden.forward(torch.cat([zs, ys], -1), *args, **kwargs)


class ConvEncoderY(nn.Module):
    def __init__(self, x_dim=NUM_PIXELS, y_dim=NUM_DIGITS,
                 eps=EPS):
        super(self.__class__, self).__init__()
        # network
        layers = [
            nn.Conv2d(1, 16, kernel_size=8, stride=2, padding=3),  # 1x28x28 -> 16x14x14
            nn.BatchNorm2d(16),
            Swish(),
            nn.Conv2d(16, 32, 8, 2, 3),  # 16x14x14 -> 32x7x7
            nn.BatchNorm2d(32),
            Swish(),
        ]
        self.hidden1 = nn.Sequential(*layers)
        layers = [
            nn.Linear(32 * 7 * 7, y_dim),
            ClippedSoftmax(eps, dim=1),
        ]
        self.hidden2 = nn.Sequential(*layers)

    def forward(self, xs, *args, **kwargs):
        h1 = self.hidden1.forward(xs.view(-1, 1, 28, 28), *args, **kwargs)
        ys = self.hidden2.forward(h1.view(-1, 32 * 7 * 7), *args, **kwargs)
        return ys


class ConvDecoder(nn.Module):

    def __init__(self, x_dim=NUM_PIXELS, y_dim=NUM_DIGITS, z_dim=NUM_STYLE,
                 eps=EPS):
        super(self.__class__, self).__init__()
        # network
        layers = [
            nn.Linear(y_dim + z_dim, 32 * 7 * 7),
            Swish(),
        ]
        self.hidden1 = nn.Sequential(*layers)
        layers = [
            nn.ConvTranspose2d(32, 16, kernel_size=8, stride=2, padding=3),  # 32x7x7 -> 16x14x14
            Swish(),
            nn.ConvTranspose2d(16, 1, 8, 2, 3),  # 16x14x14 -> 1x28x28
            Swish(),
        ]
        self.hidden2 = nn.Sequential(*layers)
        layers = [
            nn.Linear(x_dim, x_dim),
            ClippedSigmoid(eps)
        ]
        self.hidden3 = nn.Sequential(*layers)

    def forward(self, zs, ys, *args, **kwargs):
        h1 = self.hidden1.forward(torch.cat([zs, ys], -1), *args, **kwargs)
        h2 = self.hidden2.forward(h1.view(-1, 32, 7, 7), *args, **kwargs)
        xs = self.hidden3.forward(h2.view(-1, 28 * 28), *args, **kwargs)
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
