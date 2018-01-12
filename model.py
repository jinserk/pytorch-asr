#!python

import torch
import torch.nn as nn
from torch.nn import Parameter

import probtorch_env
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


class Encoder(nn.Module):

    def __init__(self, num_pixels=NUM_PIXELS, num_hidden=NUM_HIDDEN,
                 num_digits=NUM_DIGITS, num_style=NUM_STYLE):
        super(self.__class__, self).__init__()
        self.enc_hidden = nn.Sequential(nn.Linear(num_pixels, num_hidden),
                                        nn.ReLU())
        self.digit_log_weights = nn.Linear(num_hidden, num_digits)
        self.digit_temp = 0.66
        self.style_mean = nn.Linear(num_hidden + num_digits, num_style)
        self.style_log_std = nn.Linear(num_hidden + num_digits, num_style)

    @expand_inputs
    def forward(self, images, labels=None, num_samples=None):
        q = probtorch.Trace()
        hiddens = self.enc_hidden(images)
        digits = q.concrete(self.digit_log_weights(hiddens),
                            self.digit_temp,
                            value=labels,
                            name='digits')
        hiddens2 = torch.cat([digits, hiddens], -1)
        styles_mean = self.style_mean(hiddens2)
        styles_std = torch.exp(self.style_log_std(hiddens2))
        q.normal(styles_mean, styles_std, name='styles')
        return q


class Decoder(nn.Module):

    def __init__(self, num_pixels=NUM_PIXELS, num_hidden=NUM_HIDDEN,
                 num_digits=NUM_DIGITS, num_style=NUM_STYLE):
        super(self.__class__, self).__init__()
        self.num_digits = num_digits
        self.digit_log_weights = Parameter(torch.zeros(num_digits))
        self.digit_temp = 0.66
        self.style_mean = Parameter(torch.zeros(num_style))
        self.style_log_std = Parameter(torch.zeros(num_style))
        self.dec_hidden = nn.Sequential(nn.Linear(num_style + num_digits, num_hidden),
                                        nn.ReLU())
        self.dec_image = nn.Sequential(nn.Linear(num_hidden, num_pixels),
                                       nn.Sigmoid())

    def forward(self, images, q=None, num_samples=None):
        p = probtorch.Trace()
        digits = p.concrete(self.digit_log_weights, self.digit_temp,
                            value=q['digits'],
                            name='digits')
        styles = p.normal(0.0, 1.0,
                          value=q['styles'],
                          name='styles')
        hiddens = self.dec_hidden(torch.cat([digits, styles], -1))
        images_mean = self.dec_image(hiddens)

        p.loss(lambda x_hat, x: -(torch.log(x_hat + EPS) * x +
                                  torch.log(1 - x_hat + EPS) * (1 - x)).sum(-1),
               images_mean, images, name='images')
        return p


if __name__ == "__main__":
    enc = Encoder()
    dec = Decoder()
