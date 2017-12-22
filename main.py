#!python

import os
import sys
import time
import logging
import argparse
from random import random

import numpy as np
from scipy.stats import norm

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
from torchvision import datasets, transforms

sys.path.append('/home/jbaik/setup/pytorch/probtorch')
import probtorch
from functools import wraps

# command line options
parser = argparse.ArgumentParser(description='SS-VAE example')
parser.add_argument('--batch_size', default=128, type=int, help='batch size for training')
parser.add_argument('--epochs', default=200, type=int, help='number of training epochs')
parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--num_samples', default=8, type=int, help='number of samples (?)')
parser.add_argument('--label_fraction', default=0.1, type=float, help='fraction for the labeled data')
parser.add_argument('--no_cuda', dest='no_cuda', action='store_true', help='do not use cuda')
parser.add_argument('--data_path', default='./data', help='path to download/read data')
parser.add_argument('--model_path', default='./models', help='path to store trained models')
parser.add_argument('--image_path', default='./images', help='path to stored images for check')
parser.add_argument('--model_prefix', default='final', help='indicate model name prefix to store')
parser.add_argument('--load_model', dest='load_model', action='store_true', help='load a model instead of training')

# logging
log = logging.getLogger('deepspeech.pytorch')
log.setLevel(logging.DEBUG)
fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

chdr = logging.StreamHandler()
chdr.setLevel(logging.DEBUG)
chdr.setFormatter(fmt)
log.addHandler(chdr)

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
CUDA = torch.cuda.is_available()

class Encoder(nn.Module):
    def __init__(self, num_pixels=NUM_PIXELS,
                       num_hidden=NUM_HIDDEN,
                       num_digits=NUM_DIGITS,
                       num_style=NUM_STYLE):
        super(self.__class__, self).__init__()
        self.enc_hidden = nn.Sequential(
                            nn.Linear(num_pixels, num_hidden),
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
        q.normal(styles_mean,
                 styles_std,
                 name='styles')
        return q

class Decoder(nn.Module):
    def __init__(self, num_pixels=NUM_PIXELS,
                       num_hidden=NUM_HIDDEN,
                       num_digits=NUM_DIGITS,
                       num_style=NUM_STYLE):
        super(self.__class__, self).__init__()
        self.num_digits = num_digits
        self.digit_log_weights = Parameter(torch.zeros(num_digits))
        self.digit_temp = 0.66
        self.style_mean = Parameter(torch.zeros(num_style))
        self.style_log_std = Parameter(torch.zeros(num_style))
        self.dec_hidden = nn.Sequential(
                            nn.Linear(num_style + num_digits, num_hidden),
                            nn.ReLU())
        self.dec_image = nn.Sequential(
                           nn.Linear(num_hidden, num_pixels),
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
                                  torch.log(1 - x_hat + EPS) * (1-x)).sum(-1),
               images_mean, images, name='images')
        return p


class SSVAE(object):

    def __init__(self, args):
        self.enc = Encoder()
        self.dec = Decoder()

        if CUDA:
            self.enc.cuda()
            self.dec.cuda()

        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.num_samples = args.num_samples
        self.label_fraction = args.label_fraction

        parameters = list(self.enc.parameters()) + list(self.dec.parameters())
        self.optimizer =  torch.optim.Adam(parameters, lr=args.lr, betas=(0.9, 0.999))

    def elbo(self, q, p, alpha=0.1):
        if self.num_samples is None:
            return probtorch.objectives.montecarlo.elbo(q, p, sample_dim=None, batch_dim=0, alpha=alpha)
        else:
            return probtorch.objectives.montecarlo.elbo(q, p, sample_dim=0, batch_dim=1, alpha=alpha)

    def train_epoch(self, data, batch_size=None, label_fraction=None, label_mask={}):
        if batch_size is None:
            batch_size = self.batch_size
        if label_fraction is None:
            label_fraction = self.label_fraction

        epoch_elbo = 0.0
        self.enc.train()
        self.dec.train()

        N = 0
        for b, (images, labels) in enumerate(data):
            if images.size()[0] != batch_size:
                continue
            N += batch_size
            images = images.view(-1, NUM_PIXELS)
            labels_onehot = torch.zeros(batch_size, NUM_DIGITS)
            labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
            labels_onehot = torch.clamp(labels_onehot, EPS, 1-EPS)
            if CUDA:
                images = images.cuda()
                labels_onehot = labels_onehot.cuda()
            images = Variable(images)
            labels_onehot = Variable(labels_onehot)
            self.optimizer.zero_grad()
            if b not in label_mask:
                label_mask[b] = (random() < label_fraction)
            if label_mask[b]:
                q = self.enc(images, labels_onehot, num_samples=self.num_samples)
            else:
                q = self.enc(images, num_samples=self.num_samples)
            p = self.dec(images, q, num_samples=self.num_samples)
            loss = -self.elbo(q, p)
            loss.backward()
            self.optimizer.step()
            if CUDA:
                loss = loss.cpu()
            epoch_elbo -= loss.data.numpy()[0]
    return epoch_elbo / N, label_mask

    def test(self, data, batch_size=None, infer=True):
        if batch_size is None:
            batch_size = self.batch_size

        self.enc.eval()
        self.dec.eval()
        epoch_elbo = 0.0
        epoch_correct = 0
        N = 0
        with torch.no_grad():
            for b, (images, labels) in enumerate(data):
                if images.size()[0] != batch_size:
                    continue
                N += batch_size
                images = images.view(-1, NUM_PIXELS)
                if CUDA:
                    images = images.cuda()
                images = Variable(images)
                q = self.enc(images, num_samples=self.num_samples)
                p = self.dec(images, q, num_samples=self.num_samples)
                batch_elbo = self.elbo(q, p)
                if CUDA:
                    batch_elbo = batch_elbo.cpu()
                epoch_elbo += batch_elbo.data.numpy()[0]
                if infer:
                    log_p = p.log_joint(0, 1)
                    log_q = q.log_joint(0, 1)
                    log_w = log_p - log_q
                    w = torch.nn.functional.softmax(log_w, 0)
                    y_samples = q['digits'].value
                    y_expect = (w.unsqueeze(-1) * y_samples).sum(0)
                    _ , y_pred = y_expect.data.max(-1)
                    if CUDA:
                        y_pred = y_pred.cpu()
                    epoch_correct += (labels == y_pred).sum()
                else:
                    _, y_pred = q['digits'].value.data.max(-1)
                    if CUDA:
                        y_pred = y_pred.cpu()
                    epoch_correct += (labels == y_pred).sum()*1.0 / (self.num_samples or 1.0)
        return epoch_elbo / N, epoch_correct / N

    def save(self, model_path, model_prefix):
        try:
            os.makedirs(model_path, exist_ok=True)
        except OSError as e:
            raise
        prefix = os.path.join(model_path, model_prefix)
        log.info(f"saving the model to {prefix}-{{enc,dec}}.pth.tar")
        torch.save(self.enc.state_dict(), f"{prefix}-enc.pth.tar")
        torch.save(self.dec.state_dict(), f"{prefix}-dec.pth.tar")

    def load(self, model_path, model_prefix):
        if not os.path.isdir(model_path):
            raise IOError
        prefix = os.path.join(model_path, model_prefix)
        log.info(f"loading the model from {prefix}-{{enc,dec}}.pth.tar")
        self.enc.load_state_dict(torch.load(f"{prefix}-enc.pth.tar"))
        self.dec.load_state_dict(torch.load(f"{prefix}-dec.pth.tar"))


def main(args, model, data):
    train_data, test_data = data

    if not args.load_model:
        log.info("training begins")
        mask = {}
        for e in range(args.epochs):
            train_start = time.time()
            train_elbo, mask = model.train_epoch(train_data, label_mask=mask)
            train_time = time.time() - train_start

            test_start = time.time()
            test_elbo, test_accuracy = model.test(test_data)
            test_time = time.time() - test_start
            log.info(f"[Epoch {e:03d}] Train: ELBO {train_elbo:6.4e} ({train_time:04.1f}s) "
                     f"Test: ELBO {test_elbo:6.4e}, Accuracy {test_accuracy:5.3f} ({test_time:04.1f}s)")

        model.save(args.model_path, args.model_prefix)
        log.info("training done")
    else:
        model.load(args.model_path, args.model_prefix)
        log.info("the model is loaded successfully")

    elbo, accuracy = model.test(test_data, infer=False)
    log.info(f"[encoder] ELBO: {elbo:6.4e}, ACCURACY: {accuracy:5.3f}")

    elbo, accuracy = model.test(test_data, infer=True)
    log.info(f"[encoder+inference] ELBO: {elbo:6.4e}, ACCURACY: {accuracy:5.3f}")


def check(args, model, data):
    train_data, test_data = data
    image_path = args.image_path

    # display
    try:
        os.makedirs(image_path, exist_ok=True)
    except OSError as e:
        raise

    ys = []
    zs = []
    for (x, y) in test_data:
        if len(x) == args.batch_size:
            images = Variable(x).view(-1, NUM_PIXELS)
            if CUDA:
                q = model.enc(images.cuda())
                z = q['styles'].value.data.cpu().numpy()
            else:
                q = model.enc(images)
                z = q['styles'].value.data.numpy()
            zs.append(z)
            ys.append(y.numpy())
    ys = np.concatenate(ys,0)
    zs = np.concatenate(zs,0)

    # run TSNE when number of latent dims exceeds 2
    if NUM_STYLE > 2:
        log.info("doing T-SNE to check the latent space")
        #from MulticoreTSNE import MulticoreTSNE as TSNE
        #tsne = TSNE(n_jobs=40)
        from sklearn.manifold import TSNE
        tsne = TSNE()
        zs2 = tsne.fit_transform(zs)
        zs2_mean = zs2.mean(0)
        zs2_std = zs2.std(0)
    else:
        zs2 = zs

    # display a 2D plot of the digit classes in the latent space
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(6,6))
    ax = plt.gca()

    colors = []
    for k in range(10):
        m = (ys == k)
        p = ax.scatter(zs2[m, 0], zs2[m, 1], label='y=%d' % k, alpha=0.5, s=5)
        colors.append(p.get_facecolor())
    ax.legend()

    fig.tight_layout()
    figfile = os.path.join(image_path, '01_encodings.png')
    fig.savefig(figfile, dpi=300)
    log.info(f"the figure of latent encoding is stored to {figfile}")

    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()

    # display a 2D plot of the digit classes in the latent space
    fig = plt.figure(figsize=(10,4.25))

    for k in range(10):
        ax = plt.subplot(2,5,k+1)
        m = (ys == k)
        ax.scatter(zs2[m, 0], zs2[m, 1], alpha=0.5, s=5, c=colors[k])
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_title('y=%d' % k)

    fig.tight_layout()
    figfile = os.path.join(image_path, '02_classes.png')
    fig.savefig(figfile, dpi=300)
    log.info(f"the figure of encodings for each class is stored to {figfile}")

    x,_ = next(iter(train_data))
    x_var = Variable(x.view(-1, NUM_PIXELS))
    if CUDA:
        q = model.enc(x_var.cuda())
        p = model.dec(x_var.cuda(), q)
        x_mean = p['images'].value.view(args.batch_size, 28, 28).data.cpu().numpy()
    else:
        q = model.enc(x_var)
        p = model.dec(x_var, q)
        x_mean = p['images'].value.view(args.batch_size, 28, 28).data.numpy().squeeze()

    fig = plt.figure(figsize=(12,5.25))
    for k in range(5):
        ax = plt.subplot(2, 5, k+1)
        ax.imshow(x[k].squeeze())
        ax.set_title("original")
        plt.axis("off")
        ax = plt.subplot(2, 5, k+6)
        ax.imshow(x_mean[k].squeeze())
        ax.set_title("reconstructed")
        plt.axis("off")

    fig.tight_layout()
    figfile = os.path.join(image_path, '03_reconstructions.png')
    fig.savefig(figfile, dpi=300, facecolor=[0,0,0,0])
    log.info(f"the figure of original and reconstructed image samples is stored to {figfile}")

    # display a 2D manifold of the digits
    n = 7  # figure with 15x15 digits
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
    # to produce values of the latent variables z, since the prior of the latent space is Gaussian
    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))
    null_image = Variable(torch.Tensor(np.zeros((1, 784))))

    fig = plt.figure(figsize=(12, 30))
    for y in range(10):
        plt.subplot(5, 2, y+1)
        y_hot = np.zeros((1,10))
        y_hot[0,y] = 1
        y_hot = Variable(torch.FloatTensor(y_hot))
        my = (ys == y)
        for i, z0i in enumerate(grid_x):
            for j, z1j in enumerate(grid_y[-1::-1]):
                z = np.array([[z0i, z1j]])
                if NUM_STYLE > 2:
                    z = zs2_mean[None,:] + zs2_std[None,:] * z
                    n = ((zs2[my] - z)**2).sum(1).argmin()
                    z = zs[my][n][None,:]
                z = Variable(torch.FloatTensor(z))
                if CUDA:
                    p = model.dec(null_image.cuda(), {'styles': z.cuda(), 'digits': y_hot.cuda()})
                    images = p['images'].value.data.cpu().numpy()
                else:
                    p = model.dec(null_image, {'styles': z, 'digits': y_hot})
                    images = p['images'].value.data.numpy()
                digit = images.reshape(digit_size, digit_size)
                figure[j * digit_size: (j + 1) * digit_size,
                       i * digit_size: (i + 1) * digit_size] = digit
        plt.imshow(figure)
        plt.title('y=%d' % y)
        plt.axis('off')

    fig.tight_layout()
    figfile = os.path.join(image_path, '04_digits.png')
    fig.savefig(figfile, dpi=300)
    log.info(f"the figure of all digits variables is stored to {figfile}")


def prepare_data(data_path, batch_size):
    # prepare data
    try:
        os.makedirs(data_path, exist_ok=True)
    except OSError as e:
        raise

    train_dataset = datasets.MNIST(data_path, train=True, download=True, transform=transforms.ToTensor())
    test_dataset = datasets.MNIST(data_path, train=False, download=True, transform=transforms.ToTensor())

    train_data = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_data = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    log.info("data preparation complete")
    return train_data, test_data


if __name__ == "__main__":
    args = parser.parse_args()

    if args.no_cuda:
        CUDA = False

    # log file handler
    try:
        os.makedirs(args.model_path, exist_ok=True)
        fhdr = logging.FileHandler(os.path.join(args.model_path, "ss_vae.log"))
        fhdr.setLevel(logging.DEBUG)
        fhdr.setFormatter(fmt)
        log.addHandler(fhdr)
    except OSError as e:
        raise

    log.info(f"probtorch:{probtorch.__version__} torch:{torch.__version__} cuda:{CUDA}")

    # prepare data
    data = prepare_data(args.data_path, args.batch_size)
    # prepare model
    model = SSVAE(args)

    # train
    main(args, model, data)
    # check the result
    check(args, model, data)
