#!python

import os
from pathlib import Path

import numpy as np
from scipy.stats import norm

import torch
from torch.autograd import Variable

from logger import log
from model import NUM_PIXELS, NUM_STYLE


def save_figure(fig, image_dir, image_file):
    fig.tight_layout()
    figfile = Path(image_dir, '01_encodings.png')
    fig.savefig(str(figfile), dpi=300)
    return figfile


def check(args, network):
    # prepare data
    batch_size = network.batch_size
    train_data, test_data = prepare_data(args.data_dir, batch_size)

    ys = []
    zs = []
    for (x, y) in test_data:
        if len(x) == batch_size:
            images = Variable(x).view(-1, NUM_PIXELS)
            if network.cuda:
                q = network.enc(images.cuda())
                z = q['styles'].value.data.cpu().numpy()
            else:
                q = network.enc(images)
                z = q['styles'].value.data.numpy()
            zs.append(z)
            ys.append(y.numpy())
    ys = np.concatenate(ys, 0)
    zs = np.concatenate(zs, 0)

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
    fig = plt.figure(figsize=(6, 6))
    ax = plt.gca()

    colors = []
    for k in range(10):
        m = (ys == k)
        p = ax.scatter(zs2[m, 0], zs2[m, 1], label='y=%d' % k, alpha=0.5, s=5)
        colors.append(p.get_facecolor())
    ax.legend()

    figfile = save_figure(fig, args.image_dir, '01_encodings.png')
    log.info(f"the figure of latent encoding is stored to {figfile}")

    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()

    # display a 2D plot of the digit classes in the latent space
    fig = plt.figure(figsize=(10, 4.25))

    for k in range(10):
        ax = plt.subplot(2, 5, k + 1)
        m = (ys == k)
        ax.scatter(zs2[m, 0], zs2[m, 1], alpha=0.5, s=5, c=colors[k])
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_title('y=%d' % k)

    figfile = save_figure(fig, args.image_dir, '02_classes.png')
    log.info(f"the figure of encodings for each class is stored to {figfile}")

    x, _ = next(iter(train_data))
    x_var = Variable(x.view(-1, NUM_PIXELS))
    if network.cuda:
        q = network.enc(x_var.cuda())
        p = network.dec(x_var.cuda(), q)
        x_mean = p['images'].value.view(batch_size, 28, 28).data.cpu().numpy()
    else:
        q = network.enc(x_var)
        p = network.dec(x_var, q)
        x_mean = p['images'].value.view(batch_size, 28, 28).data.numpy().squeeze()

    fig = plt.figure(figsize=(12, 5.25))
    for k in range(5):
        ax = plt.subplot(2, 5, k + 1)
        ax.imshow(x[k].squeeze())
        ax.set_title("original")
        plt.axis("off")
        ax = plt.subplot(2, 5, k + 6)
        ax.imshow(x_mean[k].squeeze())
        ax.set_title("reconstructed")
        plt.axis("off")

    figfile = save_figure(fig, args.image_dir, '03_reconstructions.png')
    #fig.savefig(figfile, dpi=300, facecolor=[0, 0, 0, 0])
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
        plt.subplot(5, 2, y + 1)
        y_hot = np.zeros((1, 10))
        y_hot[0, y] = 1
        y_hot = Variable(torch.FloatTensor(y_hot))
        my = (ys == y)
        for i, z0i in enumerate(grid_x):
            for j, z1j in enumerate(grid_y[-1::-1]):
                z = np.array([[z0i, z1j]])
                if NUM_STYLE > 2:
                    z = zs2_mean[None, :] + zs2_std[None, :] * z
                    n = ((zs2[my] - z) ** 2).sum(1).argmin()
                    z = zs[my][n][None, :]
                z = Variable(torch.FloatTensor(z))
                if network.cuda:
                    p = network.dec(null_image.cuda(), {'styles': z.cuda(), 'digits': y_hot.cuda()})
                    images = p['images'].value.data.cpu().numpy()
                else:
                    p = network.dec(null_image, {'styles': z, 'digits': y_hot})
                    images = p['images'].value.data.numpy()
                digit = images.reshape(digit_size, digit_size)
                figure[j * digit_size: (j + 1) * digit_size,
                       i * digit_size: (i + 1) * digit_size] = digit
        plt.imshow(figure)
        plt.title('y=%d' % y)
        plt.axis('off')

    figfile = save_figure(fig, args.image_dir, '04_digits.png')
    log.info(f"the figure of all digits variables is stored to {figfile}")


if __name__ == "__main__":
    import argparse
    import probtorch_env
    import probtorch
    from network import Network
    from train import prepare_data

    # command line options
    parser = argparse.ArgumentParser(description='SS-VAE check')
    parser.add_argument('--cuda', dest='cuda', default=False, action='store_true', help='use cuda')
    parser.add_argument('--data_dir', default='./data', help='dir to download/read data')
    parser.add_argument('--image_dir', default='./images', help='dir where to store images for check')
    parser.add_argument('--model_dir', default='./models', help='dir where to read stored model file from')
    parser.add_argument('--model_prefix', default='mnist_final', help='indicate model file prefix to load')
    args = parser.parse_args()

    log.info(f"probtorch:{probtorch.__version__} torch:{torch.__version__}")

    # output path
    Path(args.image_dir).mkdir(mode=0o755, parents=True, exist_ok=True)

    # load model
    net = Network(args)

    # check the result
    check(args, net)
