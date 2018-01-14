#!python

import time
from pathlib import Path

from logger import log
from network import Network, file_suffix


def prepare_data(data_dir, batch_size):
    import torch
    from torchvision import datasets, transforms

    Path(data_dir).mkdir(mode=0o755, parents=True, exist_ok=True)

    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transforms.ToTensor())
    test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transforms.ToTensor())

    train_data = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_data = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    log.info("data preparation complete")
    return train_data, test_data


def train(args, network):
    if args.continue_from != "":
        parameters = network.load(args.model_dir, args.continue_from)
        start_epoch = parameters['epoch']
        filename = Path(args.model_dir, args.continue_from)
        log.info(f"training is resumed from {filename}.{file_suffix}")
    else:
        start_epoch = 0
        network.batch_size = args.batch_size
        log.info("new training begins")

    train_data, test_data = prepare_data(args.data_dir, network.batch_size)

    mask = {}
    for epoch in range(start_epoch, args.num_epochs):
        train_start = time.time()
        train_elbo, mask = network.train_epoch(train_data, label_mask=mask)
        train_time = time.time() - train_start

        test_start = time.time()
        test_elbo, test_accuracy = network.test(test_data)
        test_time = time.time() - test_start

        log.info(f"[Epoch {epoch+1:03d}] Train: ELBO {train_elbo:6.4e} ({train_time:04.1f}s) "
                 f"Test: ELBO {test_elbo:6.4e}, Accuracy {test_accuracy:5.3f} ({test_time:04.1f}s)")

        network.save(args.model_dir, args.model_prefix, f"epoch_{epoch+1:03d}",
                     epoch=epoch+1, elbo=test_elbo, accuracy=test_accuracy)

    elbo, accuracy = network.test(test_data, infer=False)
    log.info(f"[encoder] ELBO: {elbo:6.4e}, ACCURACY: {accuracy:5.3f}")
    elbo, accuracy = network.test(test_data, infer=True)
    log.info(f"[encoder+inference] ELBO: {elbo:6.4e}, ACCURACY: {accuracy:5.3f}")

    network.save(args.model_dir, args.model_prefix, "final",
                 epoch=epoch+1, elbo=elbo, accuracy=accuracy)
    log.info("training done")


if __name__ == "__main__":
    import argparse
    import probtorch_env
    import probtorch
    import torch

    # command line options
    parser = argparse.ArgumentParser(description='SS-VAE train')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size for training')
    parser.add_argument('--num_epochs', default=200, type=int, help='number of training epochs')
    parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate')
    parser.add_argument('--num_samples', default=8, type=int, help='number of samples (?)')
    parser.add_argument('--label_fraction', default=0.1, type=float, help='fraction for the labeled data')
    parser.add_argument('--cuda', dest='cuda', default=False, action='store_true', help='use cuda')
    parser.add_argument('--data_dir', default='./data', help='dir to download/read data')
    parser.add_argument('--model_dir', default='./models', help='dir where to store trained models')
    parser.add_argument('--model_prefix', default='mnist', help='model file prefix to store')
    parser.add_argument('--continue_from', default='', help='model filename to resume the training from')
    args = parser.parse_args()

    log.info(f"probtorch:{probtorch.__version__} torch:{torch.__version__}")

    # prepare model
    net = Network(args)

    # train
    train(args, net)
