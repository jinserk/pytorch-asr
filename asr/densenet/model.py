#!python
import sys
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchnet as tnt

from ..utils.misc import onehot2int
from ..utils.logger import logger
from ..utils import params as p

from .network import *


class DenseNetModel(nn.Module):
    """
    This class encapsulates the parameters (neural networks) and models & guides
    needed to train a supervised DenseNet on the Aspire audio dataset

    :param use_cuda: use GPUs for faster training
    :param batch_size: batch size of calculation
    :param init_lr: initial learning rate to setup the optimizer
    :param continue_from: model file path to load the model states
    """
    def __init__(self, x_dim=p.NUM_PIXELS, y_dim=p.NUM_LABELS, use_cuda=False,
                 batch_size=100, init_lr=0.001, continue_from=None, *args, **kwargs):
        super().__init__()

        # initialize the class with all arguments provided to the constructor
        self.x_dim = x_dim
        self.y_dim = y_dim

        self.use_cuda = use_cuda
        self.batch_size = batch_size
        self.init_lr = init_lr
        self.epoch = 1

        self.meter_loss = tnt.meter.AverageValueMeter()
        self.meter_accuracy = tnt.meter.ClassErrorMeter(accuracy=True)
        self.meter_confusion = tnt.meter.ConfusionMeter(p.NUM_LABELS, normalized=True)

        if continue_from is None:
            # define and instantiate the neural networks representing
            # the paramters of various distributions in the model
            self.__setup_networks()
        else:
            self.load(continue_from)

        # using GPUs for faster training of the networks
        if self.use_cuda:
            self.cuda()

    def __setup_networks(self):
        # define the neural networks used later in the model and the guide.
        self.encoder = DenseNet(x_dim=self.x_dim, y_dim=self.y_dim)

        # setup the optimizer
        parameters = self.encoder.parameters()
        self.optimizer = torch.optim.Adam(parameters, lr=self.init_lr, betas=(0.9, 0.999), eps=1e-8)
        self.loss = nn.CrossEntropyLoss()

    def __reset_meters(self):
        self.meter_loss.reset()
        self.meter_accuracy.reset()
        self.meter_confusion.reset()

    def classifier(self, xs):
        """
        classify an image (or a batch of images)

        :param xs: a batch of scaled vectors of pixels from an image
        :return: a batch of the corresponding class labels (as one-hots)
        """
        # compute all class probabilities for the image(s)
        ys_hat = self.encoder(xs)
        # convert one-hot tensor(s) to the digit(s)
        return onehot2int(ys_hat)

    def train_epoch(self, data_loader):
        self.__reset_meters()
        # count the number of supervised batches seen in this epoch
        for i, (data) in tqdm(enumerate(data_loader), total=len(data_loader), desc="training"):
            # extract the corresponding batch
            xs, ys = data
            # run the inference for each loss (loss with size_avarage=True)
            self.optimizer.zero_grad()
            ys_hat = self.encoder(xs)
            ys_int = onehot2int(ys)
            loss = self.loss(ys_hat, ys_int)
            # compute gradient
            loss.backward()
            # update meters
            self.meter_loss.add(loss)
            self.meter_accuracy.add(ys_hat.data, ys_int)
            self.meter_confusion.add(ys_hat.data, ys_int)
            # optimize
            self.optimizer.step()
            if self.use_cuda:
                torch.cuda.synchronize()
            # free
            del loss, ys_hat

    def test(self, data_loader):
        self.__reset_meters()
        for i, (data) in tqdm(enumerate(data_loader), total=len(data_loader), desc="testing "):
            xs, ys = data
            # use classification function to compute all predictions for each batch
            with torch.no_grad():
                ys_hat = self.encoder(xs)
                ys_int = onehot2int(ys)
                loss = self.loss(xs, ys_int)
                # update meters
                self.meter_loss.add(loss)
                self.meter_accuracy.add(ys_hat.data, ys_int)
                self.meter_confusion.add(ys_hat.data, ys_int)
                if self.use_cuda:
                    torch.cuda.synchronize()
                del loss, ys_hat

    def save(self, file_path, **kwargs):
        Path(file_path).parent.mkdir(mode=0o755, parents=True, exist_ok=True)
        logger.info(f"saving the model to {file_path}")
        states = kwargs
        states["epoch"] = self.epoch
        states["model"] = self.state_dict()
        states["optimizer"] = self.optimizer.state_dict()
        torch.save(states, file_path)

    def load(self, file_path):
        if isinstance(file_path, str):
            file_path = Path(file_path)
        if not file_path.exists():
            logger.error(f"no such file {file_path} exists")
            sys.exit(1)
        logger.info(f"loading the model from {file_path}")
        if not self.use_cuda:
            #states = torch.load(file_path, map_location=lambda storage, loc: storage)
            states = torch.load(file_path, map_location="cpu")
        else:
            states = torch.load(file_path)
        self.epoch = states["epoch"]

        self.__setup_networks()
        try:
            self.load_state_dict(states["model"])
        except: # for backward compatibility
            self.load_state_dict(states["conv"])
        self.optimizer.load_state_dict(states["optimizer"])

