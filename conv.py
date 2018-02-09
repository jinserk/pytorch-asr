#!python
import sys
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.autograd import Variable

from utils.logger import logger

from network import *


class ConvAM(nn.Module):
    """
    This class encapsulates the parameters (neural networks) and models & guides needed to train a
    semi-supervised variational auto-encoder on the MNIST image dataset

    :param use_cuda: use GPUs for faster training
    :param batch_size: batch size of calculation
    :param init_lr: initial learning rate to setup the optimizer
    :param continue_from: model file path to load the model states
    """
    def __init__(self, x_dim=NUM_PIXELS, y_dim=NUM_DIGITS, use_cuda=False,
                 batch_size=100, init_lr=0.001, continue_from=None, *args, **kwargs):
        super().__init__()

        # initialize the class with all arguments provided to the constructor
        self.x_dim = x_dim
        self.y_dim = y_dim

        self.use_cuda = use_cuda
        self.batch_size = batch_size
        self.init_lr = init_lr

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
        self.encoder = ConvEncoderY(x_dim=self.x_dim, y_dim=self.y_dim, softmax=False)

        # setup the optimizer
        parameters = self.encoder.parameters()
        self.optimizer = torch.optim.Adam(parameters, lr=self.init_lr, betas=(0.9, 0.999), eps=1e-8)
        self.loss = nn.CrossEntropyLoss()

    def classifier(self, xs):
        """
        classify an image (or a batch of images)

        :param xs: a batch of scaled vectors of pixels from an image
        :return: a batch of the corresponding class labels (as one-hots)
        """
        # use the trained model q(y|x) = categorical(alpha(x))
        # compute all class probabilities for the image(s)
        alpha = self.encoder.forward(xs)

        # get the index (digit) that corresponds to
        # the maximum predicted class probability
        res, ind = torch.topk(alpha, 1)

        # convert the digit(s) to one-hot tensor(s)
        ys = Variable(torch.zeros(alpha.size()))
        ys = ys.scatter_(1, ind, 1.0)
        return ys

    def train_epoch(self, epoch, data_loaders, train_data_size):
        # initialize variables to store loss values
        epoch_loss = 0.

        # setup the iterators for training data loaders
        train_iter = iter(data_loaders["train"])

        # count the number of supervised batches seen in this epoch
        for i in tqdm(range(train_data_size), desc="training  "):
            # extract the corresponding batch
            xs, ys = next(train_iter)
            res, ind = torch.topk(ys, 1)
            ys = ind.long().squeeze()
            xs, ys = Variable(xs), Variable(ys)
            self.optimizer.zero_grad()
            # run the inference for each loss (loss with size_avarage=True)
            y_hats = self.encoder(xs)
            loss = self.loss(y_hats, ys)
            epoch_loss += loss
            # compute gradient
            loss.backward()
            # optimizer step
            self.optimizer.step()
            if self.use_cuda:
                torch.cuda.synchronize()
            del loss, y_hats

        # compute average epoch loss i.e. loss per example
        avg_loss = map(lambda x: x / train_data_size, epoch_loss)
        return avg_loss

    def get_accuracy(self, data_loader, val_num, desc=None):
        """
        compute the accuracy over the supervised training set or the testing set
        """
        # use the appropriate data loader
        data_iter = iter(data_loader)

        predictions, actuals = [], []
        for i in tqdm(range(val_num), desc=desc):
            xs, ys = next(data_iter)
            xs, ys = Variable(xs), Variable(ys)
            # use classification function to compute all predictions for each batch
            with torch.no_grad():
                predictions.append(self.classifier(xs))
            actuals.append(ys)

        # compute the number of accurate predictions
        accurate_preds = 0
        for pred, act in zip(predictions, actuals):
            for i in range(pred.size(0)):
                v = torch.sum(pred[i] == act[i])
                accurate_preds += (v.data[0] == 10)

        # calculate the accuracy between 0 and 1
        accuracy = (accurate_preds * 1.0) / (len(predictions) * self.batch_size)
        return accuracy

    def save(self, file_path, **kwargs):
        Path(file_path).parent.mkdir(mode=0o755, parents=True, exist_ok=True)
        logger.info(f"saving the model to {file_path}")
        states = kwargs
        states["conv"] = self.state_dict()
        states["optimizer"] = self.optimizer.get_state()
        torch.save(states, file_path)

    def load(self, file_path):
        if isinstance(file_path, str):
            file_path = Path(file_path)
        if not file_path.exists():
            logger.error(f"no such file {file_path} exists")
            sys.exit(1)
        logger.info(f"loading the model from {file_path}")
        states = torch.load(file_path)

        self.__setup_networks()
        self.load_state_dict(states["conv"])
        self.optimizer.set_state(states["optimizer"])
        return states
