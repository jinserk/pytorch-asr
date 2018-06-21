#!python
import sys
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from warpctc_pytorch import CTCLoss
import torchnet as tnt

from ..utils.misc import onehot2int
from ..utils.logger import logger
from ..utils import params as p
from ..utils.audio import FrameSplitter

from .network import *


class ResNetCTCModel:
    """
    This class encapsulates the parameters (neural networks) and models & guides
    needed to train a supervised ResNet on the Aspire audio dataset

    :param use_cuda: use GPUs for faster training
    :param batch_size: batch size of calculation
    :param init_lr: initial learning rate to setup the optimizer
    :param continue_from: model file path to load the model states
    """
    def __init__(self, x_dim=p.NUM_PIXELS, y_dim=p.NUM_LABELS, device=torch.device("cpu"),
                 batch_size=100, init_lr=0.001, continue_from=None, *args, **kwargs):
        super().__init__()

        # initialize the class with all arguments provided to the constructor
        self.x_dim = x_dim
        self.y_dim = y_dim

        self.device = device
        self.batch_size = batch_size
        self.init_lr = init_lr
        self.epoch = 1

        self.meter_loss = tnt.meter.AverageValueMeter()
        #self.meter_accuracy = tnt.meter.ClassErrorMeter(accuracy=True)
        #self.meter_confusion = tnt.meter.ConfusionMeter(p.NUM_LABELS, normalized=True)

        if continue_from is None:
            self.__setup_networks()
        else:
            self.load(continue_from)

    def __setup_networks(self):
        self.encoder = resnet152(num_classes=self.y_dim)
        self.encoder.cuda()

        parameters = self.encoder.parameters()
        self.optimizer = torch.optim.Adam(parameters, lr=self.init_lr, betas=(0.9, 0.999), eps=1e-8)
        self.loss = CTCLoss()

    def __reset_meters(self):
        self.meter_loss.reset()
        #self.meter_accuracy.reset()
        #self.meter_confusion.reset()

    def train_epoch(self, data_loader):
        self.encoder.train()
        self.__reset_meters()
        # count the number of supervised batches seen in this epoch
        for i, (data) in tqdm(enumerate(data_loader), total=len(data_loader), desc="training  "):
            xs, ys, frame_lens, label_lens, filenames = data
            xs = xs.cuda()
            ys_hat = self.encoder.test(xs)
            #ys_hat = self.encoder(xs)
            ys_hat = ys_hat.transpose(0, 1).transpose(0, 2).contiguous()  # TxNxH
            #ys_hat = ys_hat.to(torch.device('cpu'))
            #print(ys_hat.shape, frame_lens, ys.shape, label_lens)
            try:
                loss = self.loss(ys_hat, ys, frame_lens, label_lens)

                loss = loss / xs.size(0)  # average the loss by minibatch
                loss_sum = loss.data.sum()
                inf = float("inf")
                if loss_sum == inf or loss_sum == -inf:
                    logger.warning("received an inf loss, setting loss value to 0")
                    loss_value = 0
                else:
                    loss_value = loss.data[0]

                self.optimizer.zero_grad()
                loss.backward()
            except Exception as e:
                print(e)
                print(filenames, frame_lens, label_lens)
                #sys.exit(1)
            #ys_hat = ys_hat.transpose(0, 1).cpu()
            #ys_int = onehot2int(ys_hat).squeeze()
            self.meter_loss.add(loss_value)
            #self.meter_accuracy.add(ys_int, ys)
            #self.meter_confusion.add(ys_int, ys)
            self.optimizer.step()
            del xs, ys, ys_hat, loss

    def test(self, data_loader, desc=None):
        self.encoder.eval()
        self.__reset_meters()
        with torch.no_grad():
            for i, (data) in tqdm(enumerate(data_loader), total=len(data_loader), desc=desc):
                xs, ys, frame_lens, label_lens, filenames = data
                xs = xs.cuda()
                ys_hat = self.encoder(xs)
                ys_hat = ys_hat.transpose(0, 1).transpose(0, 2).contiguous()  # TxNxH
                #ys_int = onehot2int(ys)
                loss = self.loss(ys_hat, ys, frame_lens, label_lens)

                loss = loss / xs.size(0)  # average the loss by minibatch
                loss_sum = loss.data.sum()
                inf = float("inf")
                if loss_sum == inf or loss_sum == -inf:
                    logger.warning("received an inf loss, setting loss value to 0")
                    loss_value = 0
                else:
                    loss_value = loss.data[0]

                self.meter_loss.add(loss_value)
                #self.meter_accuracy.add(ys_hat.data, ys_int)
                #self.meter_confusion.add(ys_hat.data, ys_int)
                del loss, ys_hat

    def wer(self, s1, s2):
        import Levenshtein as Lev
        # build mapping of words to integers
        b = set(s1.split() + s2.split())
        word2char = dict(zip(b, range(len(b))))
        # map the words to a char array (Levenshtein packages only accepts strings)
        w1 = [chr(word2char[w]) for w in s1.split()]
        w2 = [chr(word2char[w]) for w in s2.split()]
        return Lev.distance(''.join(w1), ''.join(w2))

    def cer(self, s1, s2, is_char=False):
        import Levenshtein as Lev
        if is_char:
            s1, s2, = s1.replace(' ', ''), s2.replace(' ', '')
            return Lev.distance(s1, s2)
        else:
            c1 = [chr(c) for c in s1]
            c2 = [chr(c) for c in s2]
            return Lev.distance(''.join(c1), ''.join(c2))

    def save(self, file_path, **kwargs):
        Path(file_path).parent.mkdir(mode=0o755, parents=True, exist_ok=True)
        logger.info(f"saving the model to {file_path}")
        states = kwargs
        states["epoch"] = self.epoch
        states["model"] = self.encoder.state_dict()
        states["optimizer"] = self.optimizer.state_dict()
        torch.save(states, file_path)

    def load(self, file_path):
        if isinstance(file_path, str):
            file_path = Path(file_path)
        if not file_path.exists():
            logger.error(f"no such file {file_path} exists")
            sys.exit(1)
        logger.info(f"loading the model from {file_path}")
        states = torch.load(file_path)
        self.epoch = states["epoch"]

        self.__setup_networks()
        try:
            self.encoder.load_state_dict(states["model"])
        except:
            self.encoder.load_state_dict(states["conv"])
        self.optimizer.load_state_dict(states["optimizer"])
        self.encoder.to(self.device)

