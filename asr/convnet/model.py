#!python
import sys
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.utils as tvu
import torchnet as tnt

from ..utils.misc import onehot2int, get_model_file_path
from ..utils.logger import logger
from ..utils import params as p

from .network import *


class ConvNetModel(nn.Module):
    """
    This class encapsulates the parameters (neural networks) and models & guides
    needed to train a supervised ConvNet model on the aspire audio dataset

    :param use_cuda: use GPUs for faster training
    :param batch_size: batch size of calculation
    :param init_lr: initial learning rate to setup the optimizer
    :param continue_from: model file path to load the model states
    """
    def __init__(self, x_dim=p.NUM_PIXELS, y_dim=p.NUM_LABELS,
                 batch_size=8, init_lr=1e-4, max_norm=400, use_cuda=False,
                 log_dir='logs', model_prefix='conv_aspire', checkpoint=False, num_ckpt=10000,
                 continue_from=None, *args, **kwargs):
        super().__init__()

        # initialize the class with all arguments provided to the constructor
        self.x_dim = x_dim
        self.y_dim = y_dim

        self.batch_size = batch_size
        self.init_lr = init_lr
        self.max_norm = max_norm
        self.use_cuda = use_cuda

        self.log_dir = log_dir
        self.model_prefix = model_prefix
        self.ckeckpoint = checkpoint
        self.num_ckpt = num_ckpt

        self.epoch = 1

        if continue_from is None:
            # define and instantiate the neural networks representing
            # the paramters of various distributions in the model
            self.__setup_networks()
        else:
            self.load(continue_from)

    def __setup_networks(self):
        # setup networks
        self.encoder = ConvEncoderY(x_dim=self.x_dim, y_dim=self.y_dim)
        if self.use_cuda:
            self.encoder.cuda()
        # setup loss
        self.loss = nn.CrossEntropyLoss()
        # setup optimizer
        parameters = self.encoder.parameters()
        self.optimizer = torch.optim.Adam(parameters, lr=self.init_lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0005, l2_reg=False)
        self.lr_scheduler = None

    def __get_model_name(self, desc):
        return str(get_model_file_path(self.log_dir, self.model_prefix, desc))

    def __remove_ckpt_files(self, epoch):
        for ckpt in Path(self.log_dir).rglob(f"*_epoch_{epoch:03d}_ckpt_*"):
            ckpt.unlink()

    def train_epoch(self, data_loader):
        self.encoder.train()
        meter_loss = tnt.meter.MovingAverageValueMeter(self.num_ckpt // 10)
        #meter_accuracy = tnt.meter.ClassErrorMeter(accuracy=True)
        #meter_confusion = tnt.meter.ConfusionMeter(p.NUM_CTC_LABELS, normalized=True)
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            logger.info(f"current lr = {self.lr_scheduler.get_lr()}")
        # count the number of supervised batches seen in this epoch
        t = tqdm(enumerate(data_loader), total=len(data_loader), desc="training ")
        for i, (data) in t:
            xs, ys = data
            try:
                if self.use_cuda:
                    xs, ys = xs.cuda(), ys.cuda()
                ys_hat = self.encoder(xs, softmax=False)
                loss = self.loss(ys_hat, ys)
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.encoder.parameters(), self.max_norm)
                self.optimizer.step()
            except Exception as e:
                print(e)
                print(filenames, frame_lens, label_lens)
            meter_loss.add(loss.item())
            t.set_description(f"training (loss: {meter_loss.value()[0]:.3f})")
            t.refresh()
            #self.meter_accuracy.add(ys_int, ys)
            #self.meter_confusion.add(ys_int, ys)
            if 0 < i < len(data_loader) and i % self.num_ckpt == 0:
                if self.checkpoint:
                    logger.info(f"training loss at epoch_{self.epoch:03d}_ckpt_{i:07d}: "
                                f"{meter_loss.value()[0]:5.3f}")
                    self.save(self.__get_model_name(f"epoch_{self.epoch:03d}_ckpt_{i:07d}"))
            del xs, ys, ys_hat, loss
            #input("press key to continue")
        self.epoch += 1
        logger.info(f"epoch {self.epoch:03d}: "
                    f"training loss {meter_loss.value()[0]:5.3f} ")
                    #f"training accuracy {meter_accuracy.value()[0]:6.3f}")
        self.save(self.__get_model_name(f"epoch_{self.epoch:03d}"))
        self.__remove_ckpt_files(self.epoch-1)

    def test(self, data_loader, desc=None):
        self.encoder.eval()
        meter_loss = tnt.meter.AverageValueMeter()
        #meter_accuracy = tnt.meter.ClassErrorMeter(accuracy=True)
        #meter_confusion = tnt.meter.ConfusionMeter(p.NUM_CTC_LABELS, normalized=True)
        for i, (data) in tqdm(enumerate(data_loader), total=len(data_loader), desc=desc):
            xs, ys = data
            if self.use_cuda:
                xs, ys = xs.cuda(), ys.cuda()
            ys_hat = self.encoder(xs, softmax=True)
            loss = self.loss(ys_hat_cat, ys)
            meter_loss.add(loss.item())
            #meter_accuracy.add(ys_hat.data, ys_int)
            #meter_confusion.add(ys_hat.data, ys_int)
            del xs, ys, loss, ys_hat
        logger.info(f"epoch {self.epoch:03d}: "
                    f"validating loss {meter_loss.value()[0]:5.3f} ")
                    #f"validating accuracy {meter_accuracy.value()[0]:6.3f}")

    def predict(self, xs):
        self.encoder.eval()
        if self.use_cuda:
            xs = xs.cuda()
        ys_hat = self.encoder(xs, softmax=True)
        return ys_hat

    def save(self, file_path, **kwargs):
        Path(file_path).parent.mkdir(mode=0o755, parents=True, exist_ok=True)
        logger.info(f"saving the model to {file_path}")
        states = kwargs
        states["epoch"] = self.epoch
        states["model"] = self.encoder.state_dict()
        states["optimizer"] = self.optimizer.state_dict()
        states["lr_scheduler"] = self.lr_scheduler.state_dict()
        torch.save(states, file_path)

    def load(self, file_path):
        if isinstance(file_path, str):
            file_path = Path(file_path)
        if not file_path.exists():
            logger.error(f"no such file {file_path} exists")
            sys.exit(1)
        logger.info(f"loading the model from {file_path}")
        if not self.use_cuda:
            states = torch.load(file_path, map_location='cpu')
        else:
            states = torch.load(file_path)
        self.epoch = states["epoch"]

        self.__setup_networks()
        self.load_state_dict(states["model"])
        self.optimizer.load_state_dict(states["optimizer"])
        self.lr_scheduler.load_state_dict(states["lr_scheduler"])
        if self.use_cuda:
            self.encoder.cuda()

