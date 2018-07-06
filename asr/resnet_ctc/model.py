#!python
import sys
import pdb
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.utils as tvu
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
    def __init__(self, x_dim=p.NUM_PIXELS, y_dim=p.NUM_CTC_LABELS, use_cuda=False, viz=None, tbd=None,
                 batch_size=100, init_lr=0.001, max_norm=400, continue_from=None, *args, **kwargs):
        super().__init__()

        # initialize the class with all arguments provided to the constructor
        self.x_dim = x_dim
        self.y_dim = y_dim

        self.use_cuda = use_cuda
        self.batch_size = batch_size
        self.init_lr = init_lr
        self.max_norm = max_norm
        self.epoch = 0

        self.meter_loss = tnt.meter.AverageValueMeter()
        #self.meter_accuracy = tnt.meter.ClassErrorMeter(accuracy=True)
        #self.meter_confusion = tnt.meter.ConfusionMeter(p.NUM_CTC_LABELS, normalized=True)

        self.viz = viz
        if self.viz is not None:
            self.viz.add_plot(title='loss', xlabel='epoch')

        self.tbd = tbd

        if continue_from is None:
            self.__setup_networks()
        else:
            self.load(continue_from)

    def __setup_networks(self):
        self.encoder = resnet101(num_classes=self.y_dim)
        if self.use_cuda:
            self.encoder.cuda()

        parameters = self.encoder.parameters()
        self.optimizer = torch.optim.Adam(parameters, lr=self.init_lr, betas=(0.9, 0.999), eps=1e-8)
        self.loss = CTCLoss(blank=0, size_average=True)

    def __reset_meters(self):
        self.meter_loss.reset()
        #self.meter_accuracy.reset()
        #self.meter_confusion.reset()

    def train_epoch(self, data_loader, prefix=None):
        self.encoder.train()
        self.__reset_meters()
        # count the number of supervised batches seen in this epoch
        t = tqdm(enumerate(data_loader), total=len(data_loader), desc="training ")
        for i, (data) in t:
            xs, ys, frame_lens, label_lens, filenames = data
            if self.use_cuda:
                xs = xs.cuda()
            #ys_hat = self.encoder.test(xs)
            ys_hat = self.encoder(xs)
            #print(onehot3int(ys_hat[0]).squeeze())
            frame_lens.div_(2)
            #torch.set_printoptions(threshold=5000000)
            #print(ys_hat.shape, frame_lens, ys.shape, label_lens)
            #print(onehot2int(ys_hat).squeeze(), ys)
            try:
                loss = self.loss(ys_hat.transpose(0, 1).contiguous(), ys, frame_lens, label_lens)
                #print(loss)

                #loss = loss / xs.size(0)  # average the loss by minibatch - size_average=True in CTC_Loss()
                loss_sum = loss.data.sum()
                inf = float("inf")
                if loss_sum == inf or loss_sum == -inf:
                    #torch.set_printoptions(threshold=5000000)
                    #print(filenames, ys_hat, frame_lens, label_lens)
                    logger.warning("received an inf loss, setting loss value to 0")
                    loss_value = 0
                else:
                    loss_value = loss.data[0]

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.encoder.parameters(), self.max_norm)
                self.optimizer.step()
            except Exception as e:
                print(e)
                print(filenames, frame_lens, label_lens)

            #ys_int = onehot2int(ys_hat).squeeze()
            self.meter_loss.add(loss_value)
            t.set_description(f"training (loss: {self.meter_loss.value()[0]:.3f})")
            t.refresh()
            #self.meter_accuracy.add(ys_int, ys)
            #self.meter_confusion.add(ys_int, ys)

            if 0 < i < len(data_loader) and i % 10000 == 0:
                if self.viz is not None:
                    self.viz.add_point(
                        title = 'loss',
                        x = self.epoch+i/len(data_loader),
                        y = self.meter_loss.value()[0]
                    )

                if self.tbd is not None:
                    x = self.epoch * len(data_loader) + i
                    self.tbd.add_graph(self.encoder, xs)
                    xs_img = tvu.make_grid(xs[0, 0], normalize=True, scale_each=True)
                    self.tbd.add_image('xs', x, xs_img)
                    ys_hat_img = tvu.make_grid(ys_hat[0].transpose(0, 1), normalize=True, scale_each=True)
                    self.tbd.add_image('ys_hat', x, ys_hat_img)
                    self.tbd.add_scalars('loss', x, { 'loss': self.meter_loss.value()[0], })

                if prefix is not None:
                    self.save(prefix.replace("ckpt", f"ckpt_{i:07d}"))

            del xs, ys, ys_hat, loss
            #input("press key to continue")

        # increase epoch #
        self.epoch += 1

    def test(self, data_loader, desc=None):
        self.encoder.eval()
        self.__reset_meters()
        with torch.no_grad():
            for i, (data) in tqdm(enumerate(data_loader), total=len(data_loader), desc=desc):
                xs, ys, frame_lens, label_lens, filenames = data
                if self.use_cuda:
                    xs = xs.cuda()
                ys_hat = self.encoder(xs)
                ys_hat = ys_hat.transpose(0, 1).contiguous()  # TxNxH
                frame_lens.div_(2)
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

    def predict(self, xs):
        self.encoder.eval()
        self.__reset_meters()
        with torch.no_grad():
            if self.use_cuda:
                xs = xs.cuda()
            ys_hat = self.encoder(xs, softmax=True)
        return ys_hat

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
        if not self.use_cuda:
            states = torch.load(file_path, map_location='cpu')
        else:
            states = torch.load(file_path)
        self.epoch = states["epoch"]

        self.__setup_networks()
        try:
            self.encoder.load_state_dict(states["model"])
        except:
            self.encoder.load_state_dict(states["conv"])
        self.optimizer.load_state_dict(states["optimizer"])
        if self.use_cuda:
            self.encoder.cuda()

