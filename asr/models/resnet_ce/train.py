#!python
import sys
from pathlib import Path, PurePath
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torchvision.utils as tvu
from warpctc_pytorch import CTCLoss
import torchnet as tnt
import Levenshtein as Lev

from ..utils.dataset import AudioCEDataset
from ..utils.dataloader import AudioNonSplitDataLoader
from ..utils.logger import logger, set_logfile, VisdomLogger, TensorboardLogger
from ..utils.misc import onehot2int, int2onehot, get_model_file_path
from ..utils import params as p

from .network import *


class Trainer:

    def __init__(self, vlog=None, tlog=None, batch_size=8, init_lr=1e-4, max_norm=400,
                 use_cuda=False, log_dir='logs_deepspeech_ctc', model_prefix='deepspeech_ctc_aspire',
                 checkpoint=False, num_ckpt=10000, continue_from=False, *args, **kwargs):
        super().__init__()

        self.batch_size = batch_size
        self.init_lr = init_lr
        self.max_norm = max_norm
        self.use_cuda = use_cuda

        self.log_dir = log_dir
        self.model_prefix = model_prefix
        self.checkpoint = checkpoint
        self.num_ckpt = num_ckpt

        self.epoch = 0
        self.opt = "sgd"

        self.vlog = vlog
        if self.vlog is not None:
            self.vlog.add_plot(title='loss', xlabel='epoch')
        self.tlog = tlog

        if continue_from is None:
            self.__setup_networks()
        else:
            self.load(continue_from)

    def __setup_networks(self):
        # setup networks
        self.encoder = resnet101(num_classes=p.NUM_CTC_LABELS)
        if self.use_cuda:
            self.encoder.cuda()
        # setup loss
        self.loss = nn.CrossEntropyLoss()
        # setup optimizer
        parameters = self.encoder.parameters()
        if self.opt == "adam":
            logger.info("using AdamW")
            self.optimizer = torch.optim.Adam(parameters, lr=self.init_lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0005, l2_reg=False)
            self.lr_scheduler = None
        else:
            logger.info("using SGDR")
            self.optimizer = torch.optim.SGD(parameters, lr=self.init_lr, momentum=0.9)
            #self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.5)
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWithRestartsLR(self.optimizer, T_max=5, T_mult=2)

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
            xs, ys, frame_lens, label_lens, filenames = data
            try:
                if self.use_cuda:
                    xs, ys = xs.cuda(), ys.cuda()
                ys_hat = self.encoder(xs)
                ys_hat_cat = torch.cat([ys_hat[b, :frame_lens[b], :] for b in range(ys_hat.size(0))])
                #torch.set_printoptions(threshold=5000000)
                #print(ys_hat.shape, frame_lens, ys.shape, label_lens)
                #print(onehot2int(ys_hat).squeeze(), ys)
                loss = self.loss(ys_hat_cat, ys.long())
                #print(loss)
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.encoder.parameters(), self.max_norm)
                self.optimizer.step()
            except Exception as e:
                print(e)
                print(filenames, frame_lens, label_lens)
            #ys_int = onehot2int(ys_hat).squeeze()
            meter_loss.add(loss.item())
            t.set_description(f"training (loss: {meter_loss.value()[0]:.3f})")
            t.refresh()
            #self.meter_accuracy.add(ys_int, ys)
            #self.meter_confusion.add(ys_int, ys)
            if 0 < i < len(data_loader) and i % self.num_ckpt == 0:
                if self.vlog is not None:
                    self.vlog.add_point(
                        title = 'loss',
                        x = self.epoch+i/len(data_loader),
                        y = meter_loss.value()[0]
                    )
                if self.tlog is not None:
                    x = self.epoch * len(data_loader) + i
                    self.tlog.add_graph(self.encoder, xs)
                    xs_img = tvu.make_grid(xs[0, 0], normalize=True, scale_each=True)
                    self.tlog.add_image('xs', x, xs_img)
                    ys_hat_img = tvu.make_grid(ys_hat[0].transpose(0, 1), normalize=True, scale_each=True)
                    self.tlog.add_image('ys_hat', x, ys_hat_img)
                    self.tlog.add_scalars('loss', x, { 'loss': meter_loss.value()[0], })
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
            xs, ys, frame_lens, label_lens, filenames = data
            if self.use_cuda:
                xs, ys = xs.cuda(), ys.cuda()
            ys_hat = self.encoder(xs)
            ys_hat_cat = torch.cat([ys_hat[b, :frame_lens[b], :] for b in range(ys_hat.size(0))])
            loss = self.loss(ys_hat_cat, ys.long())
            meter_loss.add(loss.item())
            #meter_accuracy.add(ys_hat.data, ys_int)
            #meter_confusion.add(ys_hat.data, ys_int)
            del xs, ys, loss, ys_hat
        logger.info(f"epoch {self.epoch:03d}: "
                    f"validating loss {meter_loss.value()[0]:5.3f} ")
                    #f"validating accuracy {meter_accuracy.value()[0]:6.3f}")

    def wer(self, s1, s2):
        # build mapping of words to integers
        b = set(s1.split() + s2.split())
        word2char = dict(zip(b, range(len(b))))
        # map the words to a char array (Levenshtein packages only accepts strings)
        w1 = [chr(word2char[w]) for w in s1.split()]
        w2 = [chr(word2char[w]) for w in s2.split()]
        return Lev.distance(''.join(w1), ''.join(w2))

    def cer(self, s1, s2, is_char=False):
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
        self.encoder.load_state_dict(states["model"])
        self.optimizer.load_state_dict(states["optimizer"])
        self.lr_scheduler.load_state_dict(states["lr_scheduler"])
        if self.use_cuda:
            self.encoder.cuda()


def train(argv):
    import argparse
    parser = argparse.ArgumentParser(description="ResNet AM with fully supervised training with cross entropy loss")
    # for training
    parser.add_argument('--data-path', default='data/aspire', type=str, help="dataset path to use in training")
    parser.add_argument('--min-len', default=1., type=float, help="min length of utterance to use in secs")
    parser.add_argument('--max-len', default=15., type=float, help="max length of utterance to use in secs")
    parser.add_argument('--num-workers', default=4, type=int, help="number of dataloader workers")
    parser.add_argument('--num-epochs', default=100, type=int, help="number of epochs to run")
    parser.add_argument('--batch-size', default=8, type=int, help="number of images (and labels) to be considered in a batch")
    parser.add_argument('--init-lr', default=1e-4, type=float, help="initial learning rate for Adam optimizer")
    parser.add_argument('--max-norm', default=400, type=int, help="norm cutoff to prevent explosion of gradients")
    # optional
    parser.add_argument('--use-cuda', default=False, action='store_true', help="use cuda")
    parser.add_argument('--visdom', default=False, action='store_true', help="use visdom logging")
    parser.add_argument('--tensorboard', default=False, action='store_true', help="use tensorboard logging")
    parser.add_argument('--seed', default=None, type=int, help="seed for controlling randomness in this example")
    parser.add_argument('--log-dir', default='./logs_resnet_ce', type=str, help="filename for logging the outputs")
    parser.add_argument('--model-prefix', default='resnet_ce', type=str, help="model file prefix to store")
    parser.add_argument('--checkpoint', default=False, action='store_true', help="save checkpoint")
    parser.add_argument('--num-ckpt', default=10000, type=int, help="number of batch-run to save checkpoints")
    parser.add_argument('--continue-from', default=None, type=str, help="model file path to make continued from")

    args = parser.parse_args(argv)

    print(f"begins logging to file: {str(Path(args.log_dir).resolve() / 'train.log')}")
    set_logfile(Path(args.log_dir, "train.log"))

    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"Training started with command: {' '.join(sys.argv)}")
    args_str = [f"{k}={v}" for (k, v) in vars(args).items()]
    logger.info(f"args: {' '.join(args_str)}")

    if args.use_cuda:
        logger.info("using cuda")

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if args.use_cuda:
            torch.cuda.manual_seed(args.seed)

    vlog = None
    if args.visdom:
        try:
            logger.info("using visdom")
            title = str(Path(args.log_dir).name)
            vlog = VisdomLogger(env=title)
        except:
            logger.info("error to use visdom")
            vlog = None

    tlog = None
    if args.tensorboard:
        try:
            logger.info("using tensorboard")
            tlog = TensorboardLogger(PurePath(args.log_dir, 'tensorboard'))
        except:
            logger.info("error to use tensorboard")
            tlog = None

    model = Trainer(vlog=vlog, tlog=tlog, **vars(args))

    # initializing local variables to maintain the best validation accuracy
    # seen across epochs over the supervised training set
    best_valid_acc = 0.0

    # if you want to limit the datasets' entry size
    sizes = { "train": 1600000, "dev": 1600 }
    #sizes = { "train": 10000, "dev": 100 }

    # prepare data loaders
    datasets, data_loaders = dict(), dict()
    for mode in ["train", "dev"]:
        datasets[mode] = AudioCEDataset(root=args.data_path, mode=mode, data_size=sizes[mode],
                                        min_len=args.min_len, max_len=args.max_len)
        data_loaders[mode] = AudioNonSplitDataLoader(datasets[mode], batch_size=args.batch_size,
                                                     num_workers=args.num_workers, shuffle=True,
                                                     pin_memory=args.use_cuda)

    # run inference for a certain number of epochs
    for i in range(model.epoch, args.num_epochs):
        # get the losses for an epoch
        model.train_epoch(data_loaders["train"])
        # validate
        model.test(data_loaders["dev"], "validating")

        # update the best validation accuracy and the corresponding
        # testing accuracy and the state of the parent module (including the networks)
        #if best_valid_acc < model.meter_accuracy.value()[0]:
        #    best_valid_acc = model.meter_accuracy.value()[0]

    # test
    #model.test(data_loaders["test"], "testing   ")

    #logger.info(f"best validation accuracy {best_valid_acc:6.3f} "
    #            f"test accuracy {model.meter_accuracy.value()[0]:6.3f}")


if __name__ == "__main__":
    pass
