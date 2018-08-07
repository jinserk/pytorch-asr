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

from ..utils.dataset import AudioCTCDataset
from ..utils.dataloader import AudioNonSplitDataLoader
from ..utils.logger import logger, set_logfile, VisdomLogger, TensorboardLogger
from ..utils.misc import onehot2int, remove_duplicates, get_model_file_path
from ..utils import params as p

from ..kaldi.latgen import LatGenCTCDecoder

from .network import *


FRAME_REDUCE_FACTOR = 2

OPTIMIZER_TYPES = set([
    "sgd",
    "sgdr",
    "adamw",
])


class Trainer:

    def __init__(self, vlog=None, tlog=None, batch_size=8, init_lr=1e-4, max_norm=400,
                 use_cuda=False, log_dir='logs_deepspeech_ctc', model_prefix='deepspeech_ctc',
                 checkpoint=False, continue_from=None, opt_type="sgd", *args, **kwargs):
        # training parameters
        self.batch_size = batch_size
        self.init_lr = init_lr
        self.max_norm = max_norm
        self.use_cuda = use_cuda
        self.log_dir = log_dir
        self.model_prefix = model_prefix
        self.checkpoint = checkpoint
        self.epoch = 0

        # visual logging
        self.vlog = vlog
        if self.vlog is not None:
            self.vlog.add_plot(title='loss', xlabel='epoch')
        self.tlog = tlog

        # setup model
        self.model = DeepSpeech(num_classes=p.NUM_CTC_LABELS)
        if self.use_cuda:
            self.model.cuda()

        # setup loss
        self.loss = CTCLoss(blank=0, size_average=True)

        # setup optimizer
        assert opt_type in OPTIMIZER_TYPES
        parameters = self.model.parameters()
        if opt_type == "sgd":
            logger.info("using SGD")
            self.optimizer = torch.optim.SGD(parameters, lr=self.init_lr, momentum=0.9)
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10)
        elif opt_type == "sgdr":
            logger.info("using SGDR")
            self.optimizer = torch.optim.SGD(parameters, lr=self.init_lr, momentum=0.9)
            #self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.5)
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWithRestartsLR(self.optimizer, T_max=5, T_mult=2)
        elif opt_type == "adam":
            logger.info("using AdamW")
            self.optimizer = torch.optim.Adam(parameters, lr=self.init_lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0005, l2_reg=False)
            self.lr_scheduler = None

        # setup decoder for test
        self.decoder = LatGenCTCDecoder()

        if continue_from is not None:
            self.load(continue_from)

    def __get_model_name(self, desc):
        return str(get_model_file_path(self.log_dir, self.model_prefix, desc))

    def __remove_ckpt_files(self, epoch):
        for ckpt in Path(self.log_dir).rglob(f"*_epoch_{epoch:03d}_ckpt_*"):
            ckpt.unlink()

    def train_epoch(self, data_loader):
        self.model.train()
        num_ckpt = len(data_loader) // 10
        meter_loss = tnt.meter.MovingAverageValueMeter(len(data_loader) // 100)
        #meter_accuracy = tnt.meter.ClassErrorMeter(accuracy=True)
        #meter_confusion = tnt.meter.ConfusionMeter(p.NUM_CTC_LABELS, normalized=True)
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            logger.info(f"current lr = {self.lr_scheduler.get_lr()}")
        # count the number of supervised batches seen in this epoch
        t = tqdm(enumerate(data_loader), total=len(data_loader), desc="training")
        for i, (data) in t:
            xs, ys, frame_lens, label_lens, filenames, _ = data
            try:
                if self.use_cuda:
                    xs = xs.cuda()
                ys_hat = self.model(xs)
                ys_hat = ys_hat.transpose(0, 1).contiguous()  # TxNxH
                frame_lens = torch.ceil(frame_lens.float() / FRAME_REDUCE_FACTOR).int()
                #torch.set_printoptions(threshold=5000000)
                #print(ys_hat.shape, frame_lens, ys.shape, label_lens)
                #print(onehot2int(ys_hat).squeeze(), ys)
                loss = self.loss(ys_hat, ys, frame_lens, label_lens)
                loss_value = loss.item()
                inf = float("inf")
                if loss_value == inf or loss_value == -inf:
                    logger.warning("received an inf loss, setting loss value to 0")
                    loss_value = 0
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
                self.optimizer.step()
                del loss
            except Exception as e:
                print(e)
                print(filenames, frame_lens, label_lens)
            meter_loss.add(loss_value)
            t.set_description(f"training (loss: {meter_loss.value()[0]:.3f})")
            t.refresh()
            #self.meter_accuracy.add(ys_int, ys)
            #self.meter_confusion.add(ys_int, ys)
            if 0 < i < len(data_loader) and i % num_ckpt == 0:
                if self.vlog is not None:
                    self.vlog.add_point(
                        title = 'loss',
                        x = self.epoch+i/len(data_loader),
                        y = meter_loss.value()[0]
                    )
                if self.tlog is not None:
                    x = self.epoch * len(data_loader) + i
                    self.tlog.add_graph(self.model, xs)
                    xs_img = tvu.make_grid(xs[0, 0], normalize=True, scale_each=True)
                    self.tlog.add_image('xs', x, xs_img)
                    ys_hat_img = tvu.make_grid(ys_hat[0].transpose(0, 1), normalize=True, scale_each=True)
                    self.tlog.add_image('ys_hat', x, ys_hat_img)
                    self.tlog.add_scalars('loss', x, { 'loss': meter_loss.value()[0], })
                if self.checkpoint:
                    logger.info(f"training loss at epoch_{self.epoch:03d}_ckpt_{i:07d}: "
                                f"{meter_loss.value()[0]:5.3f}")
                    self.save(self.__get_model_name(f"epoch_{self.epoch:03d}_ckpt_{i:07d}"))
            #input("press key to continue")
        self.epoch += 1
        logger.info(f"epoch {self.epoch:03d}: "
                    f"training loss {meter_loss.value()[0]:5.3f} ")
                    #f"training accuracy {meter_accuracy.value()[0]:6.3f}")
        self.save(self.__get_model_name(f"epoch_{self.epoch:03d}"))
        self.__remove_ckpt_files(self.epoch-1)

    def validate(self, data_loader):
        "validate with label error rate by the edit distance between hyps and refs"
        self.model.eval()
        with torch.no_grad():
            N, D = 0, 0
            t = tqdm(enumerate(data_loader), total=len(data_loader), desc="validating")
            for i, (data) in t:
                xs, ys, frame_lens, label_lens, filenames, texts = data
                if self.use_cuda:
                    xs = xs.cuda()
                ys_hat = self.model(xs)
                # convert likes to ctc labels
                frame_lens = torch.ceil(frame_lens.float() / FRAME_REDUCE_FACTOR).int()
                hyps = [onehot2int(yh[:s]).squeeze() for yh, s in zip(ys_hat, frame_lens)]
                hyps = [remove_duplicates(h, blank=0) for h in hyps]
                # slice the targets
                pos = torch.cat((torch.zeros((1, ), dtype=torch.long), torch.cumsum(label_lens, dim=0)))
                refs = [ys[s:l] for s, l in zip(pos[:-1], pos[1:])]
                # calculate ler
                N += self.edit_distance(refs, hyps)
                D += sum(len(r) for r in refs)
                ler = N * 100. / D
                t.set_description(f"validating (LER: {ler:.2f} %)")
                t.refresh()
            logger.info(f"validating at epoch {self.epoch:03d}: LER {ler:.2f} %")

    def test(self, data_loader):
        "test with word error rate by the edit distance between hyps and refs"
        self.model.eval()
        with torch.no_grad():
            N, D = 0, 0
            t = tqdm(enumerate(data_loader), total=len(data_loader), desc="testing")
            for i, (data) in t:
                xs, ys, frame_lens, label_lens, filenames, texts = data
                if self.use_cuda:
                    xs = xs.cuda()
                ys_hat = self.model(xs)
                frame_lens = torch.ceil(frame_lens.float() / FRAME_REDUCE_FACTOR).int()
                # latgen decoding
                loglikes = torch.log(ys_hat)
                if self.use_cuda:
                    loglikes = loglikes.cpu()
                words, alignment, w_sizes, a_sizes = self.decoder(loglikes, frame_lens)
                hyps = [w[:s] for w, s in zip(words, w_sizes)]
                # convert target texts to word indices
                w2i = lambda w: self.decoder.wordi[w] if w in self.decoder.wordi else self.decoder.wordi['<unk>']
                refs = [[w2i(w) for w in t.strip().split()] for t in texts]
                # calculate wer
                N += self.edit_distance(refs, hyps)
                D += sum(len(r) for r in refs)
                wer = N * 100. / D
                t.set_description(f"testing (WER: {wer:.2f} %)")
                t.refresh()
            logger.info(f"testing at epoch {self.epoch:03d}: WER {wer:.2f} %")

    def edit_distance(self, refs, hyps):
        assert len(refs) == len(hyps)
        n = 0
        for ref, hyp in zip(refs, hyps):
            r = [chr(c) for c in ref]
            h = [chr(c) for c in hyp]
            n += Lev.distance(''.join(r), ''.join(h))
        return n

    def save(self, file_path, **kwargs):
        Path(file_path).parent.mkdir(mode=0o755, parents=True, exist_ok=True)
        logger.info(f"saving the model to {file_path}")
        states = kwargs
        states["epoch"] = self.epoch
        states["model"] = self.model.state_dict()
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
            states = torch.load(file_path, map_location='cuda:0')
        self.epoch = states["epoch"]
        self.model.load_state_dict(states["model"])
        self.optimizer.load_state_dict(states["optimizer"])
        #self.lr_scheduler.load_state_dict(states["lr_scheduler"])


def train(argv):
    import argparse
    parser = argparse.ArgumentParser(description="DeepSpeech AM with fully supervised training")
    # for training
    parser.add_argument('--data-path', default='data/aspire', type=str, help="dataset path to use in training")
    parser.add_argument('--min-len', default=1., type=float, help="min length of utterance to use in secs")
    parser.add_argument('--max-len', default=10., type=float, help="max length of utterance to use in secs")
    parser.add_argument('--num-workers', default=16, type=int, help="number of dataloader workers")
    parser.add_argument('--num-epochs', default=100, type=int, help="number of epochs to run")
    parser.add_argument('--batch-size', default=48, type=int, help="number of images (and labels) to be considered in a batch")
    parser.add_argument('--init-lr', default=1e-4, type=float, help="initial learning rate for Adam optimizer")
    parser.add_argument('--max-norm', default=400, type=int, help="norm cutoff to prevent explosion of gradients")
    # optional
    parser.add_argument('--use-cuda', default=False, action='store_true', help="use cuda")
    parser.add_argument('--visdom', default=False, action='store_true', help="use visdom logging")
    parser.add_argument('--tensorboard', default=False, action='store_true', help="use tensorboard logging")
    parser.add_argument('--seed', default=None, type=int, help="seed for controlling randomness in this example")
    parser.add_argument('--log-dir', default='./logs_deepspeech_ctc', type=str, help="filename for logging the outputs")
    parser.add_argument('--model-prefix', default='deepspeech_ctc', type=str, help="model file prefix to store")
    parser.add_argument('--checkpoint', default=True, action='store_true', help="save checkpoint")
    parser.add_argument('--continue-from', default=None, type=str, help="model file path to make continued from")
    parser.add_argument('--opt-type', default="sgd", type=str, help=f"optimizer type in {OPTIMIZER_TYPES}")
    parser.add_argument('--test-only', default=False, action='store_true', help="do only test")

    args = parser.parse_args(argv)

    print(f"begins logging to file: {str(Path(args.log_dir).resolve() / 'train.log')}")
    set_logfile(Path(args.log_dir, "train.log"))

    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"training command options: {' '.join(sys.argv)}")
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

    trainer = Trainer(vlog=vlog, tlog=tlog, **vars(args))

    # prepare data loaders
    datasets, data_loaders = dict(), dict()
    for mode, size in zip(["train", "dev"], [16000000, 16000]):
    #for mode, size in zip(["train", "dev"], [10, 2]):
        datasets[mode] = AudioCTCDataset(root=args.data_path, mode=mode, data_size=size,
                                         min_len=args.min_len, max_len=args.max_len)
        data_loaders[mode] = AudioNonSplitDataLoader(datasets[mode], batch_size=args.batch_size,
                                                     num_workers=args.num_workers, shuffle=True,
                                                     pin_memory=args.use_cuda, frame_shift=FRAME_REDUCE_FACTOR)

    # run inference for a certain number of epochs
    if not args.test_only:
        for i in range(trainer.epoch, args.num_epochs):
            trainer.train_epoch(data_loaders["train"])
            trainer.validate(data_loaders["dev"])
    # final test to know WER
    trainer.test(data_loaders["dev"])


if __name__ == "__main__":
    pass
