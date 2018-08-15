#!python
import sys
import argparse
from pathlib import Path, PurePath

import torch
import torch.nn as nn
from torch.utils.data.dataset import ConcatDataset
from warpctc_pytorch import CTCLoss

from asr.utils.dataset import SplitTrainDataset, AudioSubset
from asr.utils.dataloader import SplitTrainDataLoader
from asr.utils.logger import logger, set_logfile, version_log
from asr.utils.misc import onehot2int , remove_duplicates
from asr.utils import params as p
from asr.kaldi.latgen import LatGenCTCDecoder

from ..trainer import FRAME_REDUCE_FACTOR, OPTIMIZER_TYPES, set_seed, Trainer
from .network import resnet50, resnet101


class SplitTrainer(Trainer):

    def unit_train(self, data):
        xs, ys, frame_lens, label_lens, filenames, _ = data
        try:
            if self.use_cuda:
                xs = xs.cuda()
            ys_hat = self.model(xs)
            ys_hat = ys_hat.unsqueeze(dim=0).transpose(1, 2)
            pos = torch.cat((torch.zeros((1, ), dtype=torch.long), torch.cumsum(frame_lens, dim=0)))
            ys_hats = [ys_hat.narrow(2, p, l).clone() for p, l in zip(pos[:-1], frame_lens)]
            max_len = torch.max(frame_lens)
            ys_hats = [nn.ConstantPad1d((0, max_len-yh.size(2)), 0)(yh) for yh in ys_hats]
            ys_hat = torch.cat(ys_hats).transpose(1, 2).transpose(0, 1)
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
            print(filenames, frame_lens, label_lens)
            raise
        return loss_value

    def unit_validate(self, data):
        xs, ys, frame_lens, label_lens, filenames, _ = data
        if self.use_cuda:
            xs = xs.cuda()
        ys_hat = self.model(xs)
        pos = torch.cat((torch.zeros((1, ), dtype=torch.long), torch.cumsum(frame_lens, dim=0)))
        ys_hat = [ys_hat.narrow(0, p, l).clone() for p, l in zip(pos[:-1], frame_lens)]
        # convert likes to ctc labels
        hyps = [onehot2int(yh[:s]).squeeze() for yh, s in zip(ys_hat, frame_lens)]
        hyps = [remove_duplicates(h, blank=0) for h in hyps]
        # slice the targets
        pos = torch.cat((torch.zeros((1, ), dtype=torch.long), torch.cumsum(label_lens, dim=0)))
        refs = [ys[s:l] for s, l in zip(pos[:-1], pos[1:])]
        return hyps, refs

    def unit_test(self, data):
        xs, ys, frame_lens, label_lens, filenames, texts = data
        if self.use_cuda:
            xs = xs.cuda()
        ys_hat = self.model(xs)
        ys_hat = ys_hat.unsqueeze(dim=0).transpose(1, 2)
        pos = torch.cat((torch.zeros((1, ), dtype=torch.long), torch.cumsum(frame_lens, dim=0)))
        ys_hats = [ys_hat.narrow(2, p, l).clone() for p, l in zip(pos[:-1], frame_lens)]
        max_len = torch.max(frame_lens)
        ys_hats = [nn.ConstantPad1d((0, max_len-yh.size(2)), 0)(yh) for yh in ys_hats]
        ys_hat = torch.cat(ys_hats).transpose(1, 2)
        # latgen decoding
        loglikes = torch.log(ys_hat)
        if self.use_cuda:
            loglikes = loglikes.cpu()
        words, alignment, w_sizes, a_sizes = self.decoder(loglikes, frame_lens)
        hyps = [w[:s] for w, s in zip(words, w_sizes)]
        # convert target texts to word indices
        w2i = self.decoder.labeler.word2idx
        refs = [[w2i(w.strip()) for w in t.strip().split()] for t in texts]
        return hyps, refs



def batch_train(argv):
    parser = argparse.ArgumentParser(description="ResNet AM with batch training")
    # for training
    parser.add_argument('--num-epochs', default=100, type=int, help="number of epochs to run")
    parser.add_argument('--init-lr', default=1e-4, type=float, help="initial learning rate for Adam optimizer")
    parser.add_argument('--max-norm', default=400, type=int, help="norm cutoff to prevent explosion of gradients")
    # optional
    parser.add_argument('--use-cuda', default=False, action='store_true', help="use cuda")
    parser.add_argument('--visdom', default=False, action='store_true', help="use visdom logging")
    parser.add_argument('--visdom-host', default="127.0.0.1", type=str, help="visdom server ip address")
    parser.add_argument('--visdom-port', default=8097, type=int, help="visdom server port")
    parser.add_argument('--tensorboard', default=False, action='store_true', help="use tensorboard logging")
    parser.add_argument('--seed', default=None, type=int, help="seed for controlling randomness in this example")
    parser.add_argument('--log-dir', default='./logs_resnet_split', type=str, help="filename for logging the outputs")
    parser.add_argument('--model-prefix', default='resnet_split', type=str, help="model file prefix to store")
    parser.add_argument('--checkpoint', default=True, action='store_true', help="save checkpoint")
    parser.add_argument('--continue-from', default=None, type=str, help="model file path to make continued from")
    parser.add_argument('--opt-type', default="sgdr", type=str, help=f"optimizer type in {OPTIMIZER_TYPES}")

    args = parser.parse_args(argv)

    set_logfile(Path(args.log_dir, "train.log"))
    version_log(args)
    set_seed(args.seed)

    # prepare trainer object
    model = resnet50(num_classes=p.NUM_CTC_LABELS)
    trainer = SplitTrainer(model, **vars(args))
    labeler = trainer.decoder.labeler

    train_datasets = [
        SplitTrainDataset(labeler=labeler, manifest_file="data/aspire/train.csv"),
        SplitTrainDataset(labeler=labeler, manifest_file="data/aspire/dev.csv"),
        SplitTrainDataset(labeler=labeler, manifest_file="data/aspire/test.csv"),
        SplitTrainDataset(labeler=labeler, manifest_file="data/swbd/train.csv"),
    ]

    datasets = {
        "train5" : ConcatDataset([AudioSubset(d, max_len=5) for d in train_datasets]),
        "train10": ConcatDataset([AudioSubset(d, max_len=10) for d in train_datasets]),
        "train15": ConcatDataset([AudioSubset(d, max_len=15) for d in train_datasets]),
        "dev"    : SplitTrainDataset(labeler=labeler, manifest_file="data/swbd/eval2000.csv"),
        "test"   : SplitTrainDataset(labeler=labeler, manifest_file="data/swbd/rt03.csv"),
    }
    dataloaders = {
        "train5" : SplitTrainDataLoader(datasets["train5"], batch_size=1, num_workers=0,
                                        shuffle=True, pin_memory=args.use_cuda),
        "train10": SplitTrainDataLoader(datasets["train10"], batch_size=1, num_workers=0,
                                        shuffle=True, pin_memory=args.use_cuda),
        "train15": SplitTrainDataLoader(datasets["train15"], batch_size=1, num_workers=0,
                                        shuffle=True, pin_memory=args.use_cuda),
        "dev"    : SplitTrainDataLoader(datasets["dev"], batch_size=1, num_workers=0,
                                        shuffle=False, pin_memory=args.use_cuda),
        "test"   : SplitTrainDataLoader(datasets["test"], batch_size=1, num_workers=0,
                                        shuffle=True, pin_memory=args.use_cuda),
    }

    # run inference for a certain number of epochs
    for i in range(trainer.epoch, args.num_epochs):
        if i < 10:
            trainer.train_epoch(dataloaders["train5"])
            trainer.validate(dataloaders["dev"])
        elif i < 30:
            trainer.train_epoch(dataloaders["train10"])
            trainer.validate(dataloaders["dev"])
        else:
            trainer.train_epoch(dataloaders["train15"])
            trainer.validate(dataloaders["dev"])

    # final test to know WER
    trainer.test(dataloaders["test"])


def train(argv):
    parser = argparse.ArgumentParser(description="ResNet AM with fully supervised training")
    # for training
    parser.add_argument('--data-path', default='data/aspire2', type=str, help="dataset path to use in training")
    parser.add_argument('--min-len', default=1., type=float, help="min length of utterance to use in secs")
    parser.add_argument('--max-len', default=10., type=float, help="max length of utterance to use in secs")
    parser.add_argument('--batch-size', default=1, type=int, help="number of images (and labels) to be considered in a batch")
    parser.add_argument('--num-workers', default=0, type=int, help="number of dataloader workers")
    parser.add_argument('--num-epochs', default=100, type=int, help="number of epochs to run")
    parser.add_argument('--init-lr', default=1e-4, type=float, help="initial learning rate for Adam optimizer")
    parser.add_argument('--max-norm', default=400, type=int, help="norm cutoff to prevent explosion of gradients")
    # optional
    parser.add_argument('--use-cuda', default=False, action='store_true', help="use cuda")
    parser.add_argument('--visdom', default=False, action='store_true', help="use visdom logging")
    parser.add_argument('--visdom-host', default="127.0.0.1", type=str, help="visdom server ip address")
    parser.add_argument('--visdom-port', default=8097, type=int, help="visdom server port")
    parser.add_argument('--tensorboard', default=False, action='store_true', help="use tensorboard logging")
    parser.add_argument('--seed', default=None, type=int, help="seed for controlling randomness in this example")
    parser.add_argument('--log-dir', default='./logs_resnet_split', type=str, help="filename for logging the outputs")
    parser.add_argument('--model-prefix', default='resnet_split', type=str, help="model file prefix to store")
    parser.add_argument('--checkpoint', default=True, action='store_true', help="save checkpoint")
    parser.add_argument('--continue-from', default=None, type=str, help="model file path to make continued from")
    parser.add_argument('--opt-type', default="sgdr", type=str, help=f"optimizer type in {OPTIMIZER_TYPES}")

    args = parser.parse_args(argv)

    set_logfile(Path(args.log_dir, "train.log"))
    version_log(args)
    set_seed(args.seed)

    # prepare trainer object
    model = resnet50(num_classes=p.NUM_CTC_LABELS)
    trainer = SplitTrainer(model=model, **vars(args))
    labeler = trainer.decoder.labeler

    data_opts = {
        "train" : (f"{args.data_path}/train.csv", 0),
        "dev"   : (f"{args.data_path}/dev.csv", 0),
        "test"  : (f"{args.data_path}/test.csv", 0),
    }
    datasets, dataloaders = dict(), dict()
    for k, (v) in data_opts.items():
        manifest_file, data_size = v
        datasets[k] = AudioSubset(SplitTrainDataset(labeler=labeler, manifest_file=manifest_file),
                                  data_size=data_size, min_len=args.min_len, max_len=args.max_len)
        dataloaders[k] = SplitTrainDataLoader(datasets[k], batch_size=args.batch_size,
                                              num_workers=args.num_workers, shuffle=True,
                                              pin_memory=args.use_cuda)

    # run inference for a certain number of epochs
    for i in range(trainer.epoch, args.num_epochs):
        trainer.train_epoch(dataloaders["train"])
        trainer.validate(dataloaders["dev"])

    # final test to know WER
    trainer.test(dataloaders["test"])


def test(argv):
    parser = argparse.ArgumentParser(description="ResNet AM testing")
    # for testing
    parser.add_argument('--data-path', default='data/swbd', type=str, help="dataset path to use in training")
    parser.add_argument('--min-len', default=1., type=float, help="min length of utterance to use in secs")
    parser.add_argument('--max-len', default=100., type=float, help="max length of utterance to use in secs")
    parser.add_argument('--num-workers', default=0, type=int, help="number of dataloader workers")
    parser.add_argument('--batch-size', default=4, type=int, help="number of images (and labels) to be considered in a batch")
    # optional
    parser.add_argument('--use-cuda', default=False, action='store_true', help="use cuda")
    parser.add_argument('--log-dir', default='./logs_resnet_split', type=str, help="filename for logging the outputs")
    parser.add_argument('--continue-from', default=None, type=str, help="model file path to make continued from")

    args = parser.parse_args(argv)

    set_logfile(Path(args.log_dir, "test.log"))
    version_log(args)

    assert args.continue_from is not None

    if args.use_cuda:
        logger.info("using cuda")

    model = resnet50(num_classes=p.NUM_CTC_LABELS)
    trainer = SplitTrainer(model=model, **vars(args))
    labeler = trainer.decoder.labeler

    manifest = f"{args.data_path}/eval2000.csv"
    dataset = AudioSubset(SplitTrainDataset(labeler=labeler, manifest_file=manifest),
                          max_len=args.max_len, min_len=args.min_len)
    dataloader = SplitTrainDataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                      shuffle=True, pin_memory=args.use_cuda)

    trainer.test(dataloader)


if __name__ == "__main__":
    pass
