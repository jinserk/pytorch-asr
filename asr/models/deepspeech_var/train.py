#!python
import sys
import argparse
from pathlib import Path, PurePath

import torch
from torch.utils.data.dataset import ConcatDataset
from torch.utils.data.distributed import DistributedSampler

from apex import amp

from asr.utils.dataset import NonSplitTrainDataset, AudioSubset
from asr.utils.dataloader import NonSplitTrainDataLoader
from asr.utils.logger import logger, init_logger
from asr.utils import params as p
from asr.kaldi.latgen import LatGenCTCDecoder

from ..trainer import *
from .network import DeepSpeech


def batch_train(argv):
    parser = argparse.ArgumentParser(description="DeepSpeech AM with batch training")
    # for training
    parser.add_argument('--data-path', default='/d1/jbaik/ics-asr/data', type=str, help="dataset path to use in training")
    parser.add_argument('--num-epochs', default=200, type=int, help="number of epochs to run")
    parser.add_argument('--init-lr', default=1e-2, type=float, help="initial learning rate for the optimizer")
    parser.add_argument('--max-norm', default=0.1, type=int, help="norm cutoff to prevent explosion of gradients")
    # optional
    parser.add_argument('--use-cuda', default=False, action='store_true', help="use cuda")
    parser.add_argument('--fp16', default=False, action='store_true', help="use FP16 model")
    parser.add_argument('--visdom', default=False, action='store_true', help="use visdom logging")
    parser.add_argument('--visdom-host', default="127.0.0.1", type=str, help="visdom server ip address")
    parser.add_argument('--visdom-port', default=8097, type=int, help="visdom server port")
    parser.add_argument('--tensorboard', default=False, action='store_true', help="use tensorboard logging")
    parser.add_argument('--slack', default=False, action='store_true', help="use slackclient logging (need to set SLACK_API_TOKEN and SLACK_API_USER env_var")
    parser.add_argument('--seed', default=None, type=int, help="seed for controlling randomness in this example")
    parser.add_argument('--log-dir', default='./logs_deepspeech_var', type=str, help="filename for logging the outputs")
    parser.add_argument('--model-prefix', default='deepspeech_var', type=str, help="model file prefix to store")
    parser.add_argument('--checkpoint', default=False, action='store_true', help="save checkpoint")
    parser.add_argument('--continue-from', default=None, type=str, help="model file path to make continued from")
    parser.add_argument('--opt-type', default="sgdr", type=str, help=f"optimizer type in {OPTIMIZER_TYPES}")
    args = parser.parse_args(argv)

    init_distributed(args.use_cuda)
    init_logger(log_file="train.log", rank=get_rank(), **vars(args))
    set_seed(args.seed)

    # check fp16
    amp_handle = amp.init(enabled=True, enable_caching=True, verbose=False) if args.fp16 else None

    # prepare trainer object
    input_folding = 3
    model = DeepSpeech(num_classes=p.NUM_CTC_LABELS, input_folding=input_folding)
    trainer = NonSplitTrainer(model, amp_handle, **vars(args))
    labeler = trainer.decoder.labeler

    train_datasets = [
        NonSplitTrainDataset(labeler=labeler, manifest_file=f"{args.data_path}/aspire/train.csv", stride=input_folding),
        NonSplitTrainDataset(labeler=labeler, manifest_file=f"{args.data_path}/aspire/dev.csv", stride=input_folding),
        NonSplitTrainDataset(labeler=labeler, manifest_file=f"{args.data_path}/aspire/test.csv", stride=input_folding),
        NonSplitTrainDataset(labeler=labeler, manifest_file=f"{args.data_path}/swbd/train.csv", stride=input_folding),
    ]

    datasets = {
        "train3" : ConcatDataset([AudioSubset(d, max_len=3) for d in train_datasets]),
        "train5" : ConcatDataset([AudioSubset(d, max_len=5) for d in train_datasets]),
        "train10": ConcatDataset([AudioSubset(d, max_len=10) for d in train_datasets]),
        "train15": ConcatDataset([AudioSubset(d, max_len=15) for d in train_datasets]),
        "dev"    : NonSplitTrainDataset(labeler=labeler, manifest_file=f"{args.data_path}/swbd/eval2000.csv", stride=input_folding),
        "test"   : NonSplitTrainDataset(labeler=labeler, manifest_file=f"{args.data_path}/swbd/rt03.csv", stride=input_folding),
    }

    dataloaders = {
        "train3" : NonSplitTrainDataLoader(datasets["train3"],
                                           sampler=(DistributedSampler(datasets["train3"])
                                                    if is_distributed() else None),
                                           batch_size=64, num_workers=32,
                                           shuffle=(not is_distributed()),
                                           pin_memory=args.use_cuda),
        "train5" : NonSplitTrainDataLoader(datasets["train5"],
                                           sampler=(DistributedSampler(datasets["train5"])
                                                    if is_distributed() else None),
                                           batch_size=64, num_workers=32,
                                           shuffle=(not is_distributed()),
                                           pin_memory=args.use_cuda),
        "train10": NonSplitTrainDataLoader(datasets["train10"],
                                           sampler=(DistributedSampler(datasets["train10"])
                                                    if is_distributed() else None),
                                           batch_size=64, num_workers=16,
                                           shuffle=(not is_distributed()),
                                           pin_memory=args.use_cuda),
        "train15": NonSplitTrainDataLoader(datasets["train15"],
                                           sampler=(DistributedSampler(datasets["train15"])
                                                    if is_distributed() else None),
                                           batch_size=64, num_workers=16,
                                           shuffle=(not is_distributed()),
                                           pin_memory=args.use_cuda),
        "dev"    : NonSplitTrainDataLoader(datasets["dev"],
                                           batch_size=32, num_workers=16,
                                           shuffle=False, pin_memory=args.use_cuda),
        "test"   : NonSplitTrainDataLoader(datasets["test"],
                                           batch_size=32, num_workers=16,
                                           shuffle=False, pin_memory=args.use_cuda),
    }

    # run inference for a certain number of epochs
    for i in range(trainer.epoch, args.num_epochs):
        #if i < 1:
        #    trainer.train_epoch(dataloaders["train3"])
        #    trainer.validate(dataloaders["dev"])
        if i < 5:
            trainer.train_epoch(dataloaders["train5"])
            trainer.validate(dataloaders["dev"])
        elif i < (5 + 10):
            trainer.train_epoch(dataloaders["train10"])
            trainer.validate(dataloaders["dev"])
        else:
            trainer.train_epoch(dataloaders["train15"])
            trainer.validate(dataloaders["dev"])

    # final test to know WER
    trainer.test(dataloaders["test"])


def train(argv):
    parser = argparse.ArgumentParser(description="DeepSpeech AM with fully supervised training")
    # for training
    parser.add_argument('--data-path', default='/d1/jbaik/ics-asr/data', type=str, help="dataset path to use in training")
    parser.add_argument('--min-len', default=1., type=float, help="min length of utterance to use in secs")
    parser.add_argument('--max-len', default=15., type=float, help="max length of utterance to use in secs")
    parser.add_argument('--batch-size', default=64, type=int, help="number of images (and labels) to be considered in a batch")
    parser.add_argument('--num-workers', default=32, type=int, help="number of dataloader workers")
    parser.add_argument('--num-epochs', default=100, type=int, help="number of epochs to run")
    parser.add_argument('--init-lr', default=1e-2, type=float, help="initial learning rate for the optimizer")
    parser.add_argument('--max-norm', default=0.1, type=int, help="norm cutoff to prevent explosion of gradients")
    # optional
    parser.add_argument('--use-cuda', default=False, action='store_true', help="use cuda")
    parser.add_argument('--fp16', default=False, action='store_true', help="use FP16 model")
    parser.add_argument('--visdom', default=False, action='store_true', help="use visdom logging")
    parser.add_argument('--visdom-host', default="127.0.0.1", type=str, help="visdom server ip address")
    parser.add_argument('--visdom-port', default=8097, type=int, help="visdom server port")
    parser.add_argument('--tensorboard', default=False, action='store_true', help="use tensorboard logging")
    parser.add_argument('--slack', default=False, action='store_true', help="use slackclient logging (need to set SLACK_API_TOKEN and SLACK_API_USER env_var")
    parser.add_argument('--seed', default=None, type=int, help="seed for controlling randomness in this example")
    parser.add_argument('--log-dir', default='./logs_deepspeech_var', type=str, help="filename for logging the outputs")
    parser.add_argument('--model-prefix', default='deepspeech_var', type=str, help="model file prefix to store")
    parser.add_argument('--checkpoint', default=False, action='store_true', help="save checkpoint")
    parser.add_argument('--continue-from', default=None, type=str, help="model file path to make continued from")
    parser.add_argument('--opt-type', default="sgdr", type=str, help=f"optimizer type in {OPTIMIZER_TYPES}")
    args = parser.parse_args(argv)

    init_distributed(args.use_cuda)
    init_logger(log_file="train.log", rank=get_rank(), **vars(args))
    set_seed(args.seed)

    # check fp16
    amp_handle = amp.init(enabled=True, enable_caching=True, verbose=False) if args.fp16 else None

    # prepare trainer object
    input_folding = 3
    model = DeepSpeech(num_classes=p.NUM_CTC_LABELS, input_folding=input_folding)
    trainer = NonSplitTrainer(model, amp_handle, **vars(args))
    labeler = trainer.decoder.labeler

    train_datasets = [
        NonSplitTrainDataset(labeler=labeler, manifest_file=f"{args.data_path}/aspire/train.csv", stride=input_folding),
        NonSplitTrainDataset(labeler=labeler, manifest_file=f"{args.data_path}/aspire/dev.csv", stride=input_folding),
        NonSplitTrainDataset(labeler=labeler, manifest_file=f"{args.data_path}/aspire/test.csv", stride=input_folding),
        NonSplitTrainDataset(labeler=labeler, manifest_file=f"{args.data_path}/swbd/train.csv", stride=input_folding),
    ]

    datasets = {
        "train": ConcatDataset([AudioSubset(d, data_size=0, min_len=args.min_len, max_len=args.max_len)
                                for d in train_datasets]),
        "dev"  : NonSplitTrainDataset(labeler=labeler, manifest_file=f"{args.data_path}/swbd/eval2000.csv", stride=input_folding),
        "test" : NonSplitTrainDataset(labeler=labeler, manifest_file=f"{args.data_path}/swbd/rt03.csv", stride=input_folding),
    }

    dataloaders = {
        "train": NonSplitTrainDataLoader(datasets["train"],
                                         sampler=(DistributedSampler(datasets["train"])
                                                  if is_distributed() else None),
                                         batch_size=args.batch_size,
                                         num_workers=args.num_workers,
                                         shuffle=(not is_distributed()),
                                         pin_memory=args.use_cuda),
        "dev"  : NonSplitTrainDataLoader(datasets["dev"],
                                         batch_size=16, num_workers=8,
                                         shuffle=False, pin_memory=args.use_cuda),
        "test" : NonSplitTrainDataLoader(datasets["test"],
                                         batch_size=16, num_workers=8,
                                         shuffle=False, pin_memory=args.use_cuda),
    }

    # run inference for a certain number of epochs
    for i in range(trainer.epoch, args.num_epochs):
        trainer.train_epoch(dataloaders["train"])
        trainer.validate(dataloaders["dev"])

    # final test to know WER
    trainer.test(dataloaders["test"])


def test(argv):
    parser = argparse.ArgumentParser(description="DeepSpeech AM testing")
    # for testing
    parser.add_argument('--data-path', default='data/swbd', type=str, help="dataset path to use in training")
    parser.add_argument('--min-len', default=1., type=float, help="min length of utterance to use in secs")
    parser.add_argument('--max-len', default=20., type=float, help="max length of utterance to use in secs")
    parser.add_argument('--num-workers', default=8, type=int, help="number of dataloader workers")
    parser.add_argument('--batch-size', default=16, type=int, help="number of images (and labels) to be considered in a batch")
    # optional
    parser.add_argument('--use-cuda', default=False, action='store_true', help="use cuda")
    parser.add_argument('--fp16', default=False, action='store_true', help="use FP16 model")
    parser.add_argument('--log-dir', default='./logs_deepspeech_var', type=str, help="filename for logging the outputs")
    parser.add_argument('--continue-from', default=None, type=str, help="model file path to make continued from")
    args = parser.parse_args(argv)

    init_logger(log_file="test.log", **vars(args))

    assert args.continue_from is not None

    # check fp16
    amp_handle = amp.init(enabled=True, enable_caching=True, verbose=False) if args.fp16 else None

    input_folding = 3
    model = DeepSpeech(num_classes=p.NUM_CTC_LABELS, input_folding=input_folding)
    trainer = NonSplitTrainer(model, amp_handle, **vars(args))
    labeler = trainer.decoder.labeler

    manifest = f"{args.data_path}/eval2000.csv"
    dataset = AudioSubset(NonSplitTrainDataset(labeler=labeler, manifest_file=manifest, stride=input_folding),
                          max_len=args.max_len, min_len=args.min_len)
    dataloader = NonSplitTrainDataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                         shuffle=True, pin_memory=args.use_cuda)

    trainer.validate(dataloader)


if __name__ == "__main__":
    pass
