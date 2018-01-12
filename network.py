#!python

import sys
from random import random

import torch
from torch.autograd import Variable
from torchvision import datasets, transforms

import probtorch_env
import probtorch

from logger import log
from model import Encoder, Decoder, NUM_PIXELS, NUM_DIGITS, EPS


file_suffix = "pth.tar"


class Network(object):

    def __init__(self, args):
        self.enc = Encoder()
        self.dec = Decoder()
        try:
            self.batch_size = args.batch_size
            self.num_samples = args.num_samples
            self.label_fraction = args.label_fraction
            self.cuda = args.cuda
            self.lr = args.lr
        except:
            try:
                self.load(args.model_dir, args.model_prefix)
            except:
                log.error("you have to provide proper args or model file info")
                sys.exit(1)

        self.initialize()

    def initialize(self):
        if self.cuda:
            torch.nn.DataParallel(self.enc).cuda()
            torch.nn.DataParallel(self.dec).cuda()

        parameters = list(self.enc.parameters()) + list(self.dec.parameters())
        self.optimizer = torch.optim.Adam(parameters, lr=self.lr, betas=(0.9, 0.999))

    def elbo(self, q, p, alpha=0.1):
        if self.num_samples is None:
            return probtorch.objectives.montecarlo.elbo(q, p, sample_dim=None, batch_dim=0, alpha=alpha)
        else:
            return probtorch.objectives.montecarlo.elbo(q, p, sample_dim=0, batch_dim=1, alpha=alpha)

    def train_epoch(self, data, batch_size=None, label_fraction=None, label_mask={}):
        if batch_size is None:
            batch_size = self.batch_size
        if label_fraction is None:
            label_fraction = self.label_fraction

        epoch_elbo = 0.0
        self.enc.train()
        self.dec.train()

        N = 0
        for b, (images, labels) in enumerate(data):
            if images.size()[0] != batch_size:
                continue
            N += batch_size
            images = images.view(-1, NUM_PIXELS)
            labels_onehot = torch.zeros(batch_size, NUM_DIGITS)
            labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
            labels_onehot = torch.clamp(labels_onehot, EPS, 1 - EPS)
            if self.cuda:
                images = images.cuda()
                labels_onehot = labels_onehot.cuda()
            images = Variable(images)
            labels_onehot = Variable(labels_onehot)
            self.optimizer.zero_grad()
            if b not in label_mask:
                label_mask[b] = (random() < label_fraction)
            if label_mask[b]:
                q = self.enc(images, labels_onehot, num_samples=self.num_samples)
            else:
                q = self.enc(images, num_samples=self.num_samples)
            p = self.dec(images, q, num_samples=self.num_samples)
            loss = -self.elbo(q, p)
            loss.backward()
            self.optimizer.step()
            if self.cuda:
                loss = loss.cpu()
            epoch_elbo -= loss.data.numpy()[0]
        return epoch_elbo / N, label_mask

    def test(self, data, batch_size=None, infer=True):
        if batch_size is None:
            batch_size = self.batch_size

        self.enc.eval()
        self.dec.eval()
        epoch_elbo = 0.0
        epoch_correct = 0
        N = 0
        with torch.no_grad():
            for b, (images, labels) in enumerate(data):
                if images.size()[0] != batch_size:
                    continue
                N += batch_size
                images = images.view(-1, NUM_PIXELS)
                if self.cuda:
                    images = images.cuda()
                images = Variable(images)
                q = self.enc(images, num_samples=self.num_samples)
                p = self.dec(images, q, num_samples=self.num_samples)
                batch_elbo = self.elbo(q, p)
                if self.cuda:
                    batch_elbo = batch_elbo.cpu()
                epoch_elbo += batch_elbo.data.numpy()[0]
                if infer:
                    log_p = p.log_joint(0, 1)
                    log_q = q.log_joint(0, 1)
                    log_w = log_p - log_q
                    w = torch.nn.functional.softmax(log_w, 0)
                    y_samples = q['digits'].value
                    y_expect = (w.unsqueeze(-1) * y_samples).sum(0)
                    _, y_pred = y_expect.data.max(-1)
                    if self.cuda:
                        y_pred = y_pred.cpu()
                    epoch_correct += (labels == y_pred).sum()
                else:
                    _, y_pred = q['digits'].value.data.max(-1)
                    if self.cuda:
                        y_pred = y_pred.cpu()
                    epoch_correct += (labels == y_pred).sum() * 1.0 / (self.num_samples or 1.0)
        return epoch_elbo / N, epoch_correct / N

    def save(self, model_dir, file_prefix, file_middle="", **kwargs):
        from pathlib import Path
        try:
            Path(model_dir).mkdir(mode=0o755, parents=True, exist_ok=True)
        except OSError as e:
            raise
        if file_middle != "":
            file_path = Path(model_dir, f"{file_prefix}_{file_middle}.{file_suffix}")
        else:
            file_path = Path(model_dir, f"{file_prefix}.{file_suffix}")
        log.info(f"saving the model to {file_path}")
        state = kwargs
        state.update({"encoder": self.enc.state_dict(),
                      "decoder": self.dec.state_dict(),
                      "batch_size": self.batch_size,
                      "num_samples": self.num_samples,
                      "label_fraction": self.label_fraction,
                      "cuda": self.cuda,
                      "lr": self.lr,
                      })
        torch.save(state, file_path)

    def load(self, model_dir, file_prefix):
        from pathlib import Path
        file_path = Path(model_dir, f"{file_prefix}.{file_suffix}")
        if not file_path.exists():
            log.error(f"no such file {file_path} exists")
            raise IOError
        log.info(f"loading the model from {file_path}")
        parameters = torch.load(file_path)
        self.enc.load_state_dict(parameters["encoder"])
        self.dec.load_state_dict(parameters["decoder"])
        self.batch_size = parameters["batch_size"]
        self.num_samples = parameters["num_samples"]
        self.label_fraction = parameters["label_fraction"]
        self.cuda = parameters["cuda"]
        self.lr = parameters["lr"]
        self.initialize()
        return parameters


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='SS-VAE example')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size for training')
    parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate')
    parser.add_argument('--num_samples', default=8, type=int, help='number of samples (?)')
    parser.add_argument('--label_fraction', default=0.1, type=float, help='fraction for the labeled data')
    parser.add_argument('--cuda', dest='cuda', action='store_true', help='use cuda')

    args = parser.parse_args()

    model = Network(args)
