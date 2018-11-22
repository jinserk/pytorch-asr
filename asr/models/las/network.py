#!python
import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from asr.utils import params as p
from asr.utils.misc import onehot2int, int2onehot, Swish, InferenceBatchSoftmax, register_nan_checks


class SequenceWise(nn.Module):

    def __init__(self, module):
        """
        Collapses input of dim T*N*H to (T*N)*H, and applies to a module.
        Allows handling of variable sequence lengths and minibatch sizes.
        :param module: Module to apply input to.
        """
        super().__init__()
        self.module = module

    def forward(self, x, *args, **kwargs):
        t, n = x.size(0), x.size(1)
        x = x.contiguous().view(t * n, -1)
        x = self.module(x, *args, **kwargs)
        x = x.contiguous().view(t, n, -1)
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr


class Listener(nn.Module):

    def __init__(self, listen_vec_size, input_folding=3, rnn_type=nn.LSTM,
                 rnn_hidden_size=256, rnn_num_layers=4, bidirectional=True, last_fc=False):
        super().__init__()

        self.rnn_num_layers = rnn_num_layers
        self.bidirectional = bidirectional

        # Based on above convolutions and spectrogram size using conv formula (W - F + 2P)/ S+1
        W0 = 129
        C0 = 2 * input_folding
        W1 = (W0 - 3 + 2*1) // 2 + 1  # 65
        C1 = 64
        W2 = (W1 - 3 + 2*1) // 2 + 1  # 33
        C2 = C1 * 2
        W3 = (W2 - 3 + 2*1) // 2 + 1  # 17
        C3 = C2 * 2
        H0 = C3 * W3

        self.feature = nn.Sequential(OrderedDict([
            ('cv1', nn.Conv2d(C0, C1, kernel_size=(11, 3), stride=(1, 1), padding=(5, 1), bias=True)),
            ('nl1', nn.LeakyReLU()),
            ('mp1', nn.AvgPool2d(kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))),
            ('bn1', nn.BatchNorm2d(C1)),
            ('cv2', nn.Conv2d(C1, C2, kernel_size=(11, 3), stride=(1, 1), padding=(5, 1), bias=True)),
            ('nl2', nn.LeakyReLU()),
            ('mp2', nn.AvgPool2d(kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))),
            ('bn2', nn.BatchNorm2d(C2)),
            ('cv3', nn.Conv2d(C2, C3, kernel_size=(11, 3), stride=(1, 1), padding=(5, 1), bias=True)),
            ('nl3', nn.LeakyReLU()),
            ('mp3', nn.AvgPool2d(kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))),
            ('bn3', nn.BatchNorm2d(C3)),
        ]))

        # using multi-layered nn.LSTM
        self.batch_first = True
        self.rnns = rnn_type(input_size=H0, hidden_size=rnn_hidden_size, num_layers=rnn_num_layers,
                             bias=True, bidirectional=bidirectional, batch_first=self.batch_first)

        if last_fc:
            self.fc = SequenceWise(nn.Sequential(OrderedDict([
                ('ln1', nn.LayerNorm(rnn_hidden_size, elementwise_affine=False)),
                ('fc1', nn.Linear(rnn_hidden_size, listen_vec_size, bias=False)),
            ])))
        else:
            assert listen_vec_size == rnn_hidden_size
            self.fc = None

    def forward(self, x, seq_lens):
        h = self.feature(x)
        h = h.view(-1, h.size(1) * h.size(2), h.size(3))  # Collapse feature dimension
        y = h.transpose(1, 2).contiguous()  # NxTxH

        ps = nn.utils.rnn.pack_padded_sequence(y, seq_lens.tolist(), batch_first=self.batch_first)
        ps, _ = self.rnns(ps)
        y, _ = nn.utils.rnn.pad_packed_sequence(ps, batch_first=self.batch_first)

        if self.bidirectional:
            y = y.view(y.size(0), y.size(1), 2, -1).sum(2).view(y.size(0), y.size(1), -1)
        if self.fc is not None:
            y = self.fc(y)

        return y


class MaskedSoftmax(nn.Module):

    def __init__(self, dim=-1, epsilon=1e-5):
        super().__init__()
        self.dim = dim
        self.epsilon = epsilon

        self.softmax = nn.Softmax(dim=dim)

    def forward(self, e, mask=None):
        # e: Bx1xTh, mask: BxTh
        if mask is None:
            return self.softmax(e)
        else:
            # masked softmax only in input_seq_len in batch
            # for stability, refered to https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
            shift_e = e - e.max()
            exps = torch.exp(shift_e) * mask
            sums = exps.sum(dim=self.dim, keepdim=True) + self.epsilon
            return (exps / sums)


class Attention(nn.Module):

    def __init__(self, state_vec_size, listen_vec_size, apply_proj=True, proj_hidden_size=256, num_heads=1):
        super().__init__()
        self.apply_proj = apply_proj
        self.num_heads = num_heads

        if apply_proj:
            self.phi = nn.Linear(state_vec_size, proj_hidden_size * num_heads, bias=True)
            # psi should have no bias since h was padded with zero
            self.psi = nn.Linear(listen_vec_size, proj_hidden_size, bias=False)
        else:
            assert state_vec_size == listen_vec_size * num_heads

        if num_heads > 1:
            input_size = listen_vec_size * num_heads
            self.reduce = nn.Linear(input_size, listen_vec_size, bias=True)

        self.normal = SequenceWise(MaskedSoftmax(dim=-1))

    def score(self, m, n):
        """ dot product as score function """
        return torch.bmm(m, n.transpose(1, 2))

    def forward(self, s, h, len_mask=None):
        # s: Bx1xHs -> m: Bx1xHe
        # h: BxThxHh -> n: BxThxHe
        if self.apply_proj:
            m = self.phi(s)
            n = self.psi(h)
        else:
            m = s
            n = h

        # <m, n> -> a, e: Bx1xTh -> c: Bx1xHh
        if self.num_heads > 1:
            proj_hidden_size = m.size(-1) // self.num_heads
            ee = [self.score(mi, n) for mi in torch.split(m, proj_hidden_size, dim=-1)]
            aa = [self.normal(e, len_mask) for e in ee]
            c = self.reduce(torch.cat([torch.bmm(a, h) for a in aa], dim=-1))
            a = torch.stack(aa).transpose(0, 1)
        else:
            e = self.score(m, n)
            a = self.normal(e, len_mask)
            c = torch.bmm(a, h)
            a = a.unsqueeze(dim=1)
        # c: context (Bx1xHh), a: Bxheadsx1xTh
        return c, a


class Speller(nn.Module):

    def __init__(self, listen_vec_size, label_vec_size, max_seq_lens=256, sos=None, eos=None,
                 rnn_type=nn.LSTM, rnn_hidden_size=512, rnn_num_layers=1,
                 apply_attend_proj=False, proj_hidden_size=256, num_attend_heads=1,
                 masked_attend=True):
        super().__init__()

        self.label_vec_size = label_vec_size
        assert sos is not None and 0 <= sos < label_vec_size
        assert eos is not None and 0 <= eos < label_vec_size
        assert sos is not None and eos is not None and sos != eos
        self.sos = label_vec_size - 2 if sos is None else sos
        self.eos = label_vec_size - 1 if eos is None else eos
        self.max_seq_lens = max_seq_lens

        Hs, Hc, Hy = rnn_hidden_size, listen_vec_size, label_vec_size

        self.rnn_num_layers = rnn_num_layers
        self.rnns = rnn_type(input_size=(Hy + Hc), hidden_size=Hs, num_layers=rnn_num_layers,
                             bias=True, bidirectional=False, batch_first=True)

        self.attention = Attention(state_vec_size=Hs, listen_vec_size=Hc,
                                   apply_proj=apply_attend_proj, proj_hidden_size=proj_hidden_size,
                                   num_heads=num_attend_heads)

        self.masked_attend = masked_attend

        self.chardist = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(Hs + Hc, 128, bias=True)),
            ('fc2', nn.Linear(128, label_vec_size, bias=False)),
        ]))

        self.softmax = nn.Softmax(dim=-1)

    def get_mask(self, h, seq_lens):
        bs, ts, hs = h.size()
        mask = h.new_ones((bs, ts), dtype=torch.float)
        for b in range(bs):
            mask[b, seq_lens[b]:] = 0.
        return mask

    def forward(self, h, x_seq_lens, y=None, y_seq_lens=None):
        batch_size = h.size(0)
        sos = int2onehot(h.new_full((batch_size, 1), self.sos), num_classes=self.label_vec_size).float()
        eos = int2onehot(h.new_full((batch_size, 1), self.eos), num_classes=self.label_vec_size).float()

        hidden = None
        y_hats = list()
        attentions = list()

        in_mask = self.get_mask(h, x_seq_lens) if self.masked_attend else None
        x = torch.cat([sos, h.narrow(1, 0, 1)], dim=-1)

        y_hats_seq_lens = torch.ones((batch_size, ), dtype=torch.int) * self.max_seq_lens

        for t in range(self.max_seq_lens):
            x, hidden = self.rnns(x, hidden)
            c, a = self.attention(x, h, in_mask)
            y_hat = self.chardist(torch.cat([x, c], dim=-1))
            y_hat = self.softmax(y_hat)

            y_hats.append(y_hat)
            attentions.append(a)

            # check eos occurrence
            bi = onehot2int(y_hat.squeeze()).eq(self.eos)
            ri = y_hats_seq_lens.gt(t)
            if bi.is_cuda:
                ri = ri.cuda()
            y_hats_seq_lens[bi * ri] = t + 1

            # early termination
            if y_hats_seq_lens.le(t + 1).all():
                break

            if y is None:
                x = torch.cat([y_hat, c], dim=-1)
            elif t < y.size(1):  # teach force
                x = torch.cat([y.narrow(1, t, 1), c], dim=-1)
            else:
                x = torch.cat([eos, c], dim=-1)

        y_hats = torch.cat(y_hats, dim=1)
        attentions = torch.cat(attentions, dim=2)

        return y_hats, y_hats_seq_lens, attentions


class TFRScheduler(object):

    def __init__(self, model, ranges=(0.9, 0.1), warm_up=5, epochs=32):
        self.model = model

        self.upper, self.lower = ranges
        assert 0. < self.lower < self.upper < 1.
        self.warm_up = warm_up
        self.end_epochs = epochs + warm_up
        self.slope = (self.lower - self.upper) / epochs

        self.last_epoch = -1

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'model'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def get_tfr(self):
        # linearly declined
        if self.last_epoch < self.warm_up:
            return self.upper
        elif self.last_epoch < self.end_epochs:
            return self.upper + self.slope * (self.last_epoch - self.warm_up)
        else:
            return self.lower

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        self.model.tfr = self.get_tfr()


class LogWithLabelSmoothing(nn.Module):

    def __init__(self, floor=0.01):
        super().__init__()
        self.floor = floor

    def forward(self, x):
        y = (1.0 - self.floor) * x + self.floor / x.size(-1)
        return y.log()


class ListenAttendSpell(nn.Module):

    def __init__(self, label_vec_size=p.NUM_CTC_LABELS, listen_vec_size=256,
                 state_vec_size=256, num_attend_heads=1, input_folding=2, smoothing=0.001):
        super().__init__()

        self.label_vec_size = label_vec_size + 2  # to add <sos>, <eos>
        self.blk = 0
        self.sos = self.label_vec_size - 2
        self.eos = self.label_vec_size - 1

        self.num_heads = num_attend_heads
        self.tfr = 1.

        self.listen = Listener(listen_vec_size=listen_vec_size, input_folding=input_folding, rnn_type=nn.LSTM,
                               rnn_hidden_size=listen_vec_size, rnn_num_layers=4, bidirectional=True,
                               last_fc=False)

        self.spell = Speller(listen_vec_size=listen_vec_size, label_vec_size=self.label_vec_size,
                             sos=self.sos, eos=self.eos, max_seq_lens=256,
                             rnn_hidden_size=state_vec_size, rnn_num_layers=2,
                             apply_attend_proj=True, proj_hidden_size=128, num_attend_heads=num_attend_heads)

        self.attentions = None
        self.log = LogWithLabelSmoothing(floor=smoothing)

    def _is_teacher_force(self):
        return np.random.random_sample() < self.tfr

    def forward(self, x, x_seq_lens, y=None, y_seq_lens=None):
        if self.training:
            assert y is not None and y_seq_lens is not None
            return self.train_forward(x, x_seq_lens, y, y_seq_lens)
        else:
            return self.eval_forward(x, x_seq_lens)

    def train_forward(self, x, x_seq_lens, y, y_seq_lens):
        # to remove the case of x_seq_lens < y_seq_lens and y_seq_lens > max_seq_lens
        bi = x_seq_lens.ge(y_seq_lens) * y_seq_lens.le(self.spell.max_seq_lens)
        x, x_seq_lens = x[bi], x_seq_lens[bi]

        # listen
        h = self.listen(x, x_seq_lens)

        # make ys from y including trailing eos
        eos_t = y.new_full((1, ), self.eos)
        ys = [torch.cat((yb, eos_t)) for yb in torch.split(y, y_seq_lens.tolist())]
        ys = nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=self.eos)
        ys, ys_seq_lens = ys[bi], y_seq_lens[bi] + 1

        # speller with teach force rate
        if self._is_teacher_force():
            floor = np.random.random_sample() * 1e-2
            yss = int2onehot(ys, num_classes=self.label_vec_size, floor=floor).float()
            noise = torch.rand_like(yss) * 0.1
            yss = yss * noise
            y_hats, y_hats_seq_lens, self.attentions = self.spell(h, x_seq_lens, yss, ys_seq_lens)
        else:
            y_hats, y_hats_seq_lens, self.attentions = self.spell(h, x_seq_lens)

        # match seq lens between y_hats and ys
        s1, s2 = y_hats.size(1), ys.size(1)
        if s1 < s2:
            dummy = y_hats.new_full((y_hats.size(0), s2 - s1, ), fill_value=self.blk, dtype=torch.int)
            dummy = int2onehot(dummy, num_classes=self.label_vec_size).float()
            y_hats = torch.cat([y_hats, dummy], dim=1)
            #y_hats = F.pad(y_hats, (0, 0, 0, s2 - s1))
        elif s1 > s2:
            ys = F.pad(ys, (0, s1 - s2), value=self.blk)

        y_hats = self.log(y_hats)
        return y_hats, y_hats_seq_lens, ys

    def eval_forward(self, x, x_seq_lens):
        # listen
        h = self.listen(x, x_seq_lens)
        # spell
        y_hats, y_hats_seq_lens, _ = self.spell(h, x_seq_lens)

        # return with seq lens without sos and eos
        y_hats = self.log(y_hats[:, :, :-2])
        return y_hats, y_hats_seq_lens


if __name__ == '__main__':
    pass
