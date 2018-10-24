#!python
import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from asr.utils import params as p
from asr.utils.misc import onehot2int, int2onehot, Swish, InferenceBatchSoftmax


class SequenceWise(nn.Module):

    def __init__(self, module):
        """
        Collapses input of dim T*N*H to (T*N)*H, and applies to a module.
        Allows handling of variable sequence lengths and minibatch sizes.
        :param module: Module to apply input to.
        """
        super().__init__()
        self.module = module

    def forward(self, x):
        t, n = x.size(0), x.size(1)
        x = x.view(t * n, -1)
        x = self.module(x)
        x = x.view(t, n, -1)
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr


class BatchRNN(nn.Module):

    def __init__(self, input_size, hidden_size, rnn_type=nn.LSTM, bidirectional=False,
                 batch_first=True, layer_norm=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.batch_first = batch_first

        self.layer_norm = SequenceWise(nn.LayerNorm(input_size, elementwise_affine=False)) if layer_norm else None

        self.rnn = rnn_type(input_size=input_size, hidden_size=hidden_size,
                            bidirectional=bidirectional, batch_first=batch_first, bias=True)

    def flatten_parameters(self):
        self.rnn.flatten_parameters()

    def forward(self, x, seq_lens, hidden=None):
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        ps = nn.utils.rnn.pack_padded_sequence(x, seq_lens.tolist(), batch_first=self.batch_first)
        ps, hidden = self.rnn(ps, hidden)
        y, _ = nn.utils.rnn.pad_packed_sequence(ps, batch_first=self.batch_first)
        if self.bidirectional:
            # (NxTxH*2) -> (NxTxH) by sum
            y = y.view(y.size(0), y.size(1), 2, -1).sum(2).view(y.size(0), y.size(1), -1)
        return y, hidden


class Listener(nn.Module):

    def __init__(self, listen_vec_size, input_folding=2, rnn_type=nn.LSTM,
                 rnn_hidden_size=512, rnn_num_layers=4, bidirectional=True, skip_fc=False):
        super().__init__()

        self.rnn_num_layers = rnn_num_layers
        self.skip_fc = skip_fc

        # Based on above convolutions and spectrogram size using conv formula (W - F + 2P)/ S+1
        W0 = 129
        C0 = 2 * input_folding
        W1 = (W0 - 41 + 2*20) // 2 + 1  # 65
        C1 = 16
        W2 = (W1 - 21 + 2*10) // 2 + 1  # 33
        C2 = C1 * 2
        W3 = (W2 - 11 + 2*5) // 2 + 1   # 17
        C3 = C2 * 2

        H0 = C3 * W3
        H1 = rnn_hidden_size

        self.conv = nn.Sequential(
            nn.Conv2d(C0, C1, kernel_size=(41, 7), stride=(1, 1), padding=(20, 3)),
            nn.BatchNorm2d(C1),
            #nn.Hardtanh(-10, 10, inplace=True),
            nn.LeakyReLU(inplace=True),
            #Swish(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.Conv2d(C1, C2, kernel_size=(21, 7), stride=(1, 1), padding=(10, 3)),
            nn.BatchNorm2d(C2),
            #nn.Hardtanh(-10, 10, inplace=True),
            nn.LeakyReLU(inplace=True),
            #Swish(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.Conv2d(C2, C3, kernel_size=(11, 7), stride=(1, 1), padding=(5, 3)),
            nn.BatchNorm2d(C3),
            #nn.Hardtanh(-10, 10, inplace=True),
            nn.LeakyReLU(inplace=True),
            #Swish(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
        )

        self.fc1 = SequenceWise(nn.Sequential(
            nn.Linear(H0, H1, bias=True),
            nn.Dropout(0.5, inplace=True),
            #nn.Hardtanh(-10, 10, inplace=True),
            nn.LeakyReLU(inplace=True),
            #Swish(inplace=True),
        ))

        # using BatchRNN
        self.rnns = nn.ModuleList([
            BatchRNN(input_size=H1 if l == 0 else rnn_hidden_size, hidden_size=rnn_hidden_size,
                     rnn_type=rnn_type, bidirectional=bidirectional, layer_norm=True)
            for l in range(rnn_num_layers)
        ])

        if not skip_fc:
            self.fc = SequenceWise(nn.Sequential(
                nn.LayerNorm(H1, elementwise_affine=False),
                nn.Linear(H1, listen_vec_size, bias=True),
                nn.Dropout(0.5, inplace=True),
            ))
        else:
            assert listen_vec_size == rnn_hidden_size

    def forward(self, x, seq_lens):
        h = self.conv(x)
        h = h.view(-1, h.size(1) * h.size(2), h.size(3))  # Collapse feature dimension
        h = h.transpose(1, 2).contiguous()  # NxTxH
        h = self.fc1(h)
        g, _ = self.rnns[0](h, seq_lens)
        for i in range(1, self.rnn_num_layers):
            g = g + h
            g, _ = self.rnns[i](g, seq_lens)
        if self.skip_fc:
            return g, seq_lens
        else:
            g = g + h
            y = self.fc(g)
            return y, seq_lens


class Attention(nn.Module):

    def __init__(self, state_vec_size, listen_vec_size, apply_proj=True, proj_hidden_size=512, num_heads=1):
        super().__init__()
        self.apply_proj = apply_proj
        self.num_heads = num_heads

        if apply_proj:
            self.phi = SequenceWise(nn.Sequential(
                nn.BatchNorm1d(state_vec_size),
                nn.Linear(state_vec_size, proj_hidden_size * num_heads, bias=False)
            ))
            self.psi = SequenceWise(nn.Sequential(
                nn.BatchNorm1d(state_vec_size),
                nn.Linear(listen_vec_size, proj_hidden_size, bias=False)
            ))
        else:
            assert state_vec_size == listen_vec_size * num_heads

        self.normal = nn.Softmax(dim=-1)

        if num_heads > 1:
            self.reduce = SequenceWise(nn.Sequential(
                nn.BatchNorm1d(listen_vec_size * num_heads),
                nn.Linear(listen_vec_size * num_heads, listen_vec_size, bias=False)
            ))

    def score(self, m, n):
        """ dot product as score function """
        return torch.bmm(m, n.transpose(1, 2))

    def forward(self, s, h):
        # s: Bx1xHs -> ms: Bx1xHe
        # h: BxThxHh -> mh: BxThxHe
        if self.apply_proj:
            ms = self.phi(s)
            mh = self.psi(h)
        else:
            ms = s
            mh = h

        # <ms, mh> -> e: Bx1xTh -> c: Bx1xHh
        if self.num_heads > 1:
            proj_hidden_size = ms.size(-1) // self.num_heads
            ee = [self.score(msi, mh) for msi in torch.split(ms, proj_hidden_size, dim=-1)]
            aa = [self.normal(e) for e in ee]
            c = self.reduce(torch.cat([torch.bmm(a, h) for a in aa], dim=-1))
            return torch.stack(aa), c
        else:
            e = self.score(ms, mh)
            a = self.normal(e)
            c = torch.bmm(a, h)
            return a.unsqueeze(dim=0), c


class Speller(nn.Module):

    def __init__(self, listen_vec_size, label_vec_size, rnn_type=nn.LSTM,
                 rnn_hidden_size=512, rnn_num_layers=1, max_seq_len=100,
                 apply_attend_proj=False, num_attend_heads=1):
        super().__init__()

        self.max_seq_len = max_seq_len
        self.label_vec_size = label_vec_size
        self.sos = label_vec_size - 2
        self.eos = label_vec_size - 1

        Hs, Hc, Hy = rnn_hidden_size, listen_vec_size, label_vec_size

        self.rnn_num_layers = rnn_num_layers
        self.rnns = nn.ModuleList([
            BatchRNN(input_size=((Hy + Hc) if l == 0 else Hs), hidden_size=Hs,
                     rnn_type=rnn_type, bidirectional=False, layer_norm=True)
            for l in range(rnn_num_layers)
        ])

        self.attention = Attention(state_vec_size=Hs, listen_vec_size=Hc,
                                   apply_proj=apply_attend_proj, num_heads=num_attend_heads)

        self.chardist = SequenceWise(nn.Sequential(
            nn.Linear(Hs + Hc, label_vec_size),
            InferenceBatchSoftmax()
        ))

    def forward(self, h, y=None, teacher_force_rate=0.):
        teacher_force = True if y is not None and np.random.random_sample() < teacher_force_rate else False
        batch_size = h.size(0)
        sos = int2onehot(h.new_full((batch_size, 1), self.sos), num_classes=self.label_vec_size).float()
        x = torch.cat([sos, h.narrow(1, 0, 1)], dim=-1)

        hidden = [ None ] * self.rnn_num_layers
        y_hats = list()
        attentions = list()
        max_seq_len = self.max_seq_len if not teacher_force else y.size(1)
        unit_len = torch.ones((batch_size, ))
        for i in range(max_seq_len):
            for l, rnn in enumerate(self.rnns):
                x, hidden[l] = rnn(x, unit_len, hidden[l])
            a, c = self.attention(x, h)
            y_hat = self.chardist(torch.cat([x, c], dim=-1))

            y_hats.append(y_hat)
            attentions.append(a)

            # if eos occurs in all batch, stop iteration
            if not onehot2int(y_hat.squeeze()).ne(self.eos).nonzero().numel():
                break

            if teacher_force:
                x = torch.cat([y.narrow(1, i, 1), c], dim=-1)
            else:
                x = torch.cat([y_hat, c], dim=-1)

        y_hats = torch.cat(y_hats, dim=1)
        attentions = torch.stack(attentions)

        seq_lens = torch.full((batch_size,), max_seq_len, dtype=torch.int)
        for b, y_hat in enumerate(y_hats):
            idx = onehot2int(y_hat).eq(self.label_vec_size - 1).nonzero()
            if idx.numel():
                seq_lens[b] = idx[0][0]

        return y_hats, seq_lens, attentions


class ListenAttendSpell(nn.Module):

    def __init__(self, label_vec_size=p.NUM_CTC_LABELS, listen_vec_size=256,
                 state_vec_size=512, num_attend_heads=2, max_seq_len=200,
                 tf_rate_range=(0.9, 0.1), tf_total_steps=50, input_folding=2):
        super().__init__()

        self.label_vec_size = label_vec_size + 2  # to add <sos>, <eos>
        self.max_seq_len = max_seq_len
        self.tf_rate_range = tf_rate_range
        self.tf_rate_total_step = tf_total_steps
        self.tf_rate_step = 0
        self.tf_rate = tf_rate_range[0]

        self.listen = Listener(listen_vec_size=listen_vec_size, input_folding=input_folding, rnn_type=nn.LSTM,
                               rnn_hidden_size=listen_vec_size, rnn_num_layers=4, bidirectional=True,
                               skip_fc=True)

        self.spell = Speller(listen_vec_size=listen_vec_size, label_vec_size=self.label_vec_size,
                             rnn_hidden_size=state_vec_size, rnn_num_layers=1, max_seq_len=max_seq_len,
                             apply_attend_proj=False, num_attend_heads=num_attend_heads)

    def step_tf_rate(self):
        if self.tf_rate_step < self.tf_rate_total_step:
            upper, lower = self.tf_rate_range
            self.tf_rate = upper - (upper - lower) * self.tf_rate_step / self.tf_rate_total_step
        else:
            self.tf_rate = self.tf_rate_range[1]

    def forward(self, x, x_seq_lens, y=None, y_seq_lens=None):
        # listener
        h, x_seq_lens = self.listen(x, x_seq_lens)
        # speller
        if self.training:
            if y_seq_lens.ge(self.max_seq_len).nonzero().numel():
                return None, None, None
            # change y to one-hot tensors
            eos = self.label_vec_size - 1
            eos_tensor = torch.cuda.IntTensor([eos, ]) if y.is_cuda else torch.IntTensor([eos, ])
            ys = [torch.cat([yb, eos_tensor]) for yb in torch.split(y, y_seq_lens.tolist())]
            ys = nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=eos)
            yss = int2onehot(ys, num_classes=self.label_vec_size).float()
            # do spell
            y_hats, y_hats_seq_lens, _ = self.spell(h, yss, teacher_force_rate=self.tf_rate)
            # match seq lens between y_hats and ys
            s1, s2 = y_hats.size(1), ys.size(1)
            if s1 < s2:
                # append one-hot tensors of eos to y_hats
                dummy = y_hats.new_full((y_hats.size(0), s2 - s1, ), fill_value=eos)
                dummy = int2onehot(dummy, num_classes=self.label_vec_size).float()
                y_hats = torch.cat([y_hats, dummy], dim=1)
            elif s1 > s2:
                ys = F.pad(ys, (0, s1 - s2), value=eos)
            # return with seq lens
            return y_hats, y_hats_seq_lens, ys
        else:
            # do spell
            y_hats, y_hats_seq_lens, _ = self.spell(h)
            # return with seq lens
            return y_hats, y_hats_seq_lens, None


if __name__ == '__main__':
    pass
