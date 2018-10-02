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
                 bias=True, batch_first=True, batch_norm=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.batch_first = batch_first

        self.batch_norm = SequenceWise(nn.BatchNorm1d(input_size)) if batch_norm else None

        self.rnn = rnn_type(input_size=input_size, hidden_size=hidden_size,
                            bidirectional=bidirectional, batch_first=batch_first, bias=bias)

    def flatten_parameters(self):
        self.rnn.flatten_parameters()

    def forward(self, x, seq_lens):
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        if torch.is_tensor(seq_lens):
            seq_lens = seq_lens.tolist()
        ps = nn.utils.rnn.pack_padded_sequence(x, seq_lens, batch_first=self.batch_first)
        ps, _ = self.rnn(ps)
        x, _ = nn.utils.rnn.pad_packed_sequence(ps, batch_first=self.batch_first)
        if self.bidirectional:
            x = x.view(x.size(0), x.size(1), 2, -1).sum(2).view(x.size(0), x.size(1), -1)  # (NxTxH*2) -> (NxTxH) by sum
        return x


class Listener(nn.Module):

    def __init__(self, listen_vec_size, input_folding=3, rnn_type=nn.LSTM,
                 rnn_hidden_size=512, rnn_num_layers=[4], bidirectional=True, skip_fc=False):
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

        H0 = [C3 * W3, rnn_hidden_size, rnn_hidden_size]

        self.conv = nn.Sequential(
            nn.Conv2d(C0, C1, kernel_size=(41, 11), stride=(2, 1), padding=(20, 5)),
            nn.BatchNorm2d(C1),
            #nn.ReLU(inplace=True),
            Swish(inplace=True),
            nn.Conv2d(C1, C2, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(C2),
            #nn.ReLU(inplace=True),
            Swish(inplace=True),
            nn.Conv2d(C2, C3, kernel_size=(11, 11), stride=(2, 1), padding=(5, 5)),
            nn.BatchNorm2d(C3),
            #nn.ReLU(inplace=True),
            Swish(inplace=True),
        )

        # using BatchRNN
        self.rnns = nn.ModuleList()
        for g in range(len(rnn_num_layers)):
            for l in range(rnn_num_layers[g]):
                self.rnns.append(
                    BatchRNN(input_size=(H0[g] if l == 0 else rnn_hidden_size),
                             hidden_size=rnn_hidden_size, rnn_type=rnn_type,
                             bidirectional=bidirectional, batch_norm=True)
                )

        if not skip_fc:
            self.fc = SequenceWise(nn.Sequential(
                nn.BatchNorm1d(H1),
                nn.Linear(H1, listen_vec_size, bias=False)
            ))
        else:
            assert listen_vec_size == rnn_hidden_size

    def forward(self, x, seq_lens):
        x = self.conv(x)

        x = x.view(-1, x.size(1) * x.size(2), x.size(3))  # Collapse feature dimension
        x = x.transpose(1, 2).contiguous()  # NxTxH
        for i in range(self.rnn_num_layers[0]):
            x = self.rnns[i](x, seq_lens)
        for g in range(1, len(self.rnn_num_layers)):
            x = x[:, :((x.size(1) // 2) * 2), :].view(-1, x.size(1) // 2, 2, x.size(2))
            x = x.contiguous().view(-1, x.size(1) // 2, 2 * x.size(2))
            seq_lens.div_(2)
            for i in range(self.rnn_num_layers[g]):
                j = np.cumsum(self.rnn_num_layers)[g-1] + i
                x = self.rnns[j](x, seq_lens)

        if not self.skip_fc:
            x = self.fc(x)

        return x, seq_lens


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


class BatchRNNCell(nn.Module):

    def __init__(self, input_size, hidden_size, rnn_type=nn.LSTMCell, bias=True, batch_norm=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.batch_norm = SequenceWise(nn.BatchNorm1d(input_size)) if batch_norm else None

        self.rnn = rnn_type(input_size=input_size, hidden_size=hidden_size, bias=bias)

    def flatten_parameters(self):
        self.rnn.flatten_parameters()

    def forward(self, x, hidden):
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        x, hidden = self.rnn(x, hidden)
        return x


class SpellerRNNCell(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=1, bias=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnns = nn.ModuleList([
            BatchRNNCell(input_size=(input_size if l == 0 else hidden_size),
                         hidden_size=hidden_size, bias=bias)
            for l in range(num_layers)
        ])

    def forward(self, x, hidden=None):
        # x: Nx1xHx, hidden: (LxNxHr, LxNxHr)
        if hidden is None:
            hx = x.new_zeros(self.num_layers, x.size(0), self.hidden_size, requires_grad=False)
            cx = x.new_zeros(self.num_layers, x.size(0), self.hidden_size, requires_grad=False)
        else:
            hx, cx = hidden

        ht = [None, ] * self.num_layers
        ct = [None, ] * self.num_layers

        x = x.squeeze(dim=1)
        h, c = hx, cx
        for l, rnn in enumerate(self.rnns):
            ht[l], ct[l] = rnn(x, (h[l], c[l]))
            x = ht[l]

        return ht[-1].unsqueeze(dim=1), (torch.stack(ht), torch.stack(ct))


class Speller(nn.Module):

    def __init__(self, listen_vec_size, label_vec_size,
                 rnn_hidden_size=512, rnn_num_layers=1, max_seq_len=100,
                 apply_attend_proj=False, num_attend_heads=1):
        super().__init__()

        self.max_seq_len = max_seq_len
        self.label_vec_size = label_vec_size

        Hs, Hc, Hy = rnn_hidden_size, listen_vec_size, label_vec_size

        self.rnns = SpellerRNNCell(input_size=(Hy + Hc), hidden_size=Hs, num_layers=rnn_num_layers)
        #self.rnns = nn.ModuleList([
        #     BatchRNN(input_size=((Hy + Hc) if l == 0 else Hs), hidden_size=Hs, rnn_type=nn.LSTM,
        #              bidirectional=False, batch_norm=True)
        #     for l in range(rnn_num_layers)
        #])

        self.attention = Attention(state_vec_size=Hs, listen_vec_size=Hc,
                                   apply_proj=apply_attend_proj, num_heads=num_attend_heads)

        self.chardist = SequenceWise(nn.Sequential(
            nn.Linear(Hs + Hc, label_vec_size),
            InferenceBatchSoftmax()
        ))

    def forward(self, h, y=None, teacher_force_rate=0.):
        teacher_force = True if y is not None and np.random.random_sample() < teacher_force_rate else False
        batch_size = h.size(0)
        sos = int2onehot(h.new_full((batch_size, 1), self.label_vec_size - 2), num_classes=self.label_vec_size).float()
        x = torch.cat([sos, h.narrow(1, 0, 1)], dim=-1)

        hidden = None
        y_hats = list()
        attentions = list()

        max_seq_len = self.max_seq_len if not teacher_force else y.size(1)
        for i in range(max_seq_len):
            #s = x
            #for rnn in self.rnns:
            #    s = rnn(s, [1] * batch_size)
            s, hidden = self.rnns(x, hidden)
            a, c = self.attention(s, h)
            y_hat = self.chardist(torch.cat([s, c], dim=-1))

            y_hats.append(y_hat)
            attentions.append(a)

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
                 state_vec_size=512, num_attend_heads=2, max_seq_len=100,
                 tf_rate_range=(0.9, 0.1), tf_total_steps=50):
        super().__init__()

        self.label_vec_size = label_vec_size + 2  # to add <sos>, <eos>
        self.max_seq_len = max_seq_len
        self.tf_rate_range = tf_rate_range
        self.tf_rate_total_step = tf_total_steps
        self.tf_rate_step = 0
        self.tf_rate = tf_rate_range[0]

        self.listen = Listener(listen_vec_size=listen_vec_size, input_folding=3, rnn_type=nn.LSTM,
                               rnn_hidden_size=listen_vec_size, rnn_num_layers=[4], bidirectional=True,
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
        h, _ = self.listen(x, x_seq_lens)
        # speller
        if self.training:
            # change y to one-hot tensors
            ys = [yb for yb in torch.split(y, y_seq_lens.tolist())]
            ys = nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=(self.label_vec_size - 1))
            yss = int2onehot(ys, num_classes=self.label_vec_size).float()
            # do spell
            y_hats, y_hats_seq_lens, _ = self.spell(h, yss, teacher_force_rate=self.tf_rate)
            # match seq lens between y_hats and ys
            s1, s2 = y_hats.size(1), ys.size(1)
            if s1 < s2:
                y_hats = F.pad(y_hats.transpose(1, 2), (0, s2 - s1)).transpose(1, 2)
            elif s1 > s2:
                ys = F.pad(ys, (0, s1 - s2), value=(self.label_vec_size - 1))
            # return with seq lens
            return y_hats, y_hats_seq_lens, ys
        else:
            # do spell
            y_hats, y_hats_seq_lens, _ = self.spell(h)
            # return with seq lens
            return y_hats, y_hats_seq_lens, None


if __name__ == '__main__':
    pass
