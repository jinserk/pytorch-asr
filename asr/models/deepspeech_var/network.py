#!python
import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#from torch.nn._functions.thnn import rnnFusedPointwise as fusedBackend
from torch.autograd import Variable

from asr.utils import params as p
from asr.utils.misc import Swish, InferenceBatchSoftmax


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


'''
class LSTMCell(nn.Module):
    """A basic LSTM cell."""

    def __init__(self, input_size, hidden_size, bias=True, use_layernorm=False):
        """
        Most parts are copied from torch.nn.LSTMCell.
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_layernorm = use_layernorm
        self.use_bias = bias
        if self.use_layernorm:
            self.use_bias = False
        #print("LSTMCell: use_layernorm=%s" % use_layernorm)
        self.weight_ih = nn.Parameter(torch.FloatTensor(input_size, 4 * hidden_size))
        self.weight_hh = nn.Parameter(torch.FloatTensor(hidden_size, 4 * hidden_size))
        if self.use_layernorm:
            self.ln_ih = nn.LayerNorm(4 * hidden_size)
            self.ln_hh = nn.LayerNorm(4 * hidden_size)
        if self.use_bias:
            self.bias_ih = Parameter(torch.FloatTensor(4 * hidden_size))
            self.bias_hh = Parameter(torch.FloatTensor(4 * hidden_size))
        self.state = fusedBackend.LSTMFused.apply
        self.init_weights()

    def init_weights(self):
        """
        Initialize parameters following the way proposed in the paper.
        """
        stdv = 1.0 / np.sqrt(self.hidden_size)
        self.weight_ih.data.uniform_(-stdv, stdv)
        nn.init.orthogonal_(self.weight_hh.data)
        if self.use_bias:
            self.bias_ih.data.fill_(0)
            self.bias_hh.data.fill_(0)

    def forward(self, input_, hx):
        """
        Args:
            input_: A (batch, input_size) tensor containing input
                features.
            hx: A tuple (h_0, c_0), which contains the initial hidden
                and cell state, where the size of both states is
                (batch, hidden_size).
        Returns:
            h_1, c_1: Tensors containing the next hidden and cell state.
        """
        assert input_.is_cuda
        h_0, c_0 = hx
        igates = torch.mm(input_, self.weight_ih)
        hgates = torch.mm(h_0, self.weight_hh)
        if self.use_layernorm:
            igates = self.ln_ih(igates)
            hgates = self.ln_hh(hgates)
            return self.state(igates, hgates, c_0)
        elif self.use_bias:
            return self.state(igates, hgates, c_0, self.bias_ih, self.bias_hh)
        else:
            return self.state(igates, hgates, c_0)

    def __repr__(self):
        s = '{name}({input_size}, {hidden_size})'
        return s.format(name=self.__class__.__name__, **self.__dict__)
'''

class LSTMCell(nn.LSTMCell):

    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__(input_size, hidden_size, bias)

        self.ln_ih = nn.LayerNorm(4 * hidden_size)
        self.ln_hh = nn.LayerNorm(4 * hidden_size)
        self.ln_ho = nn.LayerNorm(hidden_size)

    def forward(self, input, hidden=None):
        self.check_forward_input(input)
        if hidden is None:
            hx = input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)
            cx = input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)
        else:
            hx, cx = hidden
        self.check_forward_hidden(input, hx, '[0]')
        self.check_forward_hidden(input, cx, '[1]')

        gates = self.ln_ih(F.linear(input, self.weight_ih, self.bias_ih)) \
                 + self.ln_hh(F.linear(hx, self.weight_hh, self.bias_hh))
        i, f, o = gates[:, :(3 * self.hidden_size)].sigmoid().chunk(3, 1)
        g = gates[:, (3 * self.hidden_size):].tanh()

        cy = (f * cx) + (i * g)
        hy = o * self.ln_ho(cy).tanh()
        return hy, cy


class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, use_layernorm=False, bidirectional=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        num_directions = 2 if bidirectional else 1
        self.hidden0 = nn.ModuleList([
            LSTMCell(input_size=(input_size if layer == 0 else hidden_size * num_directions),
                     hidden_size=hidden_size, bias=bias, use_layernorm=use_layernorm)
            for layer in range(num_layers)
        ])

        if self.bidirectional:
            self.hidden1 = nn.ModuleList([
                LSTMCell(input_size=(input_size if layer == 0 else hidden_size * num_directions),
                         hidden_size=hidden_size, bias=bias, use_layernorm=use_layernorm)
                for layer in range(num_layers)
            ])

    def forward(self, input, hidden=None):
        seq_len, batch_size, hidden_size = input.size()  # supports TxNxH only
        num_directions = 2 if self.bidirectional else 1
        if hidden is None:
            hx = input.new_zeros(self.num_layers * num_directions, batch_size, self.hidden_size, requires_grad=False)
            cx = input.new_zeros(self.num_layers * num_directions, batch_size, self.hidden_size, requires_grad=False)
        else:
            hx, cx = hidden

        ht = [[None, ] * (self.num_layers * num_directions)] * seq_len
        ct = [[None, ] * (self.num_layers * num_directions)] * seq_len

        if self.bidirectional:
            xs = input
            for l, (layer0, layer1) in enumerate(zip(self.hidden0, self.hidden1)):
                l0, l1 = 2 * l, 2 * l + 1
                h0, c0, h1, c1 = hx[l0], cx[l0], hx[l1], cx[l1]
                for t, (x0, x1) in enumerate(zip(xs, reversed(xs))):
                    ht[t][l0], ct[t][l0] = layer0(x0, (h0, c0))
                    h0, c0 = ht[t][l0], ct[t][l0]
                    t = seq_len - 1 - t
                    ht[t][l1], ct[t][l1] = layer1(x1, (h1, c1))
                    h1, c1 = ht[t][l1], ct[t][l1]
                xs = [torch.cat((h[l0], h[l1]), dim=1) for h in ht]
            y  = torch.stack(xs)
            hy = torch.stack(ht[-1])
            cy = torch.stack(ct[-1])
        else:
            h, c = hx, cx
            for t, x in enumerate(input):
                for l, layer in enumerate(self.hidden0):
                    ht[t][l], ct[t][l] = layer(x, (h[l], c[l]))
                    x = ht[t][l]
                h, c = ht[t], ct[t]
            y  = torch.stack([h[-1] for h in ht])
            hy = torch.stack(ht[-1])
            cy = torch.stack(ct[-1])

        return y, (hy, cy)


class BatchRNN(nn.Module):

    def __init__(self, input_size, hidden_size, rnn_type=nn.LSTM, bidirectional=False,
                 batch_first=True, layer_norm=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.batch_first = batch_first

        self.layer_norm = SequenceWise(nn.LayerNorm(input_size)) if layer_norm else None

        self.rnn = rnn_type(input_size=input_size, hidden_size=hidden_size,
                            bidirectional=bidirectional, batch_first=batch_first, bias=True)

    def flatten_parameters(self):
        self.rnn.flatten_parameters()

    def forward(self, x, seq_lens):
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        ps = nn.utils.rnn.pack_padded_sequence(x, seq_lens.tolist(), batch_first=self.batch_first)
        ps, _ = self.rnn(ps)
        x, _ = nn.utils.rnn.pad_packed_sequence(ps, batch_first=self.batch_first)
        if self.bidirectional:
            # (NxTxH*2) -> (NxTxH) by sum
            x = x.view(x.size(0), x.size(1), 2, -1).sum(2).view(x.size(0), x.size(1), -1)
        return x


class DeepSpeech(nn.Module):

    def __init__(self, num_classes=p.NUM_CTC_LABELS, input_folding=3, rnn_type=nn.LSTM,
                 rnn_hidden_size=512, rnn_num_layers=4, bidirectional=True, context=20):
        super().__init__()

        # model metadata needed for serialization/deserialization
        self._rnn_type = rnn_type
        self._hidden_size = rnn_hidden_size
        self._hidden_layers = rnn_num_layers
        self._bidirectional = bidirectional

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
        #W5 = 2 * rnn_hidden_size if bidirectional else rnn_hidden_size
        H1 = rnn_hidden_size

        self.conv = nn.Sequential(
            nn.Conv2d(C0, C1, kernel_size=(41, 7), stride=(2, 1), padding=(20, 3)),
            nn.BatchNorm2d(C1),
            nn.Hardtanh(-10, 10, inplace=True),
            #nn.ReLU(inplace=True),
            #Swish(inplace=True),
            #nn.MaxPool2d(kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.Conv2d(C1, C2, kernel_size=(21, 7), stride=(2, 1), padding=(10, 3)),
            nn.BatchNorm2d(C2),
            nn.Hardtanh(-10, 10, inplace=True),
            #nn.ReLU(inplace=True),
            #Swish(inplace=True),
            #nn.MaxPool2d(kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.Conv2d(C2, C3, kernel_size=(11, 7), stride=(2, 1), padding=(5, 3)),
            nn.BatchNorm2d(C3),
            nn.Hardtanh(-10, 10, inplace=True),
            #nn.ReLU(inplace=True),
            #Swish(inplace=True),
            #nn.MaxPool2d(kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
        )

        self.fc1 = SequenceWise(nn.Sequential(
            nn.Linear(H0, rnn_hidden_size, bias=True),
            nn.Dropout(0.2, inplace=True),
            nn.Hardtanh(-10, 10, inplace=True),
            #nn.ReLU(inplace=True),
            #Swish(inplace=True),
        ))

        # using BatchRNN
        self.rnns = nn.ModuleList([
            BatchRNN(input_size=rnn_hidden_size, hidden_size=rnn_hidden_size,
                     rnn_type=rnn_type, bidirectional=bidirectional, layer_norm=True)
            for _ in range(rnn_num_layers)
        ])

        # using multi-layered nn.LSTM
        #self.rnns = nn.LSTM(input_size=W4, hidden_size=rnn_hidden_size, num_layers=rnn_num_layers,
        #                    bidirectional=bidirectional, dropout=0)

        # using multi-layered LayerNorm LSTM
        #self.rnns = LSTM(input_size=W4, hidden_size=rnn_hidden_size, num_layers=4,
        #                 use_layernorm=True, bidirectional=bidirectional)

        self.fc2 = SequenceWise(nn.Sequential(
            nn.BatchNorm1d(H1),
            nn.Linear(H1, num_classes, bias=False),
            nn.Dropout(0.2, inplace=True),
            nn.Hardtanh(-10, 10, inplace=True),
            #nn.Tanh(),
        ))
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, seq_lens):
        x = self.conv(x)
        x = x.view(-1, x.size(1) * x.size(2), x.size(3))  # Collapse feature dimension
        x = x.transpose(1, 2).contiguous()  # NxTxH
        h = self.fc1(x)
        x = self.rnns[0](h, seq_lens)
        for i in range(1, self._hidden_layers):
            x = x + h
            x = self.rnns[i](x, seq_lens)
        x = self.fc2(x)
        x = self.softmax(x)
        return x, seq_lens


if __name__ == '__main__':
    import os.path
    import argparse

    parser = argparse.ArgumentParser(description='DeepSpeech model information')
    parser.add_argument('--model_path', default='models/deepspeech_final.pth.tar',
                        help='Path to model file created by training')
    args = parser.parse_args()
    package = torch.load(args.model_path, map_location=lambda storage, loc: storage)
    model = DeepSpeech.load_model(args.model_path)

    print("Model name:         ", os.path.basename(args.model_path))
    print("DeepSpeech version: ", model._version)
    print("")
    print("Recurrent Neural Network Properties")
    print("  RNN Type:         ", model._rnn_type.__name__.lower())
    print("  RNN Layers:       ", model._hidden_layers)
    print("  RNN Size:         ", model._hidden_size)
    print("")
    print("Model Features")
    print("  Sample Rate:      ", model._audio_conf.get("sample_rate", "n/a"))
    print("  Window Type:      ", model._audio_conf.get("window", "n/a"))
    print("  Window Size:      ", model._audio_conf.get("window_size", "n/a"))
    print("  Window Stride:    ", model._audio_conf.get("window_stride", "n/a"))

    if package.get('loss_results', None) is not None:
        print("")
        print("Training Information")
        epochs = package['epoch']
        print("  Epochs:           ", epochs)
        print("  Current Loss:      {0:.3f}".format(package['loss_results'][epochs - 1]))
        print("  Current CER:       {0:.3f}".format(package['cer_results'][epochs - 1]))
        print("  Current WER:       {0:.3f}".format(package['wer_results'][epochs - 1]))

    if package.get('meta', None) is not None:
        print("")
        print("Additional Metadata")
        for k, v in model._meta:
            print("  ", k, ": ", v)

