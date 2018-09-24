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


class TemporalRowConvolution(nn.Module):

    def __init__(self, input_size, kernel_size, stride=1, padding=0, feat_first=False, bias=False):
        super().__init__()
        self.input_size = input_size
        self.kernel_size = _single(kernel_size)
        self.stride = _single(stride)
        self.padding = _single(padding)
        self.weight = nn.Parameter(torch.Tensor(input_size, 1, *kernal_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(input_size))
        else:
            self.register_parameter('bias', None)
        self.feat_first = feat_first
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.kernel_size * self.input_size)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input_):
        return nn._functions.thnn.auto.TemporalRowConvolution.apply(input_, kernel_size, stride, padding)

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

    def __init__(self, input_size, hidden_size, rnn_type=nn.LSTM, bidirectional=False, batch_norm=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.batch_norm = SequenceWise(nn.BatchNorm1d(input_size)) if batch_norm else None
        self.rnn = rnn_type(input_size=input_size, hidden_size=hidden_size,
                            bidirectional=bidirectional, bias=True)

    def flatten_parameters(self):
        self.rnn.flatten_parameters()

    def forward(self, x, seq_lens):
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        ps = nn.utils.rnn.pack_padded_sequence(x, seq_lens.tolist())
        ps, _ = self.rnn(ps)
        x, _ = nn.utils.rnn.pad_packed_sequence(ps)
        if self.bidirectional:
            x = x.view(x.size(0), x.size(1), 2, -1).sum(2).view(x.size(0), x.size(1), -1)  # (TxNxH*2) -> (TxNxH) by sum
        return x


class Lookahead(nn.Module):
    # Wang et al 2016 - Lookahead Convolution Layer for Unidirectional Recurrent Neural Networks
    # input shape - sequence, batch, feature - TxNxH
    # output shape - same as input

    def __init__(self, n_features, context):
        # should we handle batch_first=True?
        super().__init__()
        self.n_features = n_features
        self.weight = nn.Parameter(torch.Tensor(n_features, context + 1))
        assert context > 0
        self.context = context
        self.register_parameter('bias', None)
        self.init_parameters()

    def init_parameters(self):  # what's a better way initialiase this layer?
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input):
        seq_len = input.size(0)
        # pad the 0th dimension (T/sequence) with zeroes whose number = context
        # Once pytorch's padding functions have settled, should move to those.
        padding = torch.zeros(self.context, *(input.size()[1:])).type_as(input.data)
        x = torch.cat((input, Variable(padding)), 0)

        # add lookahead windows (with context+1 width) as a fourth dimension
        # for each seq-batch-feature combination
        x = [x[i:i + self.context + 1] for i in range(seq_len)]  # TxLxNxH - sequence, context, batch, feature
        x = torch.stack(x)
        x = x.permute(0, 2, 3, 1)  # TxNxHxL - sequence, batch, feature, context

        x = torch.mul(x, self.weight).sum(dim=3)
        return x

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'n_features=' + str(self.n_features) \
               + ', context=' + str(self.context) + ')'


class DeepSpeech(nn.Module):

    def __init__(self, num_classes=p.NUM_CTC_LABELS, input_folding=3, rnn_type=nn.LSTM,
                 rnn_hidden_size=512, rnn_num_layers=[4], bidirectional=True, context=20):
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
        W4 = (W3 - 5 + 2*2) // 2 + 1   # 9
        C4 = C3 * 2
        W5 = (W4 - 5 + 2*2) // 2 + 1   # 5
        C5 = C4 * 2

        H0 = [C5 * W5, rnn_hidden_size * 2, rnn_hidden_size * 2]
        #W5 = 2 * rnn_hidden_size if bidirectional else rnn_hidden_size
        H1 = rnn_hidden_size

        self.conv = nn.Sequential(
            nn.Conv2d(C0, C1, kernel_size=(41, 5), stride=(2, 1), padding=(20, 2)),
            nn.BatchNorm2d(C1),
            #nn.Hardtanh(0, 20, inplace=True),
            nn.ReLU(inplace=True),
            #Swish(inplace=True),
            #nn.MaxPool2d(kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.Conv2d(C1, C2, kernel_size=(21, 5), stride=(2, 1), padding=(10, 2)),
            nn.BatchNorm2d(C2),
            #nn.Hardtanh(0, 20, inplace=True)
            nn.ReLU(inplace=True),
            #Swish(inplace=True),
            #nn.MaxPool2d(kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.Conv2d(C2, C3, kernel_size=(11, 5), stride=(2, 1), padding=(5, 2)),
            nn.BatchNorm2d(C3),
            #nn.Hardtanh(0, 20, inplace=True)
            nn.ReLU(inplace=True),
            #Swish(inplace=True),
            #nn.MaxPool2d(kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.Conv2d(C3, C4, kernel_size=(5, 5), stride=(2, 1), padding=(2, 2)),
            nn.BatchNorm2d(C4),
            #nn.Hardtanh(0, 20, inplace=True)
            nn.ReLU(inplace=True),
            #Swish(inplace=True),
            #nn.MaxPool2d(kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.Conv2d(C4, C5, kernel_size=(5, 5), stride=(2, 1), padding=(2, 2)),
            nn.BatchNorm2d(C5),
            #nn.Hardtanh(0, 20, inplace=True)
            nn.ReLU(inplace=True),
            #Swish(inplace=True),
            #nn.MaxPool2d(kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
        )

        # using BatchRNN
        self.rnns = nn.ModuleList()
        for group in range(len(rnn_num_layers)):
            for layer in range(rnn_num_layers[group]):
                self.rnns.append(BatchRNN(input_size=(H0[group] if layer == 0 else rnn_hidden_size),
                                          hidden_size=rnn_hidden_size, rnn_type=rnn_type,
                                          bidirectional=bidirectional, batch_norm=True))

        # using multi-layered nn.LSTM
        #self.rnns = nn.LSTM(input_size=W4, hidden_size=rnn_hidden_size, num_layers=rnn_num_layers,
        #                    bidirectional=bidirectional, dropout=0)

        # using multi-layered LayerNorm LSTM
        #self.rnns = LSTM(input_size=W4, hidden_size=rnn_hidden_size, num_layers=4,
        #                 use_layernorm=True, bidirectional=bidirectional)

        self.lookahead = nn.Sequential(
            # consider adding batch norm?
            Lookahead(H1, context=context),
            #nn.Hardtanh(0, 20, inplace=True)
            nn.ReLU(inplace=True),
            #Swish(inplace=True)
        ) if not bidirectional else None

        fully_connected = nn.Sequential(
            nn.BatchNorm1d(H1),
            nn.Linear(H1, num_classes, bias=False)
        )
        self.fc = nn.Sequential(
            SequenceWise(fully_connected),
        )
        self.inference_softmax = InferenceBatchSoftmax()

    def forward(self, x, seq_lens):
        x = self.conv(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # Collapse feature dimension
        x = x.transpose(1, 2).transpose(0, 1).contiguous()  # TxNxH
        for i in range(self._hidden_layers[0]):
            x = self.rnns[i](x, seq_lens)
        for g in range(1, len(self._hidden_layers)):
            sizes = x.size()
            x = x[:((sizes[0] // 2) * 2), :, :].view(sizes[0] // 2, 2, sizes[1], sizes[2])
            x = x.transpose(1, 2).contiguous().view(sizes[0] // 2, sizes[1], 2 * sizes[2])
            seq_lens.div_(2)
            for i in range(self._hidden_layers[g]):
                j = np.cumsum(self._hidden_layers)[g-1] + i
                x = self.rnns[j](x, seq_lens)
        #x, _ = self.rnns(x)
        if not self._bidirectional:  # no need for lookahead layer in bidirectional
            x = self.lookahead(x)
        x = self.fc(x)
        x = x.transpose(0, 1)
        # identity in training mode, softmax in eval mode
        x = self.inference_softmax(x)
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

