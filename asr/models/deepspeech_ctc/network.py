#!python
import math
from collections import OrderedDict

import torch
import torch.nn as nn
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


class LayerNormLSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size, bias):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        if bias:
            self.bias_ih = Parameter(torch.Tensor(4 * hidden_size))
            self.bias_hh = Parameter(torch.Tensor(4 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        self.reset_parameters()
        self.ln_ih = nn.LayerNorm(4 * hidden_size)
        self.ln_hh = nn.LayerNorm(4 * hidden_size)
        self.ln_ho = nn.LayerNorm(hidden_size)

    def check_forward_input(self, input):
        if input.size(1) != self.input_size:
            raise RuntimeError(
                "input has inconsistent input_size: got {}, expected {}".format(
                    input.size(1), self.input_size))

    def check_forward_hidden(self, input, hx, hidden_label=''):
        if input.size(0) != hx.size(0):
            raise RuntimeError(
                "Input batch size {} doesn't match hidden{} batch size {}".format(
                    input.size(0), hidden_label, hx.size(0)))

        if hx.size(1) != self.hidden_size:
            raise RuntimeError(
                "hidden{} has inconsistent hidden_size: got {}, expected {}".format(
                    hidden_label, hx.size(1), self.hidden_size))

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)

    def forward(self, input, hidden):
        self.check_forward_input(input)
        hx, cx = hidden
        self.check_forward_hidden(input, hx, 'hx')
        self.check_forward_hidden(input, cx, 'cx')
        gates = self.ln_ih(F.linear(input, self.weight_ih, self.bias_ih)) + self.ln_hh(F.linear(hx, self.weight_hh, self.bias_hh))
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * F.tanh(self.ln_ho(cy))
        return hy, cy


class BatchRNN(nn.Module):

    def __init__(self, input_size, hidden_size, rnn_type=nn.LSTM, bidirectional=False, batch_norm=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.batch_norm = SequenceWise(nn.BatchNorm1d(input_size)) if batch_norm else None
        self.rnn = rnn_type(input_size=input_size, hidden_size=hidden_size,
                            bidirectional=bidirectional, bias=True, dropout=0.5)

    def flatten_parameters(self):
        self.rnn.flatten_parameters()

    def forward(self, x):
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        x, _ = self.rnn(x)
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

    def __init__(self, rnn_type=nn.LSTM, num_classes=p.NUM_CTC_LABELS, input_folding=3,
                 rnn_hidden_size=512, nb_layers=4, bidirectional=True, context=20):
        super().__init__()

        # model metadata needed for serialization/deserialization
        self._hidden_size = rnn_hidden_size
        self._hidden_layers = nb_layers
        self._rnn_type = rnn_type
        self._bidirectional = bidirectional

        # Based on above convolutions and spectrogram size using conv formula (W - F + 2P)/ S+1
        w0 = 129
        w1 = (w0 - 41 + 2*20) // 2 + 1
        w2 = (w1 - 21 + 2*10) // 2 + 1
        w3 = (w2 - 11 + 2*5) // 2 + 1
        c0 = 2 * input_folding
        c1 = c0 * 6
        w4 = c1 * w3
        w5 = 2 * rnn_hidden_size if bidirectional else rnn_hidden_size

        self.conv = nn.Sequential(
            nn.Conv2d(c0, c1, kernel_size=(41, 11), stride=(2, 1), padding=(20, 5)),
            nn.GroupNorm(c0, c1),
            #nn.BatchNorm2d(32),
            #nn.Hardtanh(0, 20, inplace=True),
            #nn.ReLU(inplace=True),
            Swish(inplace=True),
            nn.Conv2d(c1, c1, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.GroupNorm(c0, c1),
            #nn.BatchNorm2d(32),
            #nn.Hardtanh(0, 20, inplace=True)
            #nn.ReLU(inplace=True),
            Swish(inplace=True),
            nn.Conv2d(c1, c1, kernel_size=(11, 11), stride=(2, 1), padding=(5, 5)),
            nn.GroupNorm(c0, c1),
            #nn.BatchNorm2d(32),
            #nn.Hardtanh(0, 20, inplace=True)
            #nn.ReLU(inplace=True),
            Swish(inplace=True),
        )

        #rnns = []
        #rnn = BatchRNN(input_size=w4, hidden_size=rnn_hidden_size, rnn_type=rnn_type,
        #               bidirectional=bidirectional, batch_norm=False)
        #rnns.append(('0', rnn))
        #for x in range(nb_layers - 1):
        #    rnn = BatchRNN(input_size=rnn_hidden_size, hidden_size=rnn_hidden_size, rnn_type=rnn_type,
        #                   bidirectional=bidirectional)
        #    rnns.append(('%d' % (x + 1), rnn))
        #self.rnns = nn.Sequential(OrderedDict(rnns))

        self.rnns = nn.LSTM(input_size=w4, hidden_size=rnn_hidden_size, num_layers=nb_layers,
                            bidirectional=bidirectional, dropout=0.5)

        self.lookahead = nn.Sequential(
            # consider adding batch norm?
            Lookahead(rnn_hidden_size, context=context),
            #nn.Hardtanh(0, 20, inplace=True)
            #nn.ReLU(inplace=True),
            Swish(inplace=True)
        ) if not bidirectional else None

        fully_connected = nn.Sequential(
            nn.BatchNorm1d(w5),
            nn.Linear(w5, num_classes, bias=False)
        )
        self.fc = nn.Sequential(
            SequenceWise(fully_connected),
        )
        self.inference_softmax = InferenceBatchSoftmax()

    def forward(self, x):
        x = self.conv(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # Collapse feature dimension
        x = x.transpose(1, 2).transpose(0, 1).contiguous()  # TxNxH
        #x = self.rnns(x)
        x, _ = self.rnns(x)

        if not self._bidirectional:  # no need for lookahead layer in bidirectional
            x = self.lookahead(x)
        x = self.fc(x)
        x = x.transpose(0, 1)
        # identity in training mode, softmax in eval mode
        x = self.inference_softmax(x)
        return x


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

