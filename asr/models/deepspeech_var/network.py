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

    def forward(self, x, seq_lens):
        if self.layer_norm is not None:
            h = self.layer_norm(x)
        ps = nn.utils.rnn.pack_padded_sequence(h, seq_lens.tolist(), batch_first=self.batch_first)
        ps, _ = self.rnn(ps)
        y, _ = nn.utils.rnn.pad_packed_sequence(ps, batch_first=self.batch_first)
        if self.bidirectional:
            # (NxTxH*2) -> (NxTxH) by sum
            y = y.view(y.size(0), y.size(1), 2, -1).sum(2).view(y.size(0), y.size(1), -1)
        return y


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
            nn.Linear(H0, rnn_hidden_size, bias=True),
            nn.Dropout(0.5, inplace=True),
            #nn.Hardtanh(-10, 10, inplace=True),
            nn.LeakyReLU(inplace=True),
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

        self.fc2 = SequenceWise(nn.Sequential(
            nn.LayerNorm(H1, elementwise_affine=False),
            nn.Linear(H1, 256, bias=True),
            nn.Dropout(0.5, inplace=True),
            nn.Linear(256, num_classes, bias=True),
        ))
        #self.softmax = nn.LogSoftmax(dim=-1)
        self.softmax = InferenceBatchSoftmax()

    def forward(self, x, seq_lens):
        h = self.conv(x)
        h = h.view(-1, h.size(1) * h.size(2), h.size(3))  # Collapse feature dimension
        h = h.transpose(1, 2).contiguous()  # NxTxH
        h = self.fc1(h)
        g = self.rnns[0](h, seq_lens)
        for i in range(1, self._hidden_layers):
            g = g + h
            g = self.rnns[i](g, seq_lens)
        g = g + h
        y = self.fc2(g)
        y = self.softmax(y)
        return y, seq_lens


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

