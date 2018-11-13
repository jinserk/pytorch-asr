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
        x = x.contiguous().view(t * n, -1)
        x = self.module(x)
        x = x.contiguous().view(t, n, -1)
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr


class DeepSpeech(nn.Module):

    def __init__(self, num_classes=p.NUM_CTC_LABELS, input_folding=2, rnn_type=nn.LSTM,
                 rnn_hidden_size=256, rnn_num_layers=4, bidirectional=True, smoothing=0.01):
        super().__init__()

        # model metadata needed for serialization/deserialization
        self._rnn_type = rnn_type
        self._hidden_size = rnn_hidden_size
        self._hidden_layers = rnn_num_layers
        self._bidirectional = bidirectional

        # Based on above convolutions and spectrogram size using conv formula (W - F + 2P)/ S+1
        W0 = 129
        C0 = 2 * input_folding
        W1 = (W0 - 11 + 2*5) // 2 + 1  # 65
        C1 = 64
        W2 = (W1 - 11 + 2*5) // 2 + 1  # 33
        C2 = 2 * C1
        W3 = (W2 - 11 + 2*5) // 2 + 1  # 17
        C3 = 2 * C2
        #W4 = (W3 - 11 + 2*5) // 2 + 1  # 9
        #C4 = 128

        H0 = C3 * W3
        H1 = rnn_hidden_size * 2 if bidirectional else rnn_hidden_size

        self.feature = nn.Sequential(OrderedDict([
            ("cv1", nn.Conv2d(C0, C1, kernel_size=(11, 3), stride=(1, 1), padding=(5, 1))),
            #("nl1", nn.Hardtanh(-5, 5)),
            ("nl1", nn.LeakyReLU()),
            #("nl1", Swish()),
            ("mp1", nn.AvgPool2d(kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))),
            ("bn1", nn.BatchNorm2d(C1)),
            ("cv2", nn.Conv2d(C1, C2, kernel_size=(11, 3), stride=(1, 1), padding=(5, 1))),
            #("nl2", nn.Hardtanh(-10, 10)),
            ("nl2", nn.LeakyReLU()),
            #("nl2", Swish()),
            ("mp2", nn.AvgPool2d(kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))),
            ("bn2", nn.BatchNorm2d(C2)),
            ("cv3", nn.Conv2d(C2, C3, kernel_size=(11, 3), stride=(1, 1), padding=(5, 1))),
            #("nl3", nn.Hardtanh(-20, 20)),
            ("nl3", nn.LeakyReLU()),
            #("nl3", Swish()),
            ("mp3", nn.AvgPool2d(kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))),
            ("bn3", nn.BatchNorm2d(C3)),
            #("cv4", nn.Conv2d(C3, C4, kernel_size=(11, 3), stride=(1, 1), padding=(5, 1))),
            ##("nl4", nn.Hardtanh(-20, 20)),
            #("nl4", nn.LeakyReLU()),
            ##("nl4", Swish()),
            #("mp4", nn.MaxPool2d(kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))),
            #("bn4", nn.BatchNorm2d(C4)),
        ]))

        # using multi-layered nn.LSTM
        self.batch_first = True
        self.rnns = nn.LSTM(input_size=H0, hidden_size=rnn_hidden_size, num_layers=rnn_num_layers,
                            bias=True, bidirectional=bidirectional, batch_first=self.batch_first)

        self.fc = SequenceWise(nn.Sequential(OrderedDict([
            ("ln1", nn.LayerNorm(H1, elementwise_affine=False)),
            ("fc1", nn.Linear(H1, num_classes, bias=False)),
        ])))

        #self.softmax = InferenceBatchSoftmax()
        self.softmax = nn.Softmax(dim=-1)
        self.smoothing = smoothing

    def smooth_labels(self, x):
        return (1.0 - self.smoothing) * x + self.smoothing / x.size(-1)

    def forward(self, x, seq_lens):
        h = self.feature(x)
        h = h.view(-1, h.size(1) * h.size(2), h.size(3))  # Collapse feature dimension
        g = h.transpose(1, 2).contiguous()  # NxTxH

        ps = nn.utils.rnn.pack_padded_sequence(g, seq_lens.tolist(), batch_first=self.batch_first)
        ps, _ = self.rnns(ps)
        g, _ = nn.utils.rnn.pad_packed_sequence(ps, batch_first=self.batch_first)

        y = self.fc(g)
        y = self.smooth_labels(self.softmax(y)).log()

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

