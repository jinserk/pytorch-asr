# ASR with PyTorch and Pyro

This repository maintains an experimental code for speech recognition using [PyTorch](https://github.com/pytorch/pytorch) and [Pyro](https://github.com/pyro/pyro).
The Kaldi latgen decoder is integrated with PyTorch binding for CTC based acoustic model training.
The code was tested with Python 3.6.5, PyTorch 0.4.0, and Pyro 0.2.0a.

## Installation

We recommend [pyenv](https://github.com/pyenv/pyenv). We assume you already have pyenv and Python 3.6.5 and PyTorch 0.4.0 under it.
Do not forget to set `pyenv local 3.6.5` in the local repo if you decide to use pyenv.

Download:
```
$ git clone https://github.com/jinserk/pytorch-asr.git
```

Install required Python modules:
```
$ cd pytorch-asr
$ pip install -r requrements.txt
```

Modify the Kaldi path in `_path.py`:
```
$ cd asr/kaldi
$ vi _path.py

KALDI_ROOT = <kaldi-installation-path>
```

Build up PyTorch-binding of Kaldi decoder:
To avoid the `-fPIC` related compile error, you need configured with `--shared` option when install Kaldi itself.
```
$ python build.py
```
This takes a while to download the Kaldi's official ASpIRE chain model and its post-processing.
You can use the asr/kaldi/graph/tokens.txt file as the CTC labels in training.


## Train and checking

This repository is a kind of framework to support multi acoustic models. You have to specify one of model to train or predict.

Running visdom server if you give the commandline options `--visdom`
```
$ python -m visdom.server
```

If the model support Tensorboard, you can use `--tensorboard` option instead. You need to install Tensorboard and TensorboardX using pip.

If you do training for the first time, you have to prepare the dataset. Currently we support only the Kaldi's ASpIRE recipe datatset, originated from LDC's fisher corpus.
It uses the phone alignment data from the result of the Kaldi recipe, so you need running the recipe at least once.
```
$ python prepare.py aspire
```

If you are going to train CTC model, you need to make the ctc labeling target file.
```
$ cd asr/kaldi
$ python prep_ctc_trans.py graph/align_lexicon.int graph/words.txt ../../data/aspire
```

In order to start a new training:
```
$ python train.py <model-name>
```
add `--help` option to see what parameters are available for each model.

If you want to resume training from a model file:
```
$ python train.py <model-name> --continue_from <model-file>
```

You can predict a sample with trained model file:
```
$ python predict.py <model-name> --continue_from <model-file> <target-wav-file>
```

