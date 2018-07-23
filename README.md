# ASR with PyTorch

This repository maintains an experimental code for speech recognition using [PyTorch](https://github.com/pytorch/pytorch) and [Kaldi](https://github.com/kaldi-asr/kaldi).
The Kaldi latgen decoder is integrated with PyTorch binding for CTC based acoustic model training.
The code was tested with Python 3.7 and PyTorch 0.5.

## Installation

**Prerequisites:**
    * Python 3.6+
    * PyTorch 0.4.1+
    * Kaldi 5.3+

We recommend [pyenv](https://github.com/pyenv/pyenv).
Do not forget to set `pyenv local 3.7.0` in the local repo if you're using pyenv.

Currently, AdamW and SGDR patch ([PR #4429](https://github.com/pytorch/pytorch/pull/4429) and [PR #7821](https://github.com/pytorch/pytorch/pull/7821)) has to be applied
to Pytorch 0.5.0a0 ([git branch v0.4.1](https://github.com/pytorch/pytorch/tree/v0.4.1)) to use AdamW or SGDR optimizer.
If you don't want to use this, please correct each optimizer setting in the function of `__setup_networks()` in `model.py` files to avoid corresponding errors.

To avoid the `-fPIC` related compile error, you have to configure Kaldi with `--shared` option when you install it.

Download:
```
$ git clone https://github.com/jinserk/pytorch-asr.git
```

Install required Python modules:
```
$ cd pytorch-asr
$ pip install -r requirements.txt
```

Modify the Kaldi path in `_path.py`:
```
$ cd asr/kaldi
$ vi _path.py

KALDI_ROOT = <kaldi-installation-path>
```

Build up PyTorch-binding of Kaldi decoder:
```
$ python build.py
```
This takes a while to download the Kaldi's official ASpIRE chain model and its post-processing.
If you want to use your own language model or graphs, modify `asr/kaldi/scripts/mkgraph.sh` as your settings.


## Training

Pytorch-asr is targeted to develop a framework supporting multiple acoustic models. You have to specify one of model to train or predict.
Currently, `resnet_ctc`, `resnet_ed`, and `deepspeech` models work for training and prediction. Try these models first.

If you do training for the first time, you have to prepare the dataset.
Currently we support only the Kaldi's ASpIRE recipe datatset, originated from LDC's fisher corpus.
Please modify `asr/dataset/aspire.py` according to the location of your corpus data.
```
$ python prepare.py aspire
```

If you have any related Kaldi recipe and its `exp` directories which contains the result of training,
you can use the phone alignment result for CTC training.
If you don't use the Kaldi's training result, you can generate the ctc labeling files from each utterence's transcript and the corresponding lexicon dictionary.
```
$ cd asr/kaldi
$ python prep_ctc_trans.py ../../data/aspire
```

Start a new training with:
```
$ python train.py <model-name> --use-cuda
```
where `<model_name>` can be either of `resnet_ctc` or `resnet_ed`.
check `--help` option to see which parameters are available for the model.

If you want to resume training from a saved model file:
```
$ python train.py <model-name> --use-cuda --continue-from <model-file>
```

You can use `--tensorboard` option to see the loss propagation. You have to install Tensorboard and TensorboardX from pip to use it.


## Prediction

You can predict a sample with trained model file:
```
$ python predict.py <model-name> --continue-from <model-file> <target-wav-file1> <target-wav-file2> ...
```

