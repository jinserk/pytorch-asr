# ASR with PyTorch

This repository maintains an experimental code for speech recognition using [PyTorch](https://github.com/pytorch/pytorch) and [Kaldi](https://github.com/kaldi-asr/kaldi).
The Kaldi latgen decoder is integrated with PyTorch binding for CTC based acoustic model training.
The code was tested with Python 3.7 and PyTorch 0.4.1.

## Installation

**Prerequisites:**
* Python 3.6+
* [PyTorch 0.4.1+](https://github.com/pytorch/pytorch/pytorch/tree/v0.4.1)
* [Kaldi 5.3+](https://github.com/kaldi-asr/kaldi.git)
* [TNT](https://github.com/pytorch/tnt.git)

We recommend [pyenv](https://github.com/pyenv/pyenv).
Do not forget to set `pyenv local 3.7.0` in the local repo if you're using pyenv.

If you want to use AdamW as your optimizer, you need to patch ([PR #4429](https://github.com/pytorch/pytorch/pull/4429) to PyTorch source by yourself.
CosineAnnealingWithRestartLR for SGDR from [PR #7821](https://github.com/pytorch/pytorch/pull/7821)) is included in `asr/utils/lr_scheduler.py` as a part of project.

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
If you have installation error of `torchaudio` on CentOS machine, just comment out the line from `requirements.txt` and install it from its [source](https://github.com/pytorch/audio.git).
You need to modify `#include <sox.h>` to `#include <sox/sox.h>` in `torchaudio/torch_sox.cpp` of the source to install and run.

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
If you want to use your own language model or graphs, modify `asr/kaldi/scripts/mkgraph.sh` according to your settings.


## Training

Pytorch-asr is targeted to develop a framework supporting multiple acoustic models. You have to specify one of model to train or predict.
Currently, `resnet_{ctc,ed}`, `densenet_{ctc,ed}`, and `deepspeech_{ctc,ed}` models work for training and prediction. Try these models first.

If you do training for the first time, you have to preprocess the dataset.
Currently we utilize Kaldi's recipe directory containing preprocessed corpus data.
You need to run the preparation script in Kaldi recipe before doing the followings.
Now we support only the Kaldi's ASpIRE recipe datatset, originated from LDC's fisher corpus.
Please modify `RECIPE_PATH` variable in `asr/dataset/aspire.py` first according to the location of your Kaldi setup.
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
check `--help` option to see which parameters are available for the model.

If you want to resume training from a saved model file:
```
$ python train.py <model-name> --use-cuda --continue-from <model-file>
```

You can use `--visdom` or `--tensorboard` option to see the loss propagation.


## Prediction

You can predict a sample with trained model file:
```
$ python predict.py <model-name> --continue-from <model-file> <target-wav-file1> <target-wav-file2> ...
```

## Acknowledgement

Some models are imported from the following projects. We appreciate all their work and all right of the codes belongs to them.

* DeepSpeech : Sean Naren (https://github.com/SeanNaren/deepspeech.pytorch.git)
* ResNet : PyTorch Vision Team (https://github.com/pytorch/vision.git)
* DenseNet : PyTorch Vision Team (https://github.com/pytorch/vision.git)

