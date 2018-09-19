# ASR with PyTorch

This repository maintains an experimental code for speech recognition using [PyTorch](https://github.com/pytorch/pytorch) and [Kaldi](https://github.com/kaldi-asr/kaldi).
The Kaldi latgen decoder is integrated with PyTorch binding for CTC based acoustic model training.
The code was tested with Python 3.7 and PyTorch 0.4.1+.

## Installation

**Prerequisites:**
* Python 3.6+
* [PyTorch 0.4.1+](https://github.com/pytorch/pytorch/pytorch/tree/v0.4.1)
* [Kaldi 5.3+](https://github.com/kaldi-asr/kaldi.git)
* [TNT](https://github.com/pytorch/tnt.git)

We recommend [pyenv](https://github.com/pyenv/pyenv).
Do not forget to set `pyenv local 3.7.0` in the local repo if you're using pyenv.

If you want to use AdamW as your optimizer, you need to patch [PR #4429](https://github.com/pytorch/pytorch/pull/4429) to PyTorch source by yourself.
CosineAnnealingWithRestartLR for SGDR from [PR #7821](https://github.com/pytorch/pytorch/pull/7821) is included in `asr/utils/lr_scheduler.py` as a part of this project.

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

If you have an installation error of `torchaudio` on a CentOS machine, add the followings to your `~/.bashrc`.
```
export CPLUS_INCLUDE_PATH=/usr/include/sox:$CPLUS_INCLUDE_PATH
```
don't forget to do `$ source ~/.bashrc` before you try to install the requirements.

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

Pytorch-asr is targeted to develop a framework supporting multiple acoustic models. You have to specify one of the models to train or predict.
Currently, the `deepspeech_ctc` model is only maintained from the frequent updated training and prediction modules. Try this model first.
We'll follow up the other models for the updated interface soon. Sorry for your inconvenience.

If you do training for the first time, you need to preprocess the dataset.
Currently we utilize the contents of `data` directory in Kaldi's recipe directories that are containing preprocessed corpus data.
You need to run the preparation script in each Kaldi recipe before doing the followings.
Now we support the Kaldi's `aspire`, `swbd`, and `tedlium` recipes. You will need LDC's corpora to use `aspire` and `swbd` datasets.
Please modify `RECIPE_PATH` variable in `asr/datasets/*.py` first according to the location of your Kaldi setup.
```
$ python prepare.py aspire <data-path>
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

You can use `--visdom` option to see the loss propagation.
Please make sure that you already have a running visdom process before you start a training with `--visdom` option.
`--tensorboard` option is outdated since TensorboardX package doesn't support the latest PyTorch.

You can also use `--slack` option to redirect logs to slack DM.
If you want to use this, first setup a slack workplace and add "Bots" app to the workplace.
You must obtain the Bots' token and your id from the slack setting.
Then set environment variables `SLACK_API_TOKEN` and `SLACK_API_USER` for each of them.


## Prediction

You can predict a sample with trained model file:
```
$ python predict.py <model-name> --continue-from <model-file> <target-wav-file1> <target-wav-file2> ...
```

## Acknowledgement

Some models are imported from the following projects. We appreciate all their work and all right of the codes belongs to them.

* DeepSpeech : https://github.com/SeanNaren/deepspeech.pytorch.git
* ResNet : https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
* DenseNet : https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py

