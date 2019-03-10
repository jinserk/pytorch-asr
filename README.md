# ASR with PyTorch

This repository maintains an experimental code for speech recognition using [PyTorch](https://github.com/pytorch/pytorch) and [Kaldi](https://github.com/kaldi-asr/kaldi).
We are more focusing on better acoustic model that produce phoneme sequence than end-to-end transcription.
For this purpose, the Kaldi latgen decoder is integrated as a PyTorch CppExtension.

The code was tested with Python 3.7 and PyTorch 1.0.0rc1. We have a lot of [f-strings](https://www.python.org/dev/peps/pep-0498/), so you must use Python 3.6 or later.

## Performance

| model | train dataset | dev dataset | test dataset | LER | WER |
|-------|---------------|-------------|--------------|-----|-----|
| decoder baseline<sup id="a1">[1](#f1)</sup> | - | - | swbd rt03 | - | 1.74% |
| deepspeech_var | aspire + swbd train | swbd eval2000 | swbd rt03 | 33.73% | 37.75% |
| las | aspire + swbd train | swbd eval2000 | swbd rt03 |       |      |


<sub><sup id="f1">1. This is the result by engaging the phone label sequences (onehot vectors) into the decoder input.
The result is from < 20-sec utterances, choosing a random pronunciation for words from the lexicon if the words have multiple pronunciations, after
inserting sil phones with prob 0.2 between the words and with prob 0.8 at the beginning and end of the utterances.
please see [here](https://github.com/jinserk/pytorch-asr/blob/master/asr/models/trainer.py#L459) with `target_test=True`. [&#9166;](#a1)</sup></sub>

## Installation

**Prerequisites:**
* Python 3.6+
* [PyTorch 1.0.0+](https://github.com/pytorch/pytorch/pytorch.git)
* [Kaldi 5.3+](https://github.com/kaldi-asr/kaldi.git)
* [TNT](https://github.com/pytorch/tnt.git)

We recommend [pyenv](https://github.com/pyenv/pyenv).
Do not forget to set `pyenv local <python-version>` in the local repo if you're using pyenv.

To avoid the `-fPIC` related compile error, you have to configure Kaldi with `--shared` option when you install it.

Install dependent packages:
```
$ sudo apt install sox libsox-dev
```

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
$ python setup.py install
```
This takes a while to download the Kaldi's official ASpIRE chain model and its post-processing.
If you want to use your own language model or graphs, modify `asr/kaldi/scripts/mkgraph.sh` according to your settings.
**The binding install method has been changed to use PyTorch's CppExtension, instead of ffi.
This will install a package named `torch_asr._latgen_lib`.**


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

