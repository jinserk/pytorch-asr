# Semisupervised Sequential Variational Autoencoder

This repository maintains an experimental code of SS-VAE implementation <span id="a1">[[1]](#f1)</span> using [Pytorch](https://github.com/pytorch/pytorch) and [ProbTorch](https://github.com/probtorch/probtorch).
The code was tested with Python 3.6.4, Pytorch 0.4.0a0, and Probtorch.

## Installation

We recommend [pyenv](https://github.com/pyenv/pyenv). We assume you already have pyenv and Python 3.6.3 and Pytorch 0.4.0a0 under it.

Install required Python modules:
```
pip install -r requrements.txt
```

Download [ss-vae](https://github.com/jinserk/ss-vae.git"):
```
$ git clone https://github.com/jinserk/ss-vae.git
```

Done!

## Train and checking

Do not forget to set `pyenv local 3.6.4` in the local repo

In order to start a new training:
```
$ python train.py
```

If you want to resume training from a model file:
```
$ python train.py --continue_from <file_prefix>
```

In order to check the trained result:
```
$ python check.py --model_prefix <file_prefix>
```


## Reference
[1] <span id="f1"></span> Kingma, Diederik P, Danilo J Rezende, Shakir Mohamed, and Max Welling, “Semi-Supervised Learning with Deep Generative Models.”, 2014 (http://arxiv.org/abs/1406.5298) [$\hookleftarrow$](#a1)
