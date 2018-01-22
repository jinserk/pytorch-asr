# Semisupervised Sequential Variational Autoencoder

This repository maintains an experimental code of SS-VAE implementation <span id="a1">[[1]](#f1)</span> using [Pytorch](https://github.com/pytorch/pytorch) and [Pyro](https://github.com/pyro/pyro).
The code was tested with Python 3.6.4, Pytorch 0.4.0a0, and Pyro.

## Installation

We recommend [pyenv](https://github.com/pyenv/pyenv). We assume you already have pyenv and Python 3.6.3 and Pytorch 0.4.0a0 under it.

Download [ss-vae](https://github.com/jinserk/ss-vae.git"):
```
$ git clone https://github.com/jinserk/ss-vae.git
```

Install required Python modules:
```
$ cd ss-vae
$ pip install -r requrements.txt
```

Done!

## Train and checking

Do not forget to set `pyenv local 3.6.4` in the local repo

Running visdom server if you give the commandline options `--visualize`
```
$ python -m visdom.server
```

In order to start a new training:
```
$ python train.py
```

If you want to resume training from a model file:
```
$ python train.py --continue_from <file_prefix>
```

You can see the sampled digits and latent variable T-SNE plots via web browser:
```
https://localhost:8097
```

## Reference
[1] <span id="f1"></span> Kingma, Diederik P, Danilo J Rezende, Shakir Mohamed, and Max Welling, “Semi-Supervised Learning with Deep Generative Models.”, 2014 (http://arxiv.org/abs/1406.5298) [$\hookleftarrow$](#a1)
