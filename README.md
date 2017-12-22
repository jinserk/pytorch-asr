# SS VAE

This repository maintains an experimental code of SS VAE implementation using [Pytorch](https://github.com/pytorch/pytorch) and [ProbTorch](https://github.com/probtorch/probtorch).
The code was tested with Python 3.6.3, Pytorch 0.4.0a0+32a4a52, and Probtorch 0.0+579de67.

## Installation

We recommend [pyenv](https://github.com/pyenv/pyenv). All descriptions here is under the assumption you already have pyenv and Python 3.6.3 and Pytorch 0.4.0a0 under it.

1. download [Probtorch](https://github.com/probtorch/probtorch)
```
$ git clone https://github.com/probtorch/probtorch.git
```
    modify `probtorch/probtorch/version.py` as following:

