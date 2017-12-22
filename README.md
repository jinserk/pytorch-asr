# SS VAE

This repository maintains an experimental code of SS VAE implementation<span id="a1">[[1]](#f1)</span> using [Pytorch](https://github.com/pytorch/pytorch) and [ProbTorch](https://github.com/probtorch/probtorch).
The code was tested with Python 3.6.3, Pytorch 0.4.0a0+32a4a52, and Probtorch 0.0+579de67.

## Installation

We recommend [pyenv](https://github.com/pyenv/pyenv). We assume you already have pyenv and Python 3.6.3 and Pytorch 0.4.0a0 under it.

download [Probtorch](https://github.com/probtorch/probtorch):
```
$ git clone https://github.com/probtorch/probtorch.git
```

modify `probtorch/probtorch/version.py` as following:
```
def git_revision():
    import os, subprocess
    tmp = os.getcwd()
    os.chdir('<full path of probtorch you downloaded>')
    rev = subprocess.check_output("git rev-parse --short HEAD".split())
    os.chdir(tmp)
    return rev.strip().decode('utf-8')

__version__ = "0.0+" + str(git_revision())
```

download [SS_VAE](https://github.com/jinserk/ss_vae.git"):
```
$ git clone https://github.com/jinserk/ss_vae.git
```

modify line 19 in `ss_vae/main.py`:
```
sys.path.append('<absolute path of probtorch you downloaded>')
```

done!

## Training

do not forget to set `pyenv local 3.6.3` in the local repo

```
$ python main.py
```

## Load model from stored file

```
$ python main.py --load_model
```

## Reference
1. <span id="f1"></span> Kingma, Diederik P, Danilo J Rezende, Shakir Mohamed, and Max Welling. 2014. “Semi-Supervised Learning with Deep Generative Models.”[[Paper]](http://arxiv.org/abs/1406.5298) [$\hookleftarrow$](#a1)
