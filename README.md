# SS VAE

This repository maintains an experimental code of SS VAE implementation <span id="a1">[[1]](#f1)</span> using [Pytorch](https://github.com/pytorch/pytorch) and [ProbTorch](https://github.com/probtorch/probtorch).
The code was tested with Python 3.6.4, Pytorch 0.4.0a0+32a4a52, and Probtorch 0.0+579de67.

## Installation

We recommend [pyenv](https://github.com/pyenv/pyenv). We assume you already have pyenv and Python 3.6.3 and Pytorch 0.4.0a0 under it.

Download [Probtorch](https://github.com/probtorch/probtorch) to somewhere:
```
$ git clone https://github.com/probtorch/probtorch.git
```

Modify `probtorch/probtorch/version.py` as following:
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

Download [SS_VAE](https://github.com/jinserk/ss_vae.git"):
```
$ git clone https://github.com/jinserk/ss_vae.git
```

Modify line 4 in `ss_vae/probtorch_env.py`:
```
sys.path.append('<absolute path of probtorch you've downloaded>')
```

Install required Python modules:
```
pip install -r requrements.txt
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
