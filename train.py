#!python

import sys
import argparse
from pathlib import Path

from asr import convnet
from asr import densenet
from asr import densenet_ctc
from asr import capsule1
from asr import capsule2

argv = sys.argv[1:]
models = set(["convnet", "densenet", "densenet_ctc", "capsule1", "capsule2"])
model = None
for opt in argv:
    if opt in models:
        model = opt
        argv.remove(opt)
        break

if model == "convnet":
    convnet.train(argv)
elif model == "densenet":
    densenet.train(argv)
elif model == "densenet_ctc":
    densenet_ctc.train(argv)
elif model == "capsule1":
    capsule1.train(argv)
elif model == "capsule2":
    capsule2.train(argv)
else:
    print(f"Error: choose one of models in {models}")
    sys.exit(1)
