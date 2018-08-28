#!python

import sys
import importlib

models = set([
    "convnet",
    "densenet",
    "densenet_ctc",
    "deepspeech_ctc",
    "deepspeech_ce",
    "resnet_ctc",
    "resnet_ce",
    "resnet_split",
    "resnet_split_ce",
    "capsule1",
    "capsule2",
])

try:
    model, argv = sys.argv[7], sys.argv[8:]
    if model not in models:
        raise
except:
    print(f"Error: choose one of models in {models}")
    sys.exit(1)

try:
    m = importlib.import_module(f"asr.models.{model}")
    m.train(argv)
except:
    raise

