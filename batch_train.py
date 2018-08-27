#!python

import sys
import importlib

models = set([
    "densenet_ctc",
    "deepspeech_ctc",
    "resnet_ctc",
])

try:
    model, argv = sys.argv[1], sys.argv[2:]
    if model not in models:
        raise
except:
    print(f"Error: choose one of models in {models}")
    sys.exit(1)

try:
    m = importlib.import_module(f"asr.models.{model}")
    m.batch_train(argv)
except:
    raise

