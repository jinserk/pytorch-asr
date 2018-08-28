#!python

import sys
import importlib

models = set([
    "densenet_ctc",
    "deepspeech_ctc",
    "resnet_ctc",
])

argv = sys.argv

try:
    model = argv[2]
    if model not in models:
        raise
except:
    print(f"Error: choose one of models in {models}")
    sys.exit(1)

argv.remove(model)

try:
    m = importlib.import_module(f"asr.models.{model}")
    m.batch_train(argv[1:])
except:
    raise

