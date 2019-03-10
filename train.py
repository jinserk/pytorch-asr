#!python

import sys
import importlib
from pathlib import Path

models = set([
    x.name for x in Path('asr/models').iterdir()
    if x.is_dir() and x.name[0] not in ['_', '.']
])

try:
    model = [arg for arg in sys.argv[1:] if arg in models]
    if not model:
        raise RuntimeError
    model = model[0]
    argv = sys.argv[1:]
    argv.remove(model)
    #model, argv = sys.argv[1], sys.argv[2:]
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

