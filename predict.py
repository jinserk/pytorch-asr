#!python

import sys
import argparse
from pathlib import Path

data_root = Path.cwd().resolve() / "data" / "aspire"

argv = sys.argv[1:]
models = set(["convnet", "densenet", "capsule1"])
model = None
for opt in argv:
    if opt in models:
        model = opt
        argv.remove(opt)
        break

if model == "convnet":
    from asr import convnet
    convnet.predict(argv)
elif model == "densenet":
    from asr import densenet
    densenet.predict(argv)
elif model == "capsule1":
    from asr import capsule1
    capsule1.predict(argv)
else:
    print(f"Error: choose one of models in {models}")
    sys.exit(1)
