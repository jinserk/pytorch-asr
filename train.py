#!python

import sys
import argparse
from pathlib import Path

argv = sys.argv[1:]
models = set(["convnet", "densenet", "densenet_ctc", "resnet_ctc", "resnet_ed", "capsule1", "capsule2"])
model = None
for opt in argv:
    if opt in models:
        model = opt
        argv.remove(opt)
        break

if model == "convnet":
    from asr import convnet
    convnet.train(argv)
elif model == "densenet":
    from asr import densenet
    densenet.train(argv)
elif model == "densenet_ctc":
    from asr import densenet_ctc
    densenet_ctc.train(argv)
elif model == "resnet_ctc":
    from asr import resnet_ctc
    resnet_ctc.train(argv)
elif model == "resnet_ed":
    from asr import resnet_ed
    resnet_ed.train(argv)
elif model == "capsule1":
    from asr import capsule1
    capsule1.train(argv)
elif model == "capsule2":
    from asr import capsule2
    capsule2.train(argv)
else:
    print(f"Error: choose one of models in {models}")
    sys.exit(1)
