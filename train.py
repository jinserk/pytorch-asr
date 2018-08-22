#!python

import sys


argv = sys.argv[1:]
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

if argv[0] not in models:
    print(f"Error: choose one of models in {models}")
    sys.exit(1)

model = argv[0]
argv.remove(model)

if model == "convnet":
    from asr.models import convnet
    convnet.train(argv)
elif model == "densenet":
    from asr.models import densenet
    densenet.train(argv)
elif model == "densenet_ctc":
    from asr.models import densenet_ctc
    densenet_ctc.train(argv)
elif model == "deepspeech_ctc":
    from asr.models import deepspeech_ctc
    deepspeech_ctc.train(argv)
elif model == "deepspeech_ce":
    from asr.models import deepspeech_ce
    deepspeech_ce.train(argv)
elif model == "resnet_ctc":
    from asr.models import resnet_ctc
    resnet_ctc.train(argv)
elif model == "resnet_ce":
    from asr.models import resnet_ce
    resnet_ce.train(argv)
elif model == "resnet_split":
    from asr.models import resnet_split
    resnet_split.train(argv)
elif model == "resnet_split_ce":
    from asr.models import resnet_split_ce
    resnet_split_ce.train(argv)
elif model == "capsule1":
    from asr.models import capsule1
    capsule1.train(argv)
elif model == "capsule2":
    from asr.models import capsule2
    capsule2.train(argv)
