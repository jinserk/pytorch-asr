#!/bin/bash
eval "$(pyenv init -)"

python -m torch.distributed.launch \
  --nproc_per_node=1 \
  --nnodes=2 \
  --node_rank=$1 \
  --master_addr="172.30.1.237" \
  --master_port 24000 \
  batch_train.py deepspeech_ctc \
  --use-cuda \
  --log-dir logs_20180827_deepspeech_ctc \
  --visdom \
  --visdom-host 172.26.15.44
