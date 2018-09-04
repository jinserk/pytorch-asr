#!/bin/bash -l

#SBATCH -q debug
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:gtx1070:1
#SBATCH --job-name=jbaik
#SBATCH --output=slurm.%J.%t.out
#SBATCH --error=slurm.%J.%t.err
#SBATCH --network="MPI,DEVNAME=bond0"

eval "$(pyenv init -)"
export NCCL_DEBUG=INFO
export MASTER_ADDR="172.30.1.237"
export MASTER_PORT="23456"

srun --export=ALL \
  python batch_train.py deepspeech_ctc \
    --use-cuda \
    --log-dir logs_20180901_deepspeech_ctc_h512_l4 \
    --visdom \
    --visdom-host 172.26.15.44 \
    --continue-from logs_20180901_deepspeech_ctc_h512_l4/deepspeech_ctc_epoch_003.pth.tar \
