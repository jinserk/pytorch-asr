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

srun \
  python train.py deepspeech_ctc \
    --use-cuda \
    --max-len 15 \
    --log-dir logs_20180829_deepspeech_ctc \
    --visdom \
    --visdom-host 172.26.15.44
