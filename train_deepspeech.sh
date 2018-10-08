#!/bin/bash -l

#SBATCH -q debug
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:gtx1070:1
#SBATCH --job-name=jbaik
#SBATCH --output=slurmctld.%j.out
#SBATCH --error=slurmctld.%j.err

eval "$(pyenv init -)"
export NCCL_DEBUG=INFO
export MASTER_ADDR="172.30.1.237"
export MASTER_PORT="23456"

. .slackbot

srun -o slurmd.%j.%t.out -e slurmd.%j.%t.err --export=ALL --network="MPI,DEVNAME=bond0" \
  python batch_train.py deepspeech_ctc \
    --use-cuda \
    --fp16 \
    --slack \
    --visdom \
    --visdom-host 172.26.15.44 \
    --log-dir logs_20181008_deepspeech_ctc_fold3
    #--continue-from logs_20181003_deepspeech_ctc_fold2/deepspeech_ctc_epoch_009.pth.tar
