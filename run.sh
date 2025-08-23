#! /bin/bash

cuda=3 

echo "CUDA_VISIBLE_DEVICES: $cuda"



CUDA_VISIBLE_DEVICES=$cuda /home/chenyinuo/data/miniconda3/envs/rlvla_env/bin/python SimplerEnv/simpler_env/train_ms3_ppo_mlp.py \
    --wandb \