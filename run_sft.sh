cuda=3

echo "CUDA_VISIBLE_DEVICES: $cuda"

CUDA_VISIBLE_DEVICES=$cuda /home/chenyinuo/data/miniconda3/envs/rlvla_env/bin/python finetune_mlp.py
