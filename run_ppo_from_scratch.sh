cuda=6

echo "CUDA_VISIBLE_DEVICES: $cuda"

cd SimplerEnv
CUDA_VISIBLE_DEVICES=$cuda /home/chenyinuo/data/miniconda3/envs/rlvla_env/bin/python simpler_env/train_ms3_ppo_mlp.py
