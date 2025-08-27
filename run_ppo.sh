cuda=3

echo "CUDA_VISIBLE_DEVICES: $cuda"

# 设置SFT预训练模型路径
PRETRAINED_MODEL_PATH="/mnt/public/chenyinuo/RL4VLA/runs/mlp_sft_steps_50000/step_050000"

cd SimplerEnv
CUDA_VISIBLE_DEVICES=$cuda /home/chenyinuo/data/miniconda3/envs/rlvla_env/bin/python simpler_env/train_ms3_ppo_mlp.py --pretrained_mlp_path "$PRETRAINED_MODEL_PATH"
