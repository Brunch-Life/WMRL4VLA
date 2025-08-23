## Evaluation Policy in ManiSkill simulation environment
### Project Env Installation

```bash 
conda create -n maniskill_evaluation python==3.10 -y
conda activate maniskill_evaluation
pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install gymnasium==0.29.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 -i https://pypi.tuna.tsinghua.edu.cn/simple --extra-index-url https://download.pytorch.org/whl/cu121

pip install diffusers==0.11.1 huggingface_hub==0.25.2 # for diffusion policy evaluation
pip install d3rlpy==2.8.1 # for rl policy evaluation
pip install coacd # for 3d_assets
```

You Should also download the 3d_assets and modify the `ASSET_3D_PATH` and `CONTAINER_3D_PATH` in [file](mani_skill/examples/real2sim_3d_assets/__init__.py)
```bash
pip install gdown
gdown "https://drive.google.com/uc?id=1FUOCM5mrI0xxzxvYoBBYOZtoCREDe-9n" # url from bingwen
unzip 'policy_evaluation_3d_assets.zip' -d policy_evaluation_3d_assets
rm 'policy_evaluation_3d_assets.zip'
```

### Env render on local computer
``` bash
python -m mani_skill.examples.demo_random_action -e TabletopPickEnv-v1 --render-mode="rgb_array" # human

python -m mani_skill.examples.demo_random_action -e TabletopPickPlaceEnv-v1 --render-mode="rgb_array" # human

python -m mani_skill.examples.demo_random_action_episode --render_mode human -e TabletopPickPlaceEnv-v1 -r panda_wristcam

```

### Evaluation in Tabletop for Diffusion Policy
```bash
# pick and place task
object_name="green_bell_pepper"
container_name="plate"
pick_place_ckpt_path="/nvme_data/bingwen/Documents/temp/dp/policy_best.ckpt"
pick_place_obs_normalize_params_path="/nvme_data/bingwen/Documents/temp/dp/norm_stats_1.pkl"
CUDA_VISIBLE_DEVICES=5 XLA_PYTHON_CLIENT_PREALLOCATE=false python -m mani_skill.evaluation.policy_evaluation \
    --model="diffusion_policy" --ckpt_path="${pick_place_ckpt_path}" \
    -e "TabletopPickPlaceEnv-v1" -s 0 --num-episodes 3 --num-envs 3 --save-video --max_episode_len 200 \
    --object_name="$object_name" --container_name="$container_name" -r panda_wristcam -c pd_ee_pose \
    --obs_normalize_params_path="$pick_place_obs_normalize_params_path" --is_delta
```

### Evaluation in Tabletop for CQL

```bash
# pick and place task
object_name="blueberry"
container_name="plate"
pick_place_ckpt_path="/nvme_data/bingwen/Documents/temp/rl/model_16400.d3"  # model_2000.d3
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false python -m mani_skill.evaluation.policy_evaluation \
    --model="cql" --ckpt_path="${pick_place_ckpt_path}" \
    -e "TabletopPickPlaceEnv-v1" -s 0 --num-episodes 100 --num-envs 1 --save-video --max_episode_len 50 \
    --object_name="$object_name" --container_name="$container_name" -r panda -c pd_ee_pose \
    --obs_normalize_params_path="$pick_place_obs_normalize_params_path" --is_table_green --is_delta
```
