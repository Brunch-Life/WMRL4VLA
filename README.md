
## Install

### OpenVLA, Maniskill, Training Pipeline

```bash
# create conda env: rlvla_env
conda create -n rlvla_env -y python=3.10
conda activate rlvla_env

# install dependencies
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
cd openvla && pip install -e . && cd ..
pip install -U tyro
pip install datasets==3.3.2

# special install for flash attention
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install flash_attn-2.7.4.post1+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
rm flash_attn-2.7.4.post1+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# install other dependencies
cd ManiSkill_evaluation && pip install -e . && cd ..
cd SimplerEnv && pip install -e . && cd ..

pip install coacd

# optional: for ubuntu 2204
# sudo apt-get install libglvnd-dev
```

then edit ManiSkill_evaluation/mani_skill/examples/real2sim_3d_assets/__init\__.py

to absolute path

ManiSkill_evaluation/policy_evaluation_3d_assets/policy_evaluation_3d_assets/pick_obj

ManiSkill_evaluation/policy_evaluation_3d_assets/policy_evaluation_3d_assets/container


## Train

```bash
./run.sh
```
