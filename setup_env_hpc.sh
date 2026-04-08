#!/bin/bash

apt-get update
apt-get install -y rustc cargo

module load python/3.11

python -m venv geopixel_venv

source geopixel_venv/bin/activate

pip install -r requirements.txt

pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

pip install deepspeed==0.13.1

pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.6/flash_attn-2.5.6+cu118torch2.1cxx11abiFALSE-cp311-cp311-linux_x86_64.whl 