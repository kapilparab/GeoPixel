#!/bin/bash

unzip -q dataset.zip -d dataset/

apt-get update
apt-get install -y rustc cargo

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda
source $HOME/miniconda/bin/activate

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

conda create -n geopixel python=3.10 -y
conda activate geopixel

git clone https://github.com/kapilparab/GeoPixel.git

cd GeoPixel/

pip install -r requirements.txt

pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

pip install deepspeed==0.13.1

pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.6/flash_attn-2.5.6+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

bash finetune.sh