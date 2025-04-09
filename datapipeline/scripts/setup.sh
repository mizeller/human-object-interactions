#!/bin/bash

eval "$(conda shell.bash hook)"

conda create -n preprocessing python=3.11.9 -y
conda activate preprocessing

pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

pip install --upgrade setuptools pip
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.0+cu121.html

conda install -c conda-forge suitesparse -y

# added 
pip install -r requirements.txt

************************************************************************************************************************************************************************************************************************************

pip install -U xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu121 --no-deps

# NOTE: the sudo-commands can be skipped

# sudo apt-get update
# sudo apt-get -y upgrade
# sudo apt-get -y install wget nano zip git curl libgl1-mesa-glx libglib2.0-0 libvulkan-dev vulkan-tools xorg-dev libxkbcommon-x11-dev
# sudo apt-get -y clean