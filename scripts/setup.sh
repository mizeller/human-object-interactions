#!/bin/bash

eval "$(conda shell.bash hook)"

conda create -n handobjectreconstruction python=3.11.9 -y
conda activate handobjectreconstruction

pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

pip install --upgrade setuptools pip
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.0+cu121.html

conda install -c conda-forge suitesparse -y

pip install -r requirements.txt

************************************************************************************************************************************************************************************************************************************

pip install -U xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu121 --no-deps

# https://github.com/facebookresearch/pytorch3d/issues/1842#issuecomment-2701127909
# this successfully installs pytorch3d-0.7.8 
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# NOTE: the sudo-commands can be skipped

# sudo apt-get update
# sudo apt-get -y upgrade
# sudo apt-get -y install wget nano zip git curl libgl1-mesa-glx libglib2.0-0 libvulkan-dev vulkan-tools xorg-dev libxkbcommon-x11-dev
# sudo apt-get -y clean
