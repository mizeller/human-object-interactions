#!/bin/bash

# setup the `hloc` conda env

eval "$(conda shell.bash hook)"

conda create -n hloc python=3.8.1 -y
conda activate hloc
set -e
cd submodules/hloc

pip install -e .
pip install trimesh
pip install joblib
pip install loguru
pip install imageio imageio-ffmpeg
pip uninstall pycolmap
pip install pycolmap==0.4.0 -y
cd ../../
