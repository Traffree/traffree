#!/bin/bash --login

# Stop execution if something fails
set -e

CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh

# creates the conda environment
PROJECT_DIR=$PWD
conda env create --prefix $PROJECT_DIR/env --file $PROJECT_DIR/environment.yml --force

# activate the conda env before installing via pip
conda activate $PROJECT_DIR/env

# PyTorch Geometric
TORCH=1.7.0
CUDA=cu110
python -m pip install torch-scatter --no-cache-dir --no-index --find-links https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
python -m pip install torch-sparse --no-cache-dir --no-index --find-links https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
python -m pip install torch-cluster --no-cache-dir --no-index --find-links https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
python -m pip install torch-spline-conv --no-cache-dir --no-index --find-links https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
python -m pip install torch-geometric --no-cache-dir

# Sumo
sudo apt-get install sumo sumo-tools sumo-doc

python -m pip install traci
python -m pip install neat-python


