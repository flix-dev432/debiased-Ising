#!/bin/bash
set -e

# Upgrade pip
python3 -m pip install --upgrade pip

# Install WandB and log in
pip install wandb
wandb login

# Run your analysis
cd 2d_ising_est_eidf
python3 autoDML_2dIsing_eidf.py