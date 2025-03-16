#!/bin/bash

ENV_NAME="new_env"

# create conda environment and install dependencies requirements.txt
echo "Creating conda environment: $ENV_NAME"
conda create --name $ENV_NAME python==3.9.21

# activate environment
conda activate $ENV_NAME

# Instalar paquetes de pip
pip install --upgrade pip
echo "Installing dependencies"
pip install -r pip_requirements.txt

echo "Successfully created $ENV_NAME environment!"
