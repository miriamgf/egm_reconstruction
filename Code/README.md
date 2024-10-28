
# Project Setup Guide

This is a guide to setting up the project repository and installing dependencies.

## Steps to Set Up the Repository Locally

### 1. **Create a Conda Environment with Python 3.10.13**
   Open a terminal and run the following command:

   conda create --name my_env python=3.10.13

### 2. **Activate the Environment**
After the environment is created, activate it with:

   conda activate my_env
   
### 3. **Ensure `pip` is Installed (If Not Already)**
If `pip` is not installed in your environment, install it using:

conda install pip


### 4. **Run `setup.py` to Install the Package**
Install the package and dependencies by running:

python setup.py install


> **Note:** All required dependencies will be installed automatically. However, there may be additional dependencies that need to be installed manually depending on the project.

## Additional Information
- **Be aware:** Some packages may require manual installation. Check for any errors during installation and resolve them as needed.

## Project tree

.
├── Figures --------------------------------------------------- Save here any figures you generate
├── README.md 
├── models ---------------------------------------------------- Folder with AI models backbones
│   ├── __init__.py
│   ├── autoencoder_model.py 
│   ├── autoencoder_model_2D.py
│   ├── config.py
│   ├── multioutput.py
│   ├── reconstruction_model.py
│   └── reconstruction_model_2D.py
├── noise_experiment_launcher.sh  ----------------------------- Launcher for automatic experiments
├── noise_experiment_launcher_tk.sh --------------------------- Launcher for Tikhonov automatic experiments
├── output ---------------------------------------------------- Save here any figures/ models you generate
│   └── figures
│   └── model
├── scripts --------------------------------------------------- Scripts that run the pipelines
│   ├── Model3.py
│   ├── Tikhonov
│   │   ├── Tikhonov_allmodels.ipynb
│   │   ├── Tikhonov_interactive.ipynb
│   │   ├── add_white_noise.py
│   │   ├── data_load.py
│   │   ├── filtering.py
│   │   ├── forward_inverse_problem.py
│   │   ├── freq_phase_analysis.py
│   │   ├── main.py ------------------------------------------- This script runs the main pipeline for ZOT
│   │   ├── metrics.py
│   │   ├── precompute_matrix.py
│   │   ├── tools.py
│   │   └── tools_tikhonov.py
│   ├── __init__.py
│   ├── analysis_results.py
│   ├── config.py ---------------------------------------------- file that configures paths and basic parameters 
│   ├── data_analysis.py
│   ├── evaluate_function.py
│   ├── inference_validation.py
│   ├── src 
│   │   └── training
│   │       └── optuna_opt.py
│   ├── train.py
│   ├── train_multioutput.py ----------------------------------- This file contains main pipeline to launch Training 
│   ├── train_multioutput_optuna.py ---------------------------- Work in progress
│   ├── visualization
│   │   ├── __init__.py
│   │   ├── plot_by_models.ipynb
│   │   ├── vis_functions.py
│   │   ├── visualization.py
│   │   └── visualize_EGMS.ipynb
│   ├── visualize_EGMS.ipynb
├── setup.py
├── tools ------------------------------------------------------ tools called during pipeline
│   ├── __init__.py
│   ├── add_white_noise.py
│   ├── df_mapping.py
│   ├── freq_phase_analysis.py
│   ├── generators.py
│   ├── load_dataset.py
│   ├── noise_models
│   │   ├── bw.dat
│   │   ├── em.dat
│   │   ├── em.hea
│   │   ├── ma.dat
│   ├── noise_simulation.py
│   ├── oclusion.py
│   ├── plots.py
│   ├── preprocess_data.py
│   ├── preprocessing_compression.py
│   ├── preprocessing_network.py
│   ├── tools.py
│   ├── tools_1.py
│   ├── tools_inference.py
│   └── train_model.py











