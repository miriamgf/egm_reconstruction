
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












