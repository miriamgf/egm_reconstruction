from setuptools import setup, find_packages

setup(
    name='DEEP_AF_URJC',
    version='0.1',
    packages=find_packages(),
    author= 'Miriam Gutiérrez',
    url='https://github.com/miriamgf/egm_reconstruction',       

    install_requires=[
        'tensorflow == 2.15.0',  # Will be installed by pip within the Conda environment
        'numpy>=1.23.5,<2.0.0',
        'matplotlib',
        'scikit-learn',
        'optuna',
        'mlflow',
        'scipy',
        'pandas'

    ],
    extras_require={
        'dev': [
            'pytest',  # Development dependencies
            'black',  # Code formatting tool
            'pickle'
        ]
    },
    include_package_data=True,
    python_requires='>=3.10.13',    
   
)
