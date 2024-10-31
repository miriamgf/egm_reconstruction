from setuptools import setup, find_packages

setup(
    name='DEEP_AF_URJC',
    version='0.1',
    packages=find_packages(),
    author='Miriam Gutiérrez',
    url='https://github.com/miriamgf/egm_reconstruction',

    install_requires=[
        'tensorflow==2.15.0',  # Will be installed by pip within the Conda env
        'numpy>=1.23.5,<2.0.0',
        'matplotlib',
        'scikit-learn',
        'optuna',
        'mlflow',
        'scipy',
        'pandas',
        'nvidia-cublas-cu12==12.1.3.1',
        'nvidia-cuda-cupti-cu12==12.1.105',
        'nvidia-cuda-nvrtc-cu12==12.1.105',
        'nvidia-cuda-runtime-cu12==12.1.105',
        'nvidia-cudnn-cu12==8.9.2.26',
        'nvidia-cufft-cu12==11.0.2.54',
        'nvidia-curand-cu12==10.3.2.106',
        'nvidia-cusolver-cu12==11.4.5.107',
        'nvidia-cusparse-cu12==12.1.0.106',
        'nvidia-nccl-cu12==2.19.3',
        'nvidia-nvjitlink-cu12==12.4.99',
        'nvidia-nvtx-cu12==12.1.105',
        'kymatio' #wavelet scattering module python
        #'scikit-cuda' #for kymatio WS gpu
        #'cupy' #for kymatio WS gpupipi
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
