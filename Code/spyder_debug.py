#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 16:13:04 2022

@author: mgutierrez
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 13:03:44 2022

@author: mgutierrez
"""

root_logdir = '../Logs/'
data_dir = '../Data'
figs_dir = 'Figs/'
models_dir = '../Models/'

# Tensorboard logs name generator
def get_run_logdir():
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)

import os
import time
import tensorflow as tf
import keras
from keras import models, layers
import tensorflow_addons as tfa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.utils import shuffle
from itertools import product
from tools import *

import pickle
from generators import *

n_classes = 3
X,Y,Y_model, egm_tensor=load_data(data_type='1channelTensor', n_classes=n_classes, subsampling= True, fs_sub=50, norm=False, SR=True, SNR=20)