# This script was developed by Miguel Ángel Cámara Vázquez and Miriam Gutiérrez Fernández
# """

import sys
sys.path.append('../Code')
import matplotlib.pyplot as plt
from config import TrainConfig_1
from config import DataConfig
from tensorflow.keras.optimizers import Adam
from tools_.preprocessing_network import *
from tools_.tools import *
from tools_.df_mapping import *
import tensorflow as tf
import os
import scipy
import datetime
import time
from evaluate_function import *
from numpy import *
import pickle
from models.multioutput import MultiOutput
import mlflow
import random
import argparse
tf.random.set_seed(42)
import argparse
from tensorflow.keras import backend as K
import mlflow
import tensorflow as tf
import datetime
import time
from tensorflow.keras.models import load_model
from tools_.preprocessing_compression import *
from tools_.load_dataset import LoadDataset
from tools_.preprocess_data import Preprocess_Dataset

#from noise_simulation import *

# %% Path Models
# %% Path Models
current = os.path.dirname(os.path.realpath(__file__))
torsos_dir = "../../../Labeled_torsos/"
#directory = "/home/profes/miriamgf/tesis/Autoencoders/Data/"
torsos_dir = "/home/profes/miriamgf/tesis/Autoencoders/Labeled_torsos/"

fs = 500


class TrainModel:
    '''
    The TrainModel class fetches loaded and preprocessed dataset and trains with model

    '''
    def __init__(self, x_train, x_test, x_val, y_train, y_test, y_val, models_dir, experiment_dir):
        self.x_train = x_train
        self.x_test=x_test
        self.x_val=x_val
        self.y_train=y_train
        self.y_test=y_test
        self.y_val=y_val
        self.models_dir=models_dir
        self.experiment_dir= experiment_dir
    
    def train_main(self, x_train, x_test, x_val, y_train, y_test, y_val):
                # 1. AUTOENCODER
        print('Training model...')

        optimizer = Adam(learning_rate=TrainConfig_1.learning_rate_1)

        #tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.models_dir + 'regressor' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
                                                        save_weights_only=True,
                                                        verbose=1)
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)

        # Compilar el modelo
        #Configure GPU for paralellism 
        strategy = tf.distribute.MirroredStrategy()


        if TrainConfig_1.parallelism:
            with strategy.scope():
                model = MultiOutput().assemble_full_model(input_shape=x_train.shape[1:], n_nodes=y_train.shape[-1])
                model.compile(optimizer='adam',
                            loss=['mean_squared_error', 'mean_squared_error'], metrics=['mean_absolute_error'], 
                                loss_weights=[1.0, 5.0])

                print(model.summary())
                history = model.fit(x=x_train, y=[x_train, y_train], batch_size=1, epochs=TrainConfig_1.n_epoch_1,
                                            validation_data=(x_val, [x_val, y_val]),
                                            callbacks=[early_stopping_callback, cp_callback])
        else:
            if TrainConfig_1.inference_pretrained_model:
                model = load_model('/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/experiments/experiments_CINC/20240827-112359_EXP_0/model_mo.h5')

            else:
                model = MultiOutput().assemble_full_model(input_shape=x_train.shape[1:], n_nodes=y_train.shape[-1])
                model.compile(optimizer='adam', loss=['mean_squared_error', 'mean_squared_error'], loss_weights=[1.0, 5.0], metrics=['mean_absolute_error'])
                print(model.summary())
                history = model.fit(x=x_train, y=[x_train, y_train], batch_size=1, epochs=TrainConfig_1.n_epoch_1,
                                            validation_data=(x_val, [x_val, y_val]),
                                            callbacks=[early_stopping_callback, cp_callback])

                # summarize history for loss
                plt.figure()
                plt.plot(history.history['val_loss'], label = 'Global loss (Validation)')
                plt.plot(history.history['val_Autoencoder_output_loss'], label = 'Autoencoder loss (Validation)' )
                plt.plot(history.history['val_Regressor_output_loss'], label = 'Regressor loss (Validation)')
                plt.plot(history.history['loss'], label = 'Global loss (Train)')
                plt.plot(history.history['Autoencoder_output_loss'], label = 'Autoencoder loss (Train)' )
                plt.plot(history.history['Regressor_output_loss'], label = 'Regressor loss (Train)')
                plt.legend( loc='upper left')
                plt.title('model loss')
                plt.ylabel('MSE')
                plt.xlabel('epoch')
                plt.title('Training and validation curves ')
                plt.savefig(self.experiment_dir + 'Learning_curves.png')
                plt.show()

        return model, history
    
    def __call__(self, verbose=False, all=False):
        """Calls the Load class.

   
        """
        return self.train_main(x_train = self.x_train,
                                x_test = self.x_test,
                                x_val= self.x_val, 
                                y_train= self.y_train, 
                                y_test=self.y_test,
                                y_val=self.y_val)


