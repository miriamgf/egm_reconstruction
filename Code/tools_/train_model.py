import sys
sys.path.append("../Code")
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
from models.multioutput_VAE import MultiOutput_VAE
from models.multioutput_skip import MultiOutput_skip
from models.multioutput_VAE_tf import MultiOutput_VAE_TF
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

class TrainModel:
    """
    A class to manage the training of a multi-output model, including the loading of preprocessed datasets, 
    model initialization, training, and validation. The class handles model saving, callbacks, and can also 
    load a pretrained model for inference.

    Attributes:
    -----------
    params : dict
        Dictionary containing configuration parameters such as learning rate, number of epochs, and parallelism settings.
    x_train : numpy.array
        Training data inputs.
    x_test : numpy.array
        Test data inputs.
    x_val : numpy.array
        Validation data inputs.
    y_train : numpy.array
        Training data labels (outputs).
    y_test : numpy.array
        Test data labels (outputs).
    y_val : numpy.array
        Validation data labels (outputs).
    models_dir : str
        Directory path where trained models will be saved.
    experiment_dir : str
        Directory path where experiment outputs like learning curves will be stored.

    Methods:
    --------
    train_main(self, x_train, x_test, x_val, y_train, y_test, y_val) -> (Model, History)
        Trains a model using the provided data and configuration. Supports multi-GPU parallelism and includes 
        callback functions for early stopping and model checkpointing.

    __call__(self, verbose=False, all=False) -> (Model, History)
        Calls the training process by invoking the `train_main` method and returns the trained model and its history.
    """

    def __init__(
        self, params, x_train, x_test, x_val, y_train, y_test, y_val, models_dir, experiment_dir
    ):
        """
        Initializes the TrainModel class with the necessary data and configuration.

        Args:
        -----
        params : dict
            Dictionary of model configuration parameters including the learning rate, 
            number of epochs, and whether to enable parallelism.
        x_train : numpy.array
            Input data for training the model.
        x_test : numpy.array
            Input data for testing the model.
        x_val : numpy.array
            Input data for validating the model.
        y_train : numpy.array
            Output labels for training.
        y_test : numpy.array
            Output labels for testing.
        y_val : numpy.array
            Output labels for validation.
        models_dir : str
            Path to the directory where models will be saved during training.
        experiment_dir : str
            Path to the directory where outputs like training curves and logs will be saved.
        """
        self.params = params
        self.x_train = x_train
        self.x_test = x_test
        self.x_val = x_val
        self.y_train = y_train
        self.y_test = y_test
        self.y_val = y_val
        self.models_dir = models_dir
        self.experiment_dir = experiment_dir
    

    def train_main(self, x_train, x_test, x_val, y_train, y_test, y_val):
        """
        Executes the model training process.

        This method trains a multi-output model, with the ability to perform training either in a parallel setting 
        (using multiple GPUs) or in a standard setting. It also includes callbacks for early stopping and checkpointing.

        Args:
        -----
        x_train : numpy.array
            Training input data.
        x_test : numpy.array
            Testing input data.
        x_val : numpy.array
            Validation input data.
        y_train : numpy.array
            Training output labels.
        y_test : numpy.array
            Testing output labels.
        y_val : numpy.array
            Validation output labels.

        Returns:
        --------
        model : tensorflow.keras.Model
            The trained Keras model.
        history : tensorflow.keras.callbacks.History
            The history object generated during model training, containing metrics like loss and accuracy.
        """
        print("Training model...")

        # Optimizer configuration
        optimizer = Adam(learning_rate=self.params['learning_rate'])

        # Callbacks
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.models_dir
            + "regressor"
            + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
            save_weights_only=True,
            verbose=1,
        )
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=50
        )

       
        #Choose algorithm {OMAMI, OMAMI_VAE, OMAMI_ski, OMAMI_VAE_ski}
        if self.params["algorithm"] == "OMAMI":

            model = MultiOutput(params = self.params).assemble_full_model(
                input_shape=x_train.shape[1:], n_nodes=y_train.shape[-1]
            )
            loss=["mean_squared_error", "mean_squared_error"]

        elif self.params["algorithm"] == "OMAMI_VAE":

            '''
            multi_output_model = MultiOutput_VAE(self.params)
            model = multi_output_model.assemble_full_model(input_shape=x_train.shape[1:], n_nodes=y_train.shape[-1])

            # Compilar el modelo
            model.compile(optimizer='adam', 
            loss=[multi_output_model.vae_loss(), 'mse'])
            '''

            MultiOutput_VAE_TF_model = MultiOutput_VAE_TF(self.params, input_shape_=x_train.shape[1:], n_nodes=y_train.shape[-1])

            # Ensambla el modelo completo
            model = MultiOutput_VAE_TF_model.assemble_full_model(input_shape=x_train.shape[1:], n_nodes=y_train.shape[-1])

            # Compila el modelo ensamblado
            model.compile(optimizer=tf.keras.optimizers.Adam())

            #model = MultiOutput_VAE_TF_model.assemble_full_model(input_shape=x_train.shape[1:], n_nodes=y_train.shape[-1])
            #model.compile(optimizer=tf.keras.optimizers.Adam())#, loss=MultiOutput_VAE_TF_model.vae_loss())

        elif self.params["algorithm"] == "OMAMI_skip":
            model = MultiOutput_skip(params = self.params).assemble_full_model(
                input_shape=x_train.shape[1:], n_nodes=y_train.shape[-1]
            )
            loss=["mean_squared_error", "mean_squared_error"]


        print(model.summary())

        # Train the model
        history = model.fit(
            x=x_train,
            y=[x_train, y_train],
            batch_size=1,
            epochs=self.params['n_epochs'],
            validation_data=(x_val, [x_val, y_val]),
            callbacks=[early_stopping_callback, cp_callback],
        )

        # Plot and save training and validation curves
        plt.figure()
        plt.plot(history.history["val_loss"], label="Global loss (Validation)")
        plt.plot(
            history.history["val_Autoencoder_output_loss"],
            label="Autoencoder loss (Validation)",
        )
        plt.plot(
            history.history["val_Regressor_output_loss"],
            label="Regressor loss (Validation)",
        )
        plt.plot(history.history["loss"], label="Global loss (Train)")
        plt.plot(
            history.history["Autoencoder_output_loss"],
            label="Autoencoder loss (Train)",
        )
        plt.plot(
            history.history["Regressor_output_loss"],
            label="Regressor loss (Train)",
        )
        plt.legend(loc="upper left")
        plt.title("Model Loss During Training and Validation")
        plt.ylabel("Mean Squared Error (MSE)")
        plt.xlabel("Epoch")
        plt.savefig(self.experiment_dir + "Learning_curves.png")
        plt.show()

        return model, history

    def train_main_parallelism(self, x_train, x_test, x_val, y_train, y_test, y_val):
        """
        Executes the model training process.

        This method trains a multi-output model, with the ability to perform training either in a parallel setting 
        (using multiple GPUs) or in a standard setting. It also includes callbacks for early stopping and checkpointing.

        Args:
        -----
        x_train : numpy.array
            Training input data.
        x_test : numpy.array
            Testing input data.
        x_val : numpy.array
            Validation input data.
        y_train : numpy.array
            Training output labels.
        y_test : numpy.array
            Testing output labels.
        y_val : numpy.array
            Validation output labels.

        Returns:
        --------
        model : tensorflow.keras.Model
            The trained Keras model.
        history : tensorflow.keras.callbacks.History
            The history object generated during model training, containing metrics like loss and accuracy.
        """
        print("Training model...")

        # Optimizer configuration
        optimizer = Adam(learning_rate=self.params['learning_rate'])

        # Callbacks
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.models_dir
            + "regressor"
            + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
            save_weights_only=True,
            verbose=1,
        )
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=50
        )

        # Strategy for multi-GPU parallelism
        strategy = tf.distribute.MirroredStrategy()

        # Training logic for parallel or single-GPU setting
        if self.params['parallelism']:
            with strategy.scope():
                # Assemble the multi-output model
                model = MultiOutput(params = self.params).assemble_full_model(
                    input_shape=x_train.shape[1:], n_nodes=y_train.shape[-1]
                )
                model.compile(
                    optimizer="adam",
                    loss=["mean_squared_error", "mean_squared_error"],
                    metrics=["mean_absolute_error"],
                    loss_weights=[1.0, 5.0],
                )
                print(model.summary())

                # Train the model
                history = model.fit(
                    x=x_train,
                    y=[x_train, y_train],
                    batch_size=1,
                    epochs=self.params['n_epochs'],
                    validation_data=(x_val, [x_val, y_val]),
                    callbacks=[early_stopping_callback, cp_callback],
                )
        else:
            # Load pretrained model for inference or train from scratch
            if self.params['inference_pretrained_model']:
                model = load_model(
                    "/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/experiments/experiments_CINC/20240827-112359_EXP_0/model_mo.h5"
                )
            else:
                # Assemble and compile model
                model = MultiOutput(params = self.params).assemble_full_model(
                    input_shape=x_train.shape[1:], n_nodes=y_train.shape[-1]
                )
                model.compile(
                    optimizer="adam",
                    loss=["mean_squared_error", "mean_squared_error"],
                    loss_weights=[1.0, 5.0],
                    metrics=["mean_absolute_error"],
                )
                print(model.summary())

                # Train the model
                history = model.fit(
                    x=x_train,
                    y=[x_train, y_train],
                    batch_size=1,
                    epochs=self.params['n_epochs'],
                    validation_data=(x_val, [x_val, y_val]),
                    callbacks=[early_stopping_callback, cp_callback],
                )

                # Plot and save training and validation curves
                plt.figure()
                plt.plot(history.history["val_loss"], label="Global loss (Validation)")
                plt.plot(
                    history.history["val_Autoencoder_output_loss"],
                    label="Autoencoder loss (Validation)",
                )
                plt.plot(
                    history.history["val_Regressor_output_loss"],
                    label="Regressor loss (Validation)",
                )
                plt.plot(history.history["loss"], label="Global loss (Train)")
                plt.plot(
                    history.history["Autoencoder_output_loss"],
                    label="Autoencoder loss (Train)",
                )
                plt.plot(
                    history.history["Regressor_output_loss"],
                    label="Regressor loss (Train)",
                )
                plt.legend(loc="upper left")
                plt.title("Model Loss During Training and Validation")
                plt.ylabel("Mean Squared Error (MSE)")
                plt.xlabel("Epoch")
                plt.savefig(self.experiment_dir + "Learning_curves.png")
                plt.show()

        return model, history

    def __call__(self, verbose=False, all=False):
        """
        Executes the model training pipeline when the instance is called.

        Args:
        -----
        verbose : bool, optional
            If set to True, enables detailed output. Default is False.
        all : bool, optional
            If set to True, uses all data. Default is False.

        Returns:
        --------
        model : tensorflow.keras.Model
            The trained model.
        history : tensorflow.keras.callbacks.History
            Training history, containing metrics like loss and accuracy.
        """
        return self.train_main(
            x_train=self.x_train,
            x_test=self.x_test,
            x_val=self.x_val,
            y_train=self.y_train,
            y_test=self.y_test,
            y_val=self.y_val,
        )
