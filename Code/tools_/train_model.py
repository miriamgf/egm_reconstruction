import sys

sys.path.append("../Code")
import json
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import TensorBoard
from optuna.integration import TFKerasPruningCallback
from keras.optimizers import Adam

from models.multioutput import MultiOutput
from models.multioutput_skip import MultiOutput_skip
from models.multioutput_VAE import MultiOutput_VAE, SamplingLayer
from models.multioutput_VAE_skip import MultiOutput_VAE_skip
from models.gen_vae import Gen_VAE



tf.random.set_seed(42)
import datetime

import tensorflow as tf
from keras.models import load_model

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
        self,
        params,
        x_train,
        x_test,
        x_val,
        y_train,
        y_test,
        y_val,
        models_dir,
        experiment_dir,
        trial = None
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
        trial : optuna.Trial, optional
            An optuna trial object for hyperparameter optimization. Default is None.
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
        self.trial= trial

    @tf.function(jit_compile=False)
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
        model : keras.Model
            The trained Keras model.
        history : keras.callbacks.History
            The history object generated during model training, containing metrics like loss and accuracy.
        """

        print("Training model...")

        if self.params["set_gpu"] is not False:
            print("Using GPU:", self.params["set_gpu"])
            with tf.device(f"/GPU:{self.params['set_gpu']}"):
                # Train on specified GPU
                pass
        else:
            pass

        # Callbacks
        
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.experiment_dir+ "model_weights.h5",
            save_weights_only=False,
            verbose=1,
            save_best_only=True,
        )

        initial_learning_rate = self.params["learning_rate"]
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=1000,
            decay_rate=0.96,
            staircase=True)
        
        # Optimizer configuration
        optimizer = Adam(learning_rate=lr_schedule)
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=20
        )
        
        tensorboard_callback = TensorBoard(log_dir='output/tensorboard/logs/'+self.params['algorithm'], histogram_freq=1)

        callbacks_list = [early_stopping_callback, tensorboard_callback]
        print(callbacks_list)
        #ssh -L 6006:localhost:6006 miriamgf@10.110.100.78 en terminal LOCAL
        #tensorboard --logdir=output/tensorboard/logs/ en terminal REMOTO

        if self.trial is not None:
            pruning_callback = TFKerasPruningCallback(self.trial, monitor="val_loss")
            callbacks_list.append(pruning_callback)

        
        # Choose algorithm {OMAMI, OMAMI_VAE, OMAMI_ski, OMAMI_VAE_ski} 

        if self.params["algorithm"] == "OMAMI":

            model = MultiOutput(params=self.params).assemble_full_model(
                input_shape=x_train.shape[1:], n_nodes=y_train.shape[-1]
            )
            model.compile(
                optimizer=optimizer,
                loss=["mean_squared_error", "mean_squared_error"],
                loss_weights=[self.params["loss_weight_1"], self.params["loss_weight_2"]],
                metrics=["mean_absolute_error"]
                
            )

        if self.params["algorithm"] == "OMAMI_ski":

            model = MultiOutput_skip(params=self.params).assemble_full_model(
                input_shape=x_train.shape[1:], n_nodes=y_train.shape[-1]
            )
            model.compile(
                optimizer=optimizer,
                loss=["mean_squared_error", "mean_squared_error"],
                loss_weights=[self.params["loss_weight_1"], self.params["loss_weight_2"]],
                metrics=["mean_absolute_error"]

            )
        elif self.params["algorithm"] == "OMAMI_VAE":

            # Create an instance of your model
            model = MultiOutput_VAE(
                self.params,
                input_shape_=x_train.shape[1:],
                n_nodes=y_train.shape[-1],
                tensorboard_logs=self.experiment_dir + "tb_logs/",
            )
        
            print(model.model.summary())

            # Compile the model
            model.compile(optimizer=tf.keras.optimizers.Adam(clipvalue=1.0))

        elif self.params["algorithm"] == "OMAMI_VAE_ski":
            # Create an instance of your model
            model = MultiOutput_VAE_skip(
                self.params,
                input_shape_=x_train.shape[1:],
                n_nodes=y_train.shape[-1],
                tensorboard_logs=self.experiment_dir + "tb_logs/",
            )

            print(model.model.summary())

            # Compile the model
            model.compile(optimizer=tf.keras.optimizers.Adam())
        
        elif self.params["algorithm"] == "gen_VAE":

            # Create an instance of your model
            model = Gen_VAE(
                self.params,
                input_shape_=y_train.shape[1:],
                n_nodes=2048,
                tensorboard_logs=self.experiment_dir + "tb_logs/",
            )

            print(model.model.summary())

            # Compile the model
            model.compile(optimizer=tf.keras.optimizers.Adam(clipvalue=1.0)) 
        
        else: 
            print('Error: Model name not identified. Terminating training...')
            sys.exit()


        try:
            print(model.model.summary())
        except:
            print(model.summary())
        
        
        
        history = model.fit(
            x=x_train,
            y=[x_train, y_train],
            batch_size=1,
            epochs=self.params["n_epochs"],
            validation_data=(x_val, [x_val, y_val]),
            callbacks=callbacks_list,
            )    
        # Construir el modelo antes de guardarlo si es un modelo subclasificado
        try:
            model.build(input_shape=(None, *x_train.shape[1:]))  # Define el input shape correcto
            print('Modelo construido con éxito.')
        except Exception as e:
            print(f'Error al construir el modelo: {e}')


        try:
            print('saving model')
            #Save model and history    
            model.save(self.experiment_dir+"/model_weights.h5")
        except:

            model.model.save(self.experiment_dir+"/model_weights.h5")
            model_loaded = load_model(self.experiment_dir + "/model_weights.h5", custom_objects={'SamplingLayer': SamplingLayer})
        with open(self.experiment_dir+'historial.json', 'w') as json_file:
                        json.dump(history.history, json_file)
        # Plot and save training and validation curves
        
        plt.figure()
        plt.plot(history.history["val_loss"], label="Global loss (Validation)")
        plt.plot(
            history.history["val_autoencoder_loss"],
            label="Autoencoder loss (Validation)",
        )
        plt.plot(
            history.history["val_reconstruction_loss"],
            label="Regressor loss (Validation)",
        )
        plt.plot(history.history["loss"], label="Global loss (Train)")
        plt.plot(
            history.history["autoencoder_loss"],
            label="Autoencoder loss (Train)",
        )
        plt.plot(
            history.history["reconstruction_loss"],
            label="Regressor loss (Train)",
        )
        plt.legend(loc="upper left")
        plt.title("Model Loss During Training and Validation")
        plt.ylabel("Mean Squared Error (MSE)")
        plt.xlabel("Epoch")
        plt.savefig(self.experiment_dir + "Learning_curves.png")
        plt.show()
        '''
        except: #VAE
            plt.figure()
            plt.plot(history.history["val_loss"], label="Global loss (Validation)")
            plt.plot(
                history.history["val_autoencoder_loss"],
                label="Autoencoder loss (Validation)",
            )
            plt.plot(
                history.history["val_reconstruction_loss"],
                label="Regressor loss (Validation)",
            )
            plt.plot(history.history["loss"], label="Global loss (Train)")
            plt.plot(
                history.history["autoencoder_loss"],
                label="Autoencoder loss (Train)",
            )
            plt.plot(
                history.history["reconstruction_mse"],
                label="Regressor loss (Train)",
            )
            plt.legend(loc="upper left")
            plt.title("Model Loss During Training and Validation")
            plt.ylabel("Mean Squared Error (MSE)")
            plt.xlabel("Epoch")
            plt.savefig(self.experiment_dir + "Learning_curves.png")
            plt.show()

        '''
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
        model : keras.Model
            The trained model.
        history : keras.callbacks.History
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
