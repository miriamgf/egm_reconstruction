import sys

sys.path.append("../Code")
import json
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau
from optuna.integration import TFKerasPruningCallback
from keras.optimizers import Adam
import numpy as np
import random
import subprocess
import psutil
import os
import gc


from models.multioutput import MultiOutput
from models.multioutput_skip import MultiOutput_skip
from models.multioutput_VAE import MultiOutput_VAE, SamplingLayer
from models.multioutput_VAE_skip import MultiOutput_VAE_skip
from models.multioutput_VAE_reduced import MultiOutput_VAE_Reduced
from models.gen_vae import Gen_VAE
from models.attention import BahdanauAttention, LuongAttention


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
        self.gpu_monitor_process=None

        self.SEED=self.params["seed"]
        random.seed(self.SEED)        
        np.random.seed(self.SEED)   
        tf.random.set_seed(self.SEED)
  
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


        #print("Using GPU:", self.params["set_gpu"])
        physical_devices = tf.config.experimental.list_physical_devices('GPU')

        #Log to monitor nvidia-smi 
        #log_gpu=self.monitor_gpu_log()

        #set dynamic usage of gpu
        #try:
            #with tf.device(f"/GPU:{self.params['set_gpu']}"):
                #tf.config.experimental.set_memory_growth(physical_devices[self.params['set_gpu']], True)
                #pass
        #except:
            #with tf.device(f"/GPU:{0}"):
                #tf.config.experimental.set_memory_growth(physical_devices[0], True)
                #pass
        
        # Obtener el uso de memoria antes de cargar los datos
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / (1024 * 1024)  # en MB
        print(f"Uso de memoria antes de cargar los datos: {mem_before:.2f} MB")

        # Callbacks
        callbacks_list, optimizer = self.define_callbacks(x_train)
        
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

        elif self.params["algorithm"] == "OMAMI_ski":

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
            try:
                if self.params["parallel_scope"]:
                    strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])
                    model.build(input_shape=(None, *x_train.shape[1:]))  
                    model.compile(optimizer=tf.keras.optimizers.Adam(clipvalue=1.0))

            except:
                print('Could not use mirror strategy')
                pass
            print(model.model.summary())

            # Compile the model
            model.compile(optimizer=tf.keras.optimizers.Adam(clipvalue=1.0))
        
        elif self.params["algorithm"] == "OMAMI_VAE_Reduced":

            # Create an instance of your model
            model = MultiOutput_VAE_Reduced(
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

        '''
        try:
            print(model.model.summary())
        except:
            print(model.summary())
        '''

        print("Compilando modelo...")
        process = psutil.Process(os.getpid())
        mem_after_compiling = process.memory_info().rss / (1024 * 1024)  # en MB
        print(f"Uso de memoria tras compilar modelos: {mem_after_compiling:.2f} MB")
        '''
        #converto to tensor
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_val = np.array(x_val)
        y_val = np.array(y_val)
        '''
        # Debug: try generator
   

        def train_generator():
            for x, y in zip(x_train, y_train):
                x_batch = np.expand_dims(x, axis=0)  # (1, 400, 12, 32, 1)
                y_batch = np.expand_dims(y, axis=0)  # (1, 400, 2048)
                yield x_batch, (x_batch, y_batch)

        train_dataset = tf.data.Dataset.from_generator(
            train_generator,
            output_signature=(
                tf.TensorSpec(shape=(1, self.params["batch_size"], 12, 32, 1), dtype=tf.float32),
                (
                    tf.TensorSpec(shape=(1, self.params["batch_size"], 12, 32, 1), dtype=tf.float32),
                    tf.TensorSpec(shape=(1, self.params["batch_size"], 2048), dtype=tf.float32)
                )
            )
        )
        def val_generator():
            for x, y in zip(x_val, y_val):
                yield np.expand_dims(x, 0), (np.expand_dims(x, 0), np.expand_dims(y, 0))

        val_dataset = tf.data.Dataset.from_generator(
            val_generator,
            output_signature=(
                tf.TensorSpec(shape=(1, self.params["batch_size"], 12, 32, 1), dtype=tf.float32),
                (
                    tf.TensorSpec(shape=(1, self.params["batch_size"], 12, 32, 1), dtype=tf.float32),
                    tf.TensorSpec(shape=(1, self.params["batch_size"], 2048), dtype=tf.float32)
                )
            )
        )


        #Convert to tf.Dataset format
        '''
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, (x_train, y_train))) \
            .batch(self.params["num_batch_iter"], drop_remainder=False) \
            .prefetch(tf.data.AUTOTUNE)  # Precarga automáticamente
 
        val_dataset = tf.data.Dataset.from_tensor_slices((x_val, (x_val, y_val))) \
            .batch(self.params["num_batch_iter"], drop_remainder=False)  \
            .prefetch(tf.data.AUTOTUNE)  # Precarga automáticamente
        '''

        try:
            model.build(input_shape=(None, *x_train.shape[1:]))  
            print('Modelo construido con éxito.')
        except Exception as e:
            print(f'Error al construir el modelo: {e}')
        
 

        mem_after = process.memory_info().rss / (1024 * 1024)  # en MB
        print(f"Uso de memoria después de cargar los datos: {mem_after:.2f} MB")        
        
        # Entrenar el modelo con el dataset
        self.history = model.fit(
            train_dataset,
            epochs=self.params["n_epochs"],
            validation_data=val_dataset,
            callbacks=callbacks_list,
        )
        
        '''
        self.history = model.fit(
            x=x_train,
            y=[x_train, y_train],
            batch_size=20,
            epochs=self.params["n_epochs"],
            validation_data=(x_val, [x_val, y_val]),
            callbacks=callbacks_list,
            )    
        '''



        try:
            print('saving model')
            #Save model and history    
            model.save(self.experiment_dir+"/model_weights.h5")
        except:

            try:

                model.model.save(self.experiment_dir+"/model_weights.h5")
                model_loaded = load_model(self.experiment_dir + "/model_weights.h5",
                            custom_objects={'LuongAttention': LuongAttention, 'SamplingLayer': SamplingLayer})
            
            except:

                model.model.save(self.experiment_dir+"/model_weights.h5")
                model_loaded = load_model(self.experiment_dir + "/model_weights.h5",
                            custom_objects={ 'SamplingLayer': SamplingLayer})
        
        history_serializable = {key: np.array(value).astype(float).tolist() for key, value in self.history.history.items()}
        path_history = self.experiment_dir + "history.json"
        with open(path_history, 'w') as json_file:
            json.dump(history_serializable, json_file)

        #Learning curves
        self.plot_train_curves()

        #Close gpu log
        #log_gpu.terminate()
        del x_train, y_train, x_val, y_val
        tf.keras.backend.clear_session()
        gc.collect()

        return model, self.history
    
    def define_callbacks(self, x_train):
         # Callbacks
        
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.experiment_dir+ "model_weights.h5",
            save_weights_only=False,
            verbose=1,
            save_best_only=True,
        )

        number_of_steps = len(x_train) // self.params["num_batch_iter"]  # Steps por época
        decay_steps = number_of_steps * 5  # Reducimos el LR cada 5 épocas (ajustable)
        initial_learning_rate = self.params["learning_rate"]

        #lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            #initial_learning_rate,
            #decay_steps=decay_steps,  
            #decay_rate=0.96,  
            #staircase=True  
        #)
        #print('Applying lr decay every ', decay_steps, ' steps. Start at', initial_learning_rate)

        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6)

        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=self.params["early_stopping_patience"]
        )
        # Optimizer configuration
        if self.params["algorithm"]=="OMAMI_VAE" or self.params["algorithm"]=="OMAMI_VAE_skip" or self.params["algorithm"]=="OMAMI_VAE_Reduced":
            optimizer = Adam(learning_rate=initial_learning_rate, clipvalue=1.0) #probar clipnorm
        else:
            optimizer = Adam(learning_rate=initial_learning_rate)

        tensorboard_callback = TensorBoard(log_dir='output/tensorboard/logs/'+self.params['experiment_name'], histogram_freq=1)

        callbacks_list = [early_stopping_callback, tensorboard_callback, lr_scheduler ]
        print(callbacks_list)
        #.u en terminal LOCAL
        #tensorboard --logdir=output/tensorboard/logs/ en terminal REMOTO

        if self.trial is not None:
            pruning_callback = TFKerasPruningCallback(self.trial, monitor="val_loss")
            callbacks_list.append(pruning_callback)
        
        return callbacks_list, optimizer

    def plot_train_curves(self):
        plt.figure()
        plt.plot(self.history.history["val_loss"], label="Global loss (Validation)")
        plt.plot(
            self.history.history["val_autoencoder_loss"],
            label="Autoencoder loss (Validation)",
        )
        plt.plot(
            self.history.history["val_reconstruction_loss"],
            label="Regressor loss (Validation)",
        )
        plt.plot(self.history.history["loss"], label="Global loss (Train)")
        plt.plot(
            self.history.history["autoencoder_loss"],
            label="Autoencoder loss (Train)",
        )
        plt.plot(
            self.history.history["reconstruction_loss"],
            label="Regressor loss (Train)",
        )
        plt.legend(loc="upper left")
        plt.title("Model Loss During Training and Validation")
        plt.ylabel("Mean Squared Error (MSE)")
        plt.xlabel("Epoch")
        plt.savefig(self.experiment_dir + "Learning_curves.png")
        plt.show()
    
    def monitor_gpu_log(self):

        try:
            log_file = f"{self.experiment_dir}gpu_usage_{self.trial.number}.log"
            with open(log_file, "w") as f:
                self.gpu_monitor_process = subprocess.Popen(["nvidia-smi", "-l", "1"], stdout=f, stderr=f)
        except:
                        
            log_file = f"{self.experiment_dir}gpu_usage.log"
            with open(log_file, "w") as f:
                self.gpu_monitor_process = subprocess.Popen(["nvidia-smi", "-l", "1"], stdout=f, stderr=f)
        
        return self.gpu_monitor_process
    
    def stop_gpu_monitor(self):
        if self.gpu_monitor_process:
            self.gpu_monitor_process.terminate()
            self.gpu_monitor_process = None
            print("GPU monitoring stopped.")


    

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

class InspectBatchCallback(tf.keras.callbacks.Callback):
    def on_train_batch_begin(self, batch, logs=None):
        print(f"Batch {batch} iniciado")
        if batch == 0:  # Solo inspeccionamos el primer batch
            inputs, targets = self.model.input, self.model.targets
            print("Forma de inputs:", [inp.shape for inp in inputs])
            print("Forma de targets:", [tar.shape for tar in targets])