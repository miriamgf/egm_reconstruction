import sys

sys.path.append("../Code")
import datetime
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

from keras.optimizers import Adam

from models.gen_vae_2d import Gen_VAE_2D
from models.gen_vae_2d_v2 import Gen_VAE_2D_v2
from models.gen_vae_2d_skip import Gen_VAE_2D_Skip
from models.gen_vae_3d import Gen_VAE_3D
#from models.gen_cvae_2d import Gen_CondVAE_2D
from models.gen_vae import Gen_VAE

from keras.callbacks import TensorBoard
from keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau
from optuna.integration import TFKerasPruningCallback

tf.random.set_seed(42)
import datetime

import tensorflow as tf
from keras.models import load_model

class TrainModelGen:
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
        self.trial=None

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

        # Optimizer configuration
        optimizer = Adam(learning_rate=self.params["learning_rate"])

        # Callbacks
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.experiment_dir
            + "regressor.weights.h5",
            save_weights_only=True,
            verbose=1,
            save_best_only=True,
        )

        lr_decay = tf.keras.callbacks.LearningRateScheduler(
            lambda epoch: self.params["learning_rate"] * 0.95 ** epoch
        )

        if self.params["algorithm"] == "OMAMI" or self.params["algorithm"] == "OMAMI_ski":
            early_stopping_callback = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=self.params["early_stopping_patience"]
            )
        else:
            early_stopping_callback = tf.keras.callbacks.EarlyStopping(
                monitor="val_total_loss", patience=self.params["early_stopping_patience"]
            )
        tensorboard_callback = TensorBoard(log_dir='output/tensorboard/logs/'+self.params['algorithm'], histogram_freq=1)
        #ssh -L 6006:localhost:6006 miriamgf@10.110.100.78 en terminal LOCAL
        #tensorboard --logdir=output/tensorboard/logs/

        callbacks_list, optimizer = self.define_callbacks(x_train)


        # Choose algorithm {OMAMI, OMAMI_VAE, OMAMI_ski, OMAMI_VAE_ski}

        if self.params["algorithm"] == "gen_VAE":

            # Create an instance of your model
            model = Gen_VAE(
                self.params,
                input_shape_=y_train.shape[1:],
                n_nodes=2048,
                tensorboard_logs=self.experiment_dir + "tb_logs/",
                latent_dim=self.params["latent_dim"]
            )

            print(model.model.summary())

            # Compile the model
            model.compile(optimizer=tf.keras.optimizers.Adam(clipvalue=1.0))
        
        if self.params["algorithm"] == "gen_VAE_2D_v2":
            print('gen_VAE_2D_v2')

    
            model = Gen_VAE_2D_v2(
                self.params,
                input_shape_=y_train.shape[1:],
                n_nodes=2048,
                tensorboard_logs=self.experiment_dir + "tb_logs/",
                latent_dim=self.params["latent_dim"]
            )


            # Compile the model
            model.compile(optimizer=tf.keras.optimizers.Adam(clipvalue=1.0))

            model.build(input_shape=(None,) + y_train.shape[1:])

            print(model.summary())


            self.params["beta_max"] = 8.0
            self.params["warmup_epochs"] = 20

            beta_cb = BetaWarmupEpoch(model)
        
        if self.params["algorithm"] == "gen_VAE_3D":

            # Create an instance of your model
            model = Gen_VAE_3D(
                self.params,
                input_shape_=y_train.shape[1:],
                n_nodes=2048,
                tensorboard_logs=self.experiment_dir + "tb_logs/",
                latent_dim=self.params["latent_dim"]
            )

            print(model.model.summary())

            # Compile the model
            model.compile(optimizer=tf.keras.optimizers.Adam(clipvalue=1.0))
        
        if self.params["algorithm"] == "gen_VAE_2D":

            # Create an instance of your model
            model = Gen_VAE_2D(
                self.params,
                input_shape_=y_train.shape[1:],
                n_nodes=2048,
                tensorboard_logs=self.experiment_dir + "tb_logs/",
                latent_dim=self.params["latent_dim"]
            )

            print(model.model.summary())

            # Compile the model
            model.compile(optimizer=tf.keras.optimizers.Adam(clipvalue=1.0))
        
        if self.params["algorithm"] == "gen_VAE_2D_warmup":

            model = Gen_VAE_2D(
                self.params,
                input_shape_=y_train.shape[1:],
                n_nodes=2048,
                tensorboard_logs=self.experiment_dir + "tb_logs/",
                latent_dim=self.params["latent_dim"]
            )

            print(model.model.summary())

            # Compile the model
            model.compile(optimizer=tf.keras.optimizers.Adam(clipvalue=1.0))

        if self.params["algorithm"] == "gen_condVAE":

            condition_dim=len(self.params["n_clases"])

            model = Gen_CondVAE_2D(
                params={"l2_reg": 1e-5},
                input_shape_=(400, 2048),
                n_nodes=None,
                latent_dim=128,
                condition_dim=condition_dim,
                tensorboard_logs="./logs_cvae"
            )

            print(model.model.summary())

            # Compile the model
            model.compile(optimizer=tf.keras.optimizers.Adam(clipvalue=1.0))


        if self.params["algorithm"] == "gen_VAE_2D_skip":

            # Create an instance of your model
            model = Gen_VAE_2D_Skip(
                self.params,
                input_shape_=y_train.shape[1:],
                n_nodes=2048,
                tensorboard_logs=self.experiment_dir + "tb_logs/",
                latent_dim=self.params["latent_dim"]
            )

            print(model.model.summary())

            # Compile the model
            model.compile(optimizer=tf.keras.optimizers.Adam(clipvalue=1.0))
        


        try:
            print(model.model.summary())
        except:
            print(model.summary())
        
        def train_generator():
            for x in y_train:
                x_batch = np.expand_dims(x, axis=0)  # (1, 400, 2048)
                yield x_batch, x_batch  # input = output

        def val_generator():
            for x in y_train:
                x_batch = np.expand_dims(x, axis=0)  # (1, 400, 2048)
                yield x_batch, x_batch  # input = output

        train_dataset = tf.data.Dataset.from_generator(
            train_generator,
                output_signature=(
                    tf.TensorSpec(shape=(1, self.params["batch_size"], 2048), dtype=tf.float32),
                    tf.TensorSpec(shape=(1, self.params["batch_size"], 2048), dtype=tf.float32)
                )
            )

        val_dataset = tf.data.Dataset.from_generator(
            val_generator,
            output_signature=(
                tf.TensorSpec(shape=(1, self.params["batch_size"], 2048), dtype=tf.float32),
                tf.TensorSpec(shape=(1, self.params["batch_size"], 2048), dtype=tf.float32)
            )
        )

        if self.params["algorithm"] == "OMAMI_gen_VAE_warmup":
            for epoch in range(1, self.params["n_epochs"] + 1):
                new_beta = min(epoch / self.params["beta_warmup_epochs"], 1.0)  # crecimiento lineal
                model.beta.assign(new_beta)
                print(f"[Epoch {epoch}] β = {model.beta.numpy():.3f}")
                
                history = model.fit(
                    train_dataset,
                    validation_data=val_dataset,
                    batch_size=1,
                    epochs=1,
                    callbacks=[early_stopping_callback, cp_callback, tensorboard_callback, lr_decay],
                )
        elif self.params["algorithm"] == "Gen_VAE_2D_v2":
            history = model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=self.params["n_epochs"],
                callbacks=[beta_cb, early_stopping_callback, cp_callback, tensorboard_callback, lr_decay],
            )

        else:
            
            # Train the model
            history = model.fit(
                train_dataset,
                validation_data=val_dataset,
                batch_size=1,
                epochs=self.params["n_epochs"],
                callbacks=[early_stopping_callback, cp_callback, tensorboard_callback, lr_decay],
            )

        try:
            print('saving model')
            #Save model and history    
            model.save(self.experiment_dir+"/model_weights.h5")
        except:

            try:

                model.model.save(self.experiment_dir+"/model_weights.h5")
            
            except:

                model.save_weights(self.experiment_dir + "/model_weights.h5")

                
        # Plot and save training and validation curves
        try:
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
        
        except:
            plt.figure()
            plt.plot(history.history["val_total_loss"], label="Global loss (Validation)")
            plt.plot(
                history.history["val_loss_autoencoder"],
                label="Autoencoder loss (Validation)",
            )
            plt.plot(history.history["total_loss"], label="Global loss (Train)")
            plt.plot(
                history.history["loss_autoencoder"],
                label="Autoencoder loss (Train)",
            )
            
            plt.legend(loc="upper left")
            plt.title("Model Loss During Training and Validation")
            plt.ylabel("Mean Squared Error (MSE)")
            plt.xlabel("Epoch")
            plt.savefig(self.experiment_dir + "Learning_curves.png")
            plt.show()

    
        return model, history
    
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
        #ssh -L 6006:localhost:6006 miriamgf@10.110.100.78 en terminal LOCAL
        #tensorboard --logdir=output/tensorboard/logs/ en terminal REMOTO

        if self.trial is not None:
            pruning_callback = TFKerasPruningCallback(self.trial, monitor="val_loss")
            callbacks_list.append(pruning_callback)
        
        return callbacks_list, optimizer
    
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

class BetaWarmupEpoch(tf.keras.callbacks.Callback):
    def __init__(self, model, beta_max=4.0, warmup_epochs=5):
        super().__init__()
        self.model_ref = model
        self.beta_max = float(beta_max)
        self.warmup_epochs = max(1, int(warmup_epochs))

    def on_epoch_begin(self, epoch, logs=None):
        # epoch empieza en 0; usamos (epoch+1) para que en la 1ª época β > 0
        frac = min((epoch + 1) / self.warmup_epochs, 1.0)
        beta_t = frac * self.beta_max
        self.model_ref.beta.assign(beta_t)
        tf.print("[Epoch", epoch + 1, "] β =", self.model_ref.beta)
