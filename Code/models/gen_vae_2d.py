import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import layers
from keras import Model

tf.config.experimental_run_functions_eagerly(
    True
)  # Para versiones anteriores de TensorFlow 2.x
tf.executing_eagerly()


class Gen_VAE_2D(Model):
    """
    Used to generate synthetic EGMs using a trained VAE model.
    """

    def __init__(
        self, params, input_shape_, n_nodes, latent_dim=128, tensorboard_logs=None
    ):
        super(Gen_VAE_2D, self).__init__()

        if not os.path.exists(tensorboard_logs):
            os.makedirs(tensorboard_logs)

        self.input_shape_ = input_shape_
        self.params = params
        self.SEED=50
        tf.random.set_seed(self.SEED)
        self.tensorboard_logs = tensorboard_logs
        self.file_writer = tf.summary.create_file_writer(self.tensorboard_logs)


        # Define the layers as instance attributes
        self.initializer = tf.keras.initializers.HeNormal()
        self.latent_dim = latent_dim

        self.z_mean_dense = layers.Dense(latent_dim, name="z_mean")
        self.z_log_var_dense = layers.Dense(latent_dim, name="z_log_var")
        self.sampling_layer = SamplingLayer()
        #self.reshape_latent_space = layers.Reshape((self.params["batch_size"], 10, 10))
     
        self.model, self.latent_space = self.assemble_full_model(input_shape_, n_nodes)
        self.beta = tf.Variable(0.0, trainable=False, dtype=tf.float32)  # Inicialmente 0 para el warm-up


    def call(self, inputs, training=None, mask=None):

        # x = inputs[0]  # Assuming inputs is a tuple and you only need the first element
        outputs = self.model(inputs, training=training)
        return outputs
    
    def build_encoder_module(self, inputs, input_shape):
        """
        Encoder usando Conv2D para capturar patrones espacio-temporales.
        Se espera que inputs tenga shape (batch, 400, 2048), que se reinterpreta como (400, 32, 64).
        """
        batch_size = tf.shape(inputs)[0]

        # Reorganiza de (batch, 400, 2048) → (batch, 400, 32, 64, 1)
        x = layers.Reshape((400, 2048, 1))(inputs)  

        x = layers.Conv2D(
            filters=32,
            kernel_size=(5, 15),
            strides=(2, 4),
            padding='same',
            activation='leaky_relu',
            kernel_initializer=self.initializer,
            kernel_regularizer=tf.keras.regularizers.l2(self.params["l2_reg"]),
        )(x)

        x = layers.Conv2D(
            filters=64,
            kernel_size=(5, 15),
            strides=(2, 4),
            padding='same',
            activation='leaky_relu',
            kernel_regularizer=tf.keras.regularizers.l2(self.params["l2_reg"]),
        )(x)

        x = layers.Conv2D(
            filters=10,
            kernel_size=(3, 5),
            strides=(2, 2),
            padding='same',
            activation='leaky_relu',
            kernel_regularizer=tf.keras.regularizers.l2(self.params["l2_reg"]),
        )(x)

        x = layers.Flatten()(x)
        x = layers.Dense(self.latent_dim, activation='leaky_relu')(x)

        z_mean = self.z_mean_dense(x)
        z_log_var = self.z_log_var_dense(x)
        z = self.sampling_layer([z_mean, z_log_var])
        #z = z + tf.random.normal(tf.shape(z), stddev=0.05)

        return z, z_mean, z_log_var


    def build_decoder_module(self):
        """
        Build decoder model from latent vector z to reconstructed signal.
        """
        latent_inputs = layers.Input(shape=(self.latent_dim,), name="decoder_input")

        x = layers.Dense(25 * 16 * 32)(latent_inputs)
        x = layers.LeakyReLU()(x)
        x = layers.Reshape((25, 16, 32))(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Conv2DTranspose(64, (4, 4), strides=(2, 4), padding='same')(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Conv2DTranspose(32, (4, 4), strides=(2, 4), padding='same')(x)
        x = layers.LeakyReLU()(x)
        x= layers.Dropout(0.2)(x)
        x = layers.Conv2DTranspose(32, (4, 4), strides=(2, 2), padding='same')(x)
        x = layers.LeakyReLU()(x)

        x = layers.Conv2DTranspose(1, (4, 4), strides=(2, 4), padding='same')(x)
        x = tf.keras.activations.tanh(x)
        x = tf.squeeze(x, axis=-1)

        self.decoder = tf.keras.Model(latent_inputs, x, name="decoder")

        return self.decoder

    def decode_from_latent(self, z):
        if not hasattr(self, 'decoder'):
            self.build_decoder_module()
        return self.decoder(z)

    def build_autoencoder_branch(self, inputs, input_shape):
        '''
        Build assamble between encoder and decoder
        '''

        z, z_mean, z_log_var = self.build_encoder_module(inputs, input_shape)

        # Usa el modelo decoder entrenable
        if not hasattr(self, 'decoder'):
            self.build_decoder_module()

        decoded_output = self.decoder(z)
        return z_mean, z_log_var, z, decoded_output

    def assemble_full_model(self, input_shape, n_nodes):
        """
        Used to assemble our multi-output model CNN.
        """
        inputs = layers.Input(shape=input_shape)
        z_mean, z_log_var, latent_space, autoencoder_output = (
            self.build_autoencoder_branch(inputs, input_shape)
        )
       
        model = Model(
            inputs=inputs,
            outputs=autoencoder_output,
            name="MultiOutput",
        )
        return model, latent_space

    def log_normal_pdf(self, sample, mean, logvar, raxis=1):
        log2pi = tf.math.log(2.0 * np.pi)
        return tf.reduce_sum(
            -0.5 * ((sample - mean) ** 2.0 * tf.exp(-logvar) + logvar + log2pi),
            axis=raxis,
        )

    def compute_loss_VAE(self, y_pred, y_true, z_mean, z_log_var):

        mse_loss = tf.reduce_mean(tf.reduce_sum(tf.keras.losses.mean_squared_error(y_true, y_pred), axis=[1]))

        grad_loss = self.temporal_gradient_loss(y_true, y_pred)
        corr_loss=self.negative_corr_loss(y_true, y_pred)
        logvar = tf.clip_by_value(z_log_var, -10.0, 10.0)
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(logvar + 1e-8))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

        beta = self.beta

        total_loss = mse_loss + beta * kl_loss + 0.5 * corr_loss + 0.3*grad_loss
        return total_loss, mse_loss, kl_loss

    def train_step(self, data):
        """
        Define train step
        """

        # Unpack data
        x, y_true_autoencoder = data

        with tf.GradientTape() as tape:

            y_pred_autoencoder = self.model(x, training=True)

            _, z_mean, z_log_var, _ = self.build_autoencoder_branch(x, self.input_shape_)


            total_loss, mse_autoencoder, kl_loss = self.compute_loss_VAE(
                y_pred_autoencoder, y_true_autoencoder, z_mean, z_log_var
            )

            # Total loss (you can adjust weights if one branch's loss should have more influence)
            total_loss = total_loss  
        # Compute and apply gradients based on the total loss

        gradients = tape.gradient(total_loss, self.trainable_variables)
        # Clip gradients to avoid exploding gradients (based on their global norm)
        clipped_gradients, global_norm = tf.clip_by_global_norm(
            gradients, clip_norm=5.0
        )
        self.optimizer.apply_gradients(zip(clipped_gradients, self.trainable_variables))

        step = int(self.optimizer.iterations)
        with self.file_writer.as_default():
            tf.summary.scalar("loss/total_loss", total_loss, step=step)
            tf.summary.scalar("loss/loss_autoencoder", total_loss  , step=step)
            tf.summary.scalar("loss/mse_autoencoder", mse_autoencoder, step=step)
            tf.summary.scalar("loss/kl_loss", kl_loss, step=step)
        
        tf.print("KL loss:", kl_loss)


            # self.log_latent_space_image(latent_space, step)

        return {
            "total_loss": total_loss,
            "loss_autoencoder": total_loss  ,
            "mse_autoencoder": mse_autoencoder,
            "kl_loss": kl_loss  

        }

    def test_step(self, data):
        """
        Define test step (used for validation).
        """
        # Unpack data
        x, y_true_autoencoder = data

        # Forward pass through the model without gradient tracking
        y_pred_autoencoder = self.model(x, training=False)
        _, z_mean, z_log_var, _ = self.build_autoencoder_branch(x, self.input_shape_)

        total_loss, mse_autoencoder, kl_loss = self.compute_loss_VAE(
            y_pred_autoencoder, y_true_autoencoder, z_mean, z_log_var
        )
        

        # Return losses for tracking
        return {
            "total_loss": total_loss,
            "loss_autoencoder": total_loss,
            "mse_autoencoder": mse_autoencoder,
            "kl_loss": kl_loss  

        }

    def log_latent_space_image(self, latent_space, step):
        # Reshape and reduce dimensionality to 2D for visualization
        latent_space_projection = tf.reshape(latent_space, (-1, latent_space.shape[1]))
        latent_space_flat_image = latent_space[0, 0, :, :, 0]
        latent_space_flat_signal = latent_space[0, :, 0, 0, 0:5]

        # Plot the 2D latent space
        plt.figure(figsize=(6, 6))
        plt.scatter(
            latent_space_projection[:, 0], latent_space_projection[:, 1], alpha=0.5
        )
        plt.title(f"Latent Space at Step {step}")
        plt.savefig(f"{self.tensorboard_logs}latent_space_proj_{step}.png")
        plt.close()

        with self.file_writer.as_default():
            image = tf.io.read_file(
                f"{self.tensorboard_logs}latent_space_proj_{step}.png"
            )
            image = tf.image.decode_image(image)
            image = tf.expand_dims(image, axis=0)
            tf.summary.image("Latent Space - 2D projection", image, step=step)

        # Plot the 2D latent space
        plt.figure(figsize=(6, 6))
        plt.imshow(latent_space_flat_image)
        plt.title(f"Latent Space at Step {step}")
        plt.savefig(f"{self.tensorboard_logs}latent_space_img_{step}.png")
        plt.close()

        with self.file_writer.as_default():
            image = tf.io.read_file(
                f"{self.tensorboard_logs}latent_space_img_{step}.png"
            )
            image = tf.image.decode_image(image)
            image = tf.expand_dims(image, axis=0)
            tf.summary.image("Latent Space - Image First channel", image, step=step)

        # Plot the 1D latent space
        plt.figure(figsize=(12, 6))
        for channel in range(0, 4):
            plt.plot(latent_space_flat_signal[:, channel])
        plt.title(f"Latent Space at Step {step}")
        plt.savefig(f"{self.tensorboard_logs}latent_space_sig_{step}.png")
        plt.close()

        with self.file_writer.as_default():
            image = tf.io.read_file(
                f"{self.tensorboard_logs}latent_space_sig_{step}.png"
            )
            image = tf.image.decode_image(image)
            image = tf.expand_dims(image, axis=0)
            tf.summary.image("Latent Space - Signal (channels 0-5)", image, step=step)

    def negative_corr_loss(self, y_true, y_pred):
        # Centramos las señales: restamos la media por canal
        mu_x = tf.reduce_mean(y_true, axis=1, keepdims=True)  # (batch, 1, channels)
        mu_y = tf.reduce_mean(y_pred, axis=1, keepdims=True)

        x = y_true - mu_x  # (batch, time, channels)
        y = y_pred - mu_y

        # Numerador de la correlación: producto punto temporal
        num = tf.reduce_sum(x * y, axis=1)  # (batch, channels)

        # Denominador: norma de cada señal (por canal)
        den = tf.sqrt(tf.reduce_sum(tf.square(x), axis=1) * tf.reduce_sum(tf.square(y), axis=1))  # (batch, channels)

        # Correlación por canal
        corr = num / (den + 1e-8)  # (batch, channels)

        # Pérdida: 1 - correlación promedio sobre canales y batch
        return 1.0 - tf.reduce_mean(corr)
    
    def temporal_gradient_loss(y_true, y_pred):
        dy_true = y_true[:, 1:, :] - y_true[:, :-1, :]
        dy_pred = y_pred[:, 1:, :] - y_pred[:, :-1, :]
        return tf.reduce_mean(tf.abs(dy_true - dy_pred))




class SamplingLayer(tf.keras.layers.Layer):
    """Custom sampling layer for VAE"""

    @tf.function
    def call(self, inputs):
        z_mean, z_log_var = inputs
        
        z_log_var = tf.clip_by_value(z_log_var, -10.0, 10.0)
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var + 1e-8) * epsilon

