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


class Gen_VAE(Model):
    """
    Used to generate synthetic EGMs using a trained VAE model.
    """

    def __init__(
        self, params, input_shape_, n_nodes, latent_dim=128, tensorboard_logs=None
    ):
        super(Gen_VAE, self).__init__()

        if not os.path.exists(tensorboard_logs):
            os.makedirs(tensorboard_logs)

        self.input_shape_ = input_shape_
        self.params = params
        self.SEED=self.params["seed"]
        tf.random.set_seed(self.SEED)
        self.tensorboard_logs = tensorboard_logs
        self.file_writer = tf.summary.create_file_writer(self.tensorboard_logs)

        # Define the layers as instance attributes
        initializer = tf.keras.initializers.HeNormal()

        # Define encoder layers
        
        self.dense1 = layers.Flatten()
        self.dense2 = layers.Dense(1024, activation='relu')
        self.dense3 = layers.Dense(512, activation='relu')
        self.dense4 = layers.Dense(256, activation='relu')
        self.latent_dim = latent_dim

        self.z_mean_dense = layers.Dense(latent_dim, name="z_mean")
        self.z_log_var_dense = layers.Dense(latent_dim, name="z_log_var")
        self.sampling_layer = SamplingLayer()
        #self.reshape_latent_space = layers.Reshape((self.params["batch_size"], 10, 10))

        self.dense5 = layers.Dense(256, activation='relu')
        self.dense6 = layers.Dense(512, activation='relu')
        self.dense7 = layers.Dense(1024, activation='relu')
        self.dense8 = layers.Dense(2048, activation='sigmoid')
        self.decoder_output = layers.Reshape(self.input_shape_)
        self.model, self.latent_space = self.assemble_full_model(input_shape_, n_nodes)

    def call(self, inputs, training=None, mask=None):

        # x = inputs[0]  # Assuming inputs is a tuple and you only need the first element
        outputs = self.model(inputs, training=training)
        return outputs, self.latent_space

    def build_encoder_module(self, inputs, input_shape):
        """
        Encoder implementation
        """
        
        
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)

        z_mean = self.z_mean_dense(x)
        z_log_var = self.z_log_var_dense(x)
        z = self.sampling_layer([z_mean, z_log_var])
        
        #z = self.reshape_latent_space(z)

        return z, z_mean, z_log_var

    def build_decoder_module(self, inputs, input_shape, latent_inputs):
        '''
        Decoder implementation
        
        '''
        
        # x = self.decoder_conv1(latent_inputs)
        x = self.dense5(inputs)
        x = self.dense6(x)
        x = self.dense7(x)
        x = self.dense8(x)
        output = self.decoder_output(x)
        return output

    def build_autoencoder_branch(self, inputs, input_shape):
        '''
        Build assamble between encoder and decoder
        '''
        latent_space, z_mean, z_log_var = self.build_encoder_module(inputs, input_shape)
        decoded_output = self.build_decoder_module(inputs, input_shape, latent_space)
        return z_mean, z_log_var, latent_space, decoded_output

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

    def compute_loss_VAE(self, x, y):

        # inputs = layers.Input(shape=self.input_shape)
        z_mean, z_log_var, latent_space, autoencoder_output = (
            self.build_autoencoder_branch(x, self.input_shape_)
        )
        mse_loss = tf.reduce_mean(
            tf.reduce_sum(
                tf.keras.losses.mean_squared_error(
                    y, x
                ),  # x es la salida y y es la etiqueta
                axis=[1],  # Reducir todas las dimensiones excepto el batch
            )
        )

        logvar = tf.clip_by_value(z_log_var, -10.0, 10.0)
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(logvar + 1e-8))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss))
        total_loss = mse_loss + kl_loss

        return total_loss, mse_loss, kl_loss, latent_space

    def train_step(self, data):
        """
        Define train step
        """

        # Unpack data
        x, y_true_autoencoder = data

        with tf.GradientTape() as tape:
            # Forward pass through the model
            y_pred_autoencoder = self.model(x, training=True)
            # Compute loss for the autoencoder branch (e.g., VAE loss with reconstruction + KL divergence)
            loss_autoencoder, mse_autoencoder, kl_loss, latent_space = (
                self.compute_loss_VAE(y_pred_autoencoder, y_true_autoencoder)
            )

            # Total loss (you can adjust weights if one branch's loss should have more influence)
            total_loss = loss_autoencoder
        # Compute and apply gradients based on the total loss

        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        # Clip gradients to avoid exploding gradients (based on their global norm)
        clipped_gradients, global_norm = tf.clip_by_global_norm(
            gradients, clip_norm=1.0
        )
        self.optimizer.apply_gradients(
            zip(clipped_gradients, self.model.trainable_variables)
        )
        step = int(self.optimizer.iterations)
        with self.file_writer.as_default():
            tf.summary.scalar("loss/total_loss", total_loss, step=step)
            tf.summary.scalar("loss/loss_autoencoder", loss_autoencoder, step=step)
            tf.summary.scalar("loss/mse_autoencoder", mse_autoencoder, step=step)

            # self.log_latent_space_image(latent_space, step)

        return {
            "total_loss": total_loss,
            "loss_autoencoder": loss_autoencoder,
            "mse_autoencoder": mse_autoencoder,
        }

    def test_step(self, data):
        """
        Define test step (used for validation).
        """
        # Unpack data
        x, y_true_autoencoder = data

        # Forward pass through the model without gradient tracking
        y_pred_autoencoder = self.model(x, training=False)

        # Compute loss for the autoencoder branch
        loss_autoencoder, mse_autoencoder, kl_loss, latent_space = (
            self.compute_loss_VAE(y_pred_autoencoder, y_true_autoencoder)
        )
        

        # Total loss
        total_loss = loss_autoencoder 

        # Return losses for tracking
        return {
            "total_loss": total_loss,
            "loss_autoencoder": loss_autoencoder,
            "mse_autoencoder": mse_autoencoder,
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

