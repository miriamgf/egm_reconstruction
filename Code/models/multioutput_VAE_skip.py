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


class MultiOutput_VAE_skip(Model):
    """
    Used to generate our multi-output model. This CNN contains 2 branches, one for autoencoder, other for
    regression from Bsps to EGMs.
    """

    def __init__(
        self, params, input_shape_, n_nodes, latent_dim=2, tensorboard_logs=None
    ):
        super(MultiOutput_VAE_skip, self).__init__()

        if not os.path.exists(tensorboard_logs):
            os.makedirs(tensorboard_logs)

        self.input_shape_ = input_shape_
        self.params = params
        self.tensorboard_logs = tensorboard_logs
        self.file_writer = tf.summary.create_file_writer(self.tensorboard_logs)

        # Define the layers as instance attributes
        initializer = tf.keras.initializers.HeNormal()

        # Define encoder layers
        self.conv1 = layers.Conv3D(
            64,
            (5, 2, 2),
            strides=1,
            padding="same",
            activation="leaky_relu",
            input_shape=input_shape_[2:],
            kernel_initializer=initializer,
            kernel_regularizer=tf.keras.regularizers.l2(params["l2_reg"]),
        )
        self.conv2 = layers.Conv3D(
            64, (5, 2, 2), strides=1, padding="same", activation="leaky_relu"
        )
        self.conv3 = layers.Conv3D(
            32, (5, 2, 2), strides=1, padding="same", activation="leaky_relu"
        )
        self.maxpool1 = layers.MaxPooling3D((1, 2, 2))
        self.conv4 = layers.Conv3D(
            12,
            (5, 2, 2),
            strides=1,
            padding="same",
            activation="leaky_relu",
            kernel_regularizer=tf.keras.regularizers.l2(params["l2_reg"]),
        )
        self.maxpool2 = layers.MaxPooling3D((1, 2, 2))
        self.conv5 = layers.Conv3D(
            4, (5, 2, 2), strides=1, padding="same", activation="linear"
        )
        self.maxpool3 = layers.MaxPooling3D((1, 1, 2))
        self.flatten = layers.Flatten()
        dummy_input = tf.ones(
            (1,) + input_shape_[:]
        )  # Batch size of 1, the rest of the input shape
        dummy_output = self.flatten(
            self.maxpool3(
                self.conv5(
                    self.maxpool2(
                        self.conv4(
                            self.maxpool1(
                                self.conv3(self.conv2(self.conv1(dummy_input)))
                            )
                        )
                    )
                )
            )
        )
        latent_dim = np.prod(
            dummy_output.shape[:]
        )  # Get the flattened size dynamically

        self.z_mean_dense = layers.Dense(latent_dim, name="z_mean")
        self.z_log_var_dense = layers.Dense(latent_dim, name="z_log_var")
        self.sampling_layer = SamplingLayer()
        self.reshape_latent_space = layers.Reshape((self.params["batch_size"], 3, 4, 4))

        # Define decoder layers
        self.decoder_conv1 = layers.Conv3D(
            12,
            (5, 2, 2),
            strides=1,
            padding="same",
            activation="leaky_relu",
            kernel_initializer=initializer,
        )
        self.upsample1 = layers.UpSampling3D((1, 1, 2))
        self.decoder_conv2 = layers.Conv3D(
            32, (5, 2, 2), strides=1, padding="same", activation="leaky_relu"
        )
        self.upsample2 = layers.UpSampling3D((1, 2, 2))
        self.decoder_conv3 = layers.Conv3D(
            32, (5, 2, 2), strides=1, padding="same", activation="leaky_relu"
        )
        self.upsample3 = layers.UpSampling3D((1, 2, 2))
        self.decoder_output = layers.Conv3D(
            1,
            (5, 2, 2),
            strides=1,
            padding="same",
            activation="linear",
            kernel_regularizer=tf.keras.regularizers.l2(params["l2_reg"]),
            name="Autoencoder_output",
        )

        # define reconstruction layers
        self.conv3d_1 = layers.Conv3D(
            64,
            (5, 2, 2),
            strides=(1, 1, 1),
            padding="same",
            activation="leaky_relu",
            input_shape=(3, 4, 4, 1),
            kernel_regularizer=tf.keras.regularizers.l2(self.params["l2_reg"]),
            kernel_initializer=initializer,

        )
        self.upsampling3d_1 = layers.UpSampling3D((1, 2, 2))

        self.conv3d_2 = layers.Conv3D(
            32,
            (5, 3, 3),
            strides=(1, 1, 1),
            padding="same",
            activation="leaky_relu",
            kernel_regularizer=tf.keras.regularizers.l2(l=self.params["l2_reg"]),
        )
        self.upsampling3d_2 = layers.UpSampling3D((1, 2, 2))

        self.conv3d_3 = layers.Conv3D(
            3, (5, 3, 3), strides=(1, 1, 1), padding="same", activation="leaky_relu"
        )

        self.time_distributed = layers.TimeDistributed(layers.Flatten())
        self.batch_norm = layers.BatchNormalization(axis=1)
        self.lstm = layers.LSTM(self.params["LSTM_units"], return_sequences=True)
        self.dropout = layers.Dropout(self.params["dropout"])
        self.dense = layers.Dense(
            n_nodes, activation="leaky_relu", name="Regressor_output"
        )

        self.model = self.assemble_full_model(input_shape_, n_nodes)

    def call(self, inputs, training=None, mask=None):

        # x = inputs[0]  # Assuming inputs is a tuple and you only need the first element
        outputs = self.model(inputs, training=training)
        return outputs

    def build_encoder_module(self, inputs, input_shape):

        x1 = self.conv1(inputs)  # First convolution layer
        x2 = self.conv2(x1)  # Second convolution layer
        x3 = self.conv3(x2)  # Third convolution layer
        x3_skip = x3  # Save for skip connection

        x4 = self.maxpool1(x3)  # First max pooling
        x5 = self.conv4(x4)  # Fourth convolution layer
        x6 = self.maxpool2(x5)  # Second max pooling
        x7 = self.conv5(x6)  # Fifth convolution layer
        x7_skip = x7  # Save for skip connection
        x8 = self.maxpool3(x7)  # Third max pooling
        x_flat = self.flatten(x8)  # Flatten before latent space
        # Latent space
        z_mean = self.z_mean_dense(x_flat)
        z_log_var = self.z_log_var_dense(x_flat)
        z = self.sampling_layer([z_mean, z_log_var])
        z = self.reshape_latent_space(z)  # Reshape for decoder
        skip_connections = [x3_skip, x7_skip]
        # Return both encoder outputs and the skip connections for the decoder
        return z, z_mean, z_log_var, skip_connections

    def build_decoder_module(
        self, inputs, input_shape, latent_inputs, skip_connections
    ):

        # Unpack the encoder outputs to use in skip connections
        x3_skip, x7_skip = skip_connections
        x2_dec = self.upsample1(latent_inputs)
        x2_dec = (x2_dec + x7_skip)  # Skip connection from encoder's third convolution layer
        x3_dec = self.decoder_conv2(x2_dec)
        x4_dec = self.upsample2(x3_dec)
        x5_dec = self.decoder_conv3(x4_dec)
        x6_dec = self.upsample3(x5_dec)
        x6_dec = (x6_dec + x3_skip)  # Skip connection from encoder's fifth convolution layer
        output = self.decoder_output(x6_dec)  # Final output
        return output

    def build_autoencoder_branch(self, inputs, input_shape):
        latent_space, z_mean, z_log_var, skip_connections = self.build_encoder_module(
            inputs, input_shape
        )
        decoded_output = self.build_decoder_module(
            inputs, input_shape, latent_space, skip_connections
        )
        return z_mean, z_log_var, latent_space, decoded_output

    def build_reconstruction_branch(self, inputs, input_shape_, latent_inputs, n_nodes):
        tf.keras.initializers.HeNormal()

        x = self.conv3d_1(latent_inputs)
        x = self.upsampling3d_1(x)
        x = self.conv3d_2(x)
        x = self.upsampling3d_2(x)
        x = self.conv3d_3(x)
        x = self.time_distributed(x)
        x = self.batch_norm(x)
        x = self.lstm(x)
        x = self.dropout(x)
        x = self.dense(x)

        return x

    def assemble_full_model(self, input_shape, n_nodes):
        """
        Used to assemble our multi-output model CNN.
        """
        inputs = layers.Input(shape=input_shape)
        z_mean, z_log_var, latent_space, autoencoder_branch = (
            self.build_autoencoder_branch(inputs, input_shape)
        )
        reconstruction_branch = self.build_reconstruction_branch(
            inputs, input_shape, latent_space, n_nodes
        )

        model = Model(
            inputs=inputs,
            outputs=[autoencoder_branch, reconstruction_branch],
            name="MultiOutput",
        )

        return model

    def log_normal_pdf(self, sample, mean, logvar, raxis=1):
        log2pi = tf.math.log(2.0 * np.pi)
        return tf.reduce_sum(
            -0.5 * ((sample - mean) ** 2.0 * tf.exp(-logvar) + logvar + log2pi),
            axis=raxis,
        )

    def compute_loss_VAE(self, x, y):
        z_mean, z_log_var, latent_space, autoencoder_branch = (
            self.build_autoencoder_branch(x, self.input_shape_)
        )
        mse_loss = tf.reduce_mean(
            tf.reduce_sum(
                tf.keras.losses.mean_squared_error(y, x),
                axis=[1],  # Reducir todas las dimensiones excepto el batch
            )
        )

        logvar = tf.clip_by_value(z_log_var, -1.0, 1.0)
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(logvar + 1e-8))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss))
        total_loss = mse_loss + kl_loss

        return total_loss, mse_loss, kl_loss, latent_space

    def train_step(self, data):
        """
        Define train step
        """

        # Unpack data
        x, (y_true_autoencoder, y_true_regressor) = data

        with tf.GradientTape() as tape:
            y_pred_autoencoder, y_pred_regressor = self.model(x, training=True)
            loss_autoencoder, mse_autoencoder, kl_loss, latent_space = (
                self.compute_loss_VAE(y_pred_autoencoder, y_true_autoencoder)
            )

            loss_regression = tf.reduce_mean(
                tf.keras.losses.mean_squared_error(y_true_regressor, y_pred_regressor)
            )

            total_loss = (
                loss_autoencoder + loss_regression
            )  # Or weighted: alpha*loss_autoencoder + beta*loss_regression

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
            tf.summary.scalar("loss/mse_regression", loss_regression, step=step)

            # Log latent space embeddings
            # Flatten the latent space for visualization in 2D
            self.log_latent_space_image(latent_space, step)

        return {
            "loss": total_loss,
            "autoencoder_loss": loss_autoencoder,
            "autoencoder_mse": mse_autoencoder,
            "reconstruction_mse": loss_regression,
            "reconstruction_loss": loss_regression
        }


    def test_step(self, data):
        """
        Define test step (used for validation).
        """
        # Unpack data
        x, (y_true_autoencoder, y_true_regressor) = data

        # Forward pass through the model without gradient tracking
        y_pred_autoencoder, y_pred_regressor = self.model(x, training=False)

        # Compute loss for the autoencoder branch
        loss_autoencoder, mse_autoencoder, kl_loss, latent_space = (
            self.compute_loss_VAE(y_pred_autoencoder, y_true_autoencoder)
        )

        # Compute loss for the regression branch
        # loss_regression = MeanSquaredError()(y_true_regressor, y_pred_regressor)
        loss_regression = tf.reduce_mean(
            tf.keras.losses.mean_squared_error(y_true_regressor, y_pred_regressor)
        )

        # Total loss
        total_loss = (
                        self.params["loss_weight_2"]*loss_autoencoder + self.params["loss_weight_2"]*loss_regression
                    ) 
        # Return losses for tracking
        return {
            "loss": total_loss,
            "autoencoder_loss": loss_autoencoder,
            "autoencoder_mse": mse_autoencoder,
            "reconstruction_mse": loss_regression,
            "reconstruction_loss": loss_regression
        }


    def log_latent_space_image(self, latent_space, step):
        # Reshape and reduce dimensionality to 2D for visualization
        latent_space_flat = tf.reshape(latent_space, (-1, latent_space.shape[1]))

        # Plot the 2D latent space
        plt.figure(figsize=(6, 6))
        plt.scatter(latent_space_flat[:, 0], latent_space_flat[:, 1], alpha=0.5)
        plt.title(f"Latent Space at Step {step}")
        plt.savefig(f"{self.tensorboard_logs}latent_space_{step}.png")
        plt.close()

        with self.file_writer.as_default():
            image = tf.io.read_file(f"{self.tensorboard_logs}latent_space_{step}.png")
            image = tf.image.decode_image(image)
            image = tf.expand_dims(image, axis=0)
            tf.summary.image("Latent Space", image, step=step)


class SamplingLayer(tf.keras.layers.Layer):
    """Custom sampling layer for VAE"""

    @tf.function
    def call(self, inputs):
        z_mean, z_log_var = inputs

        z_log_var = tf.clip_by_value(z_log_var, -1, 1)
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var + 1e-8) * epsilon
