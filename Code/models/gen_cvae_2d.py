import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import layers
from keras import Model

tf.config.experimental_run_functions_eagerly(True)
tf.executing_eagerly()


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


class Gen_CondVAE_2D(Model):
    """
    VAE 2D condicional: condiciones 'c' (one-hot o continuas) se concatenan en
    el cuello de botella y también se pasan al decoder.
    """

    def __init__(self, params, input_shape_, n_nodes, latent_dim=128,
                 condition_dim=0, tensorboard_logs=None):
        super().__init__()

        assert condition_dim and condition_dim > 0, \
            "Debes especificar condition_dim > 0 para usar el CVAE."

        if tensorboard_logs and not os.path.exists(tensorboard_logs):
            os.makedirs(tensorboard_logs)

        self.input_shape_ = input_shape_
        self.params = params
        self.SEED = 50
        tf.random.set_seed(self.SEED)
        self.tensorboard_logs = tensorboard_logs or "./tb_logs/"
        self.file_writer = tf.summary.create_file_writer(self.tensorboard_logs)

        self.initializer = tf.keras.initializers.HeNormal()
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim

        # Densas para parámetros del latente
        self.z_mean_dense = layers.Dense(latent_dim, name="z_mean")
        self.z_log_var_dense = layers.Dense(latent_dim, name="z_log_var")
        self.sampling_layer = SamplingLayer()

        # Construye modelo completo (con dos inputs: x y c)
        self.model = self.assemble_full_model(input_shape_, n_nodes)

        # Warm-up (beta-VAE)
        self.beta = tf.Variable(0.0, trainable=False, dtype=tf.float32)

    def call(self, inputs, training=None, mask=None):
        # inputs = [x, c]
        return self.model(inputs, training=training)

    # ---------- ENCODER ----------
    def build_encoder_module(self, x_in, c_in, input_shape):
        """
        Encoder Conv2D; concatena condición 'c' tras aplanar features.
        x_in: (batch, T, F)
        c_in: (batch, condition_dim)
        """
        x = layers.Reshape((self.input_shape_[0], self.input_shape_[1], 1))(x_in)

        x = layers.Conv2D(
            filters=32, kernel_size=(5, 15), strides=(2, 4), padding='same',
            activation='leaky_relu', kernel_initializer=self.initializer,
            kernel_regularizer=tf.keras.regularizers.l2(self.params["l2_reg"]),
        )(x)

        x = layers.Conv2D(
            filters=64, kernel_size=(5, 15), strides=(2, 4), padding='same',
            activation='leaky_relu',
            kernel_regularizer=tf.keras.regularizers.l2(self.params["l2_reg"]),
        )(x)

        x = layers.Conv2D(
            filters=10, kernel_size=(3, 5), strides=(2, 2), padding='same',
            activation='leaky_relu',
            kernel_regularizer=tf.keras.regularizers.l2(self.params["l2_reg"]),
        )(x)

        x = layers.Flatten()(x)
        x = layers.Dense(self.latent_dim, activation='leaky_relu')(x)

        # Concatenamos la condición en el cuello de botella
        xc = layers.Concatenate()([x, c_in])
        # (opcional) proyección previa
        xc = layers.Dense(self.latent_dim, activation='leaky_relu')(xc)

        z_mean = self.z_mean_dense(xc)
        z_log_var = self.z_log_var_dense(xc)
        z = self.sampling_layer([z_mean, z_log_var])
        return z, z_mean, z_log_var

    # ---------- DECODER ----------
    def build_decoder_module(self):
        """
        Decoder condicional: recibe [z, c] y decodifica a la señal.
        """
        z_in = layers.Input(shape=(self.latent_dim,), name="decoder_z")
        c_in = layers.Input(shape=(self.condition_dim,), name="decoder_c")

        zc = layers.Concatenate()([z_in, c_in])

        x = layers.Dense(25 * 16 * 32)(zc)
        x = layers.LeakyReLU()(x)
        x = layers.Reshape((25, 16, 32))(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Conv2DTranspose(64, (4, 4), strides=(2, 4), padding='same')(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Conv2DTranspose(32, (4, 4), strides=(2, 4), padding='same')(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Conv2DTranspose(32, (4, 4), strides=(2, 2), padding='same')(x)
        x = layers.LeakyReLU()(x)
        x = layers.Conv2DTranspose(1, (4, 4), strides=(2, 4), padding='same')(x)
        x = tf.keras.activations.tanh(x)
        x = tf.squeeze(x, axis=-1)  # -> (batch, T, F)

        self.decoder = tf.keras.Model([z_in, c_in], x, name="decoder")
        return self.decoder

    def decode_from_latent(self, z, c):
        if not hasattr(self, "decoder"):
            self.build_decoder_module()
        return self.decoder([z, c])

    # ---------- AUTOENCODER BRANCH ----------
    def build_autoencoder_branch(self, x_in, c_in, input_shape):
        z, z_mean, z_log_var = self.build_encoder_module(x_in, c_in, input_shape)
        if not hasattr(self, "decoder"):
            self.build_decoder_module()
        x_hat = self.decoder([z, c_in])
        return z_mean, z_log_var, z, x_hat

    # ---------- FULL MODEL ----------
    def assemble_full_model(self, input_shape, n_nodes):
        x_in = layers.Input(shape=input_shape, name="x_in")
        c_in = layers.Input(shape=(self.condition_dim,), name="c_in")

        z_mean, z_log_var, z, x_hat = self.build_autoencoder_branch(x_in, c_in, input_shape)

        # El modelo ahora devuelve también z_mean y z_log_var para la pérdida
        model = Model(inputs=[x_in, c_in], outputs=[x_hat, z_mean, z_log_var], name="C-VAE-2D")
        return model

    # ---------- PÉRDIDAS ----------
    def compute_loss_VAE(self, y_pred, y_true, z_mean, z_log_var):
        # Recon: MSE + extras
        mse_loss = tf.reduce_mean(tf.reduce_sum(tf.keras.losses.mean_squared_error(y_true, y_pred), axis=[1]))

        grad_loss = self.temporal_gradient_loss(y_true, y_pred)
        corr_loss = self.negative_corr_loss(y_true, y_pred)

        logvar = tf.clip_by_value(z_log_var, -10.0, 10.0)
        kl = -0.5 * (1 + logvar - tf.square(z_mean) - tf.exp(logvar + 1e-8))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl, axis=1))

        beta = self.beta
        total = mse_loss + beta * kl_loss + 0.5 * corr_loss  # + 0.3 * grad_loss (si quieres)
        return total, mse_loss, kl_loss

    # ---------- TRAIN / TEST ----------
    def train_step(self, data):
        # data = ((x, c), y_true) donde y_true normalmente es x (autoencoder)
        (x, c), y_true = data
        with tf.GradientTape() as tape:
            y_pred, z_mean, z_log_var = self.model([x, c], training=True)
            total, mse, kl = self.compute_loss_VAE(y_pred, y_true, z_mean, z_log_var)

        grads = tape.gradient(total, self.trainable_variables)
        clipped, global_norm = tf.clip_by_global_norm(grads, clip_norm=5.0)
        self.optimizer.apply_gradients(zip(clipped, self.trainable_variables))

        step = int(self.optimizer.iterations)
        with self.file_writer.as_default():
            tf.summary.scalar("loss/total_loss", total, step=step)
            tf.summary.scalar("loss/mse_autoencoder", mse, step=step)
            tf.summary.scalar("loss/kl_loss", kl, step=step)

        tf.print("KL loss:", kl)
        return {"total_loss": total, "mse_autoencoder": mse, "kl_loss": kl}

    def test_step(self, data):
        (x, c), y_true = data
        y_pred, z_mean, z_log_var = self.model([x, c], training=False)
        total, mse, kl = self.compute_loss_VAE(y_pred, y_true, z_mean, z_log_var)
        return {"total_loss": total, "mse_autoencoder": mse, "kl_loss": kl}
  

    def negative_corr_loss(self, y_true, y_pred):
        mu_x = tf.reduce_mean(y_true, axis=1, keepdims=True)
        mu_y = tf.reduce_mean(y_pred, axis=1, keepdims=True)
        x = y_true - mu_x
        y = y_pred - mu_y
        num = tf.reduce_sum(x * y, axis=1)
        den = tf.sqrt(tf.reduce_sum(tf.square(x), axis=1) * tf.reduce_sum(tf.square(y), axis=1))
        corr = num / (den + 1e-8)
        return 1.0 - tf.reduce_mean(corr)

    def temporal_gradient_loss(self, y_true, y_pred):
        dy_true = y_true[:, 1:, :] - y_true[:, :-1, :]
        dy_pred = y_pred[:, 1:, :] - y_pred[:, :-1, :]
        return tf.reduce_mean(tf.abs(dy_true - dy_pred))
