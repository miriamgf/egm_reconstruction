import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import layers
from keras import Model

import tensorflow as tf, numpy as np, os
tf.keras.backend.clear_session()
np.random.seed(0)
tf.random.set_seed(0)

# If you ever enabled mixed precision, turn it OFF while debugging:
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("float32") 
tf.config.experimental_run_functions_eagerly(
    True
)  # Para versiones anteriores de TensorFlow 2.x
tf.executing_eagerly()

class Gen_VAE_2D_v2(tf.keras.Model):
    def __init__(self, params, input_shape_, n_nodes, latent_dim=250, tensorboard_logs=None):
        super().__init__()
        self.params = params
        self.input_shape_ = input_shape_
        self.latent_dim = latent_dim
        self.beta = tf.Variable(0.0, trainable=False, dtype=tf.float32)  # annealed β
        self.initializer = tf.keras.initializers.HeNormal()

        # --- Encoder (definido UNA vez) ---
        inp = layers.Input(shape=input_shape_)
        x = layers.Reshape((400, 2048, 1))(inp)

        x = layers.Conv2D(32, (5,15), strides=(2,4), padding='same',
                          kernel_initializer=self.initializer,
                          kernel_regularizer=tf.keras.regularizers.l2(self.params["l2_reg"]))(x)
        x = layers.LeakyReLU()(x)

        x = layers.Conv2D(64, (5,15), strides=(2,4), padding='same',
                          kernel_regularizer=tf.keras.regularizers.l2(self.params["l2_reg"]))(x)
        x = layers.LeakyReLU()(x)

        x = layers.Conv2D(10, (3,5), strides=(2,2), padding='same',
                          kernel_regularizer=tf.keras.regularizers.l2(self.params["l2_reg"]))(x)
        x = layers.LeakyReLU()(x)

        x = layers.Flatten()(x)
        x = layers.Dense(self.latent_dim)(x)
        x = layers.LeakyReLU()(x)

        z_mean = layers.Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(self.latent_dim, name="z_log_var")(x)
        z = SamplingLayer()([z_mean, z_log_var])

        self.encoder = tf.keras.Model(inp, [z, z_mean, z_log_var], name="encoder")

        # --- Decoder (definido UNA vez) ---
        z_in = layers.Input(shape=(latent_dim,))
        y = layers.Dense(25*16*32)(z_in); y = layers.LeakyReLU()(y)
        y = layers.Reshape((25,16,32))(y)
        y = layers.Dropout(0.2)(y)
        y = layers.Conv2DTranspose(64, (4,4), strides=(2,4), padding='same')(y); y = layers.LeakyReLU()(y)
        y = layers.Dropout(0.2)(y)
        y = layers.Conv2DTranspose(32, (4,4), strides=(2,4), padding='same')(y); y = layers.LeakyReLU()(y)
        y = layers.Dropout(0.2)(y)
        y = layers.Conv2DTranspose(32, (4,4), strides=(2,2), padding='same')(y); y = layers.LeakyReLU()(y)
        y = layers.Conv2DTranspose(1, (4,4), strides=(2,4), padding='same')(y)
        y = tf.keras.activations.tanh(y)
        y = tf.squeeze(y, axis=-1)  # (batch, 400, 2048)

        self.decoder = tf.keras.Model(z_in, y, name="decoder")

        # Para TensorBoard
        self.file_writer = tf.summary.create_file_writer(tensorboard_logs)

    def call(self, inputs, training=None):
        z, z_mean, z_log_var = self.encoder(inputs, training=training)
        recon = self.decoder(z, training=training)
        return recon, z_mean, z_log_var

    @staticmethod
    def kl_normal(z_mean, z_log_var):
        # KL[q(z|x) || N(0,I)]
        return tf.reduce_mean(tf.reduce_sum(
            -0.5 * (1.0 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)), axis=1))
    
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

    def compute_loss_VAE(self, y_pred, y_true, z_mean, z_log_var):
        tf.debugging.assert_all_finite(y_pred, "y_pred bad pre-loss")
        tf.debugging.assert_all_finite(y_true, "y_true bad pre-loss")

        # Simple & stable MSE
        mse = tf.reduce_mean(tf.square(y_true - y_pred))

        # Correlation (or disable temporarily)
        corr = self.negative_corr_loss(y_true, y_pred)  # ensure eps + clip inside
        tf.debugging.assert_all_finite(corr, "corr NaN/Inf")

        # KL with clip
        z_log_var_clipped = tf.clip_by_value(z_log_var, -10.0, 10.0)
        kl = tf.reduce_mean(tf.reduce_sum(
            -0.5 * (1.0 + z_log_var_clipped - tf.square(z_mean) - tf.exp(z_log_var_clipped)), axis=1))
        tf.debugging.assert_all_finite(kl, "kl NaN/Inf")

        total = mse + self.beta * kl + 0.5 * corr
        tf.debugging.assert_all_finite(total, "total NaN/Inf")
        return total, mse, kl


    def train_step(self, data):
        x, y_true = data

        # Data check (some later batches may be bad)
        tf.debugging.assert_all_finite(x, "Input x has NaN/Inf")
        tf.debugging.assert_all_finite(y_true, "Target y_true has NaN/Inf")

        # β warmup
        step = int(self.optimizer.iterations)
        warm_steps = self.params.get("beta_warmup_steps", 200_000)  # longer warmup helps
        beta_max = float(self.params.get("beta_max", 1.0))          # start with <=1.0
        frac = min(step / max(1, warm_steps), 1.0)
        self.beta.assign(frac * beta_max)

        with tf.GradientTape() as tape:
            y_pred, z_mean, z_log_var = self(x, training=True)

            # Activation checks
            tf.debugging.assert_all_finite(z_mean, "z_mean NaN/Inf")
            tf.debugging.assert_all_finite(z_log_var, "z_log_var NaN/Inf")
            tf.debugging.assert_all_finite(y_pred, "y_pred NaN/Inf")

            total, mse, kl = self.compute_loss_VAE(y_pred, y_true, z_mean, z_log_var)

        grads = tape.gradient(total, self.trainable_variables)

        # Gradient check
        for g in grads:
            if g is not None:
                tf.debugging.assert_all_finite(g, "Gradient NaN/Inf")

        # Stronger clipping while debugging
        grads, _ = tf.clip_by_global_norm(grads, 1.0)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # Weight check (costly; keep until stable then remove)
        for v in self.trainable_variables:
            tf.debugging.assert_all_finite(v, f"Weight {v.name} became NaN/Inf")

        return {"total_loss": total, "mse_autoencoder": mse, "kl_loss": kl, "beta": self.beta}

    def test_step(self, data):
        x, y_true = data
        y_pred, z_mean, z_log_var = self(x, training=False)
        total, mse, kl = self.compute_loss_VAE(y_pred, y_true, z_mean, z_log_var)
        return {"total_loss": total, "mse_autoencoder": mse, "kl_loss": kl, "beta": self.beta}

class SamplingLayer(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        z_log_var = tf.clip_by_value(z_log_var, -10.0, 10.0)
        eps = tf.random.normal(tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * eps


