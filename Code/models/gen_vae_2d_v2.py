import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import layers
from keras import Model

tf.keras.backend.clear_session()
np.random.seed(0)
tf.random.set_seed(0)

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("float32") 
tf.config.experimental_run_functions_eagerly(True)  
tf.executing_eagerly()


class Gen_VAE_2D_v2(tf.keras.Model):
    def __init__(self, params, input_shape_, n_nodes, latent_dim=250, tensorboard_logs=None):
        super().__init__()
        self.params = params
        self.input_shape_ = input_shape_
        self.latent_dim = latent_dim
        self.beta = tf.Variable(0.0, trainable=False, dtype=tf.float32)  
        self.initializer = tf.keras.initializers.HeNormal()

        # --- Encoder ---
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

        # --- Decoder ORIGINAL (comentado) ---
        
        z_in = layers.Input(shape=(latent_dim,))
        y = layers.Dense(25*16*32)(z_in); y = layers.LeakyReLU()(y)
        y = layers.Reshape((25,16,32))(y)
        #y = layers.Dropout(0.2)(y)
        
        '''y = layers.Conv2DTranspose(64, (4,4), strides=(2,4), padding='same')(y); y = layers.LeakyReLU()(y)
        #y = layers.Dropout(0.2)(y)
        y = layers.Conv2DTranspose(32, (4,4), strides=(2,4), padding='same')(y); y = layers.LeakyReLU()(y)
        #y = layers.Dropout(0.2)(y)
        y = layers.Conv2DTranspose(32, (4,4), strides=(2,2), padding='same')(y); y = layers.LeakyReLU()(y)
        y = layers.Conv2DTranspose(1, (4,4), strides=(2,4), padding='same')(y)
        y = tf.keras.activations.tanh(y)
        y = tf.squeeze(y, axis=-1)  # (batch, 400, 2048)'''
        
        y = layers.Conv2DTranspose(64, (4,4), strides=(2,4), padding='same')(y); y = layers.LeakyReLU()(y)
        y = layers.Conv2D(64, (5,1), padding='same', kernel_initializer=self.initializer)(y); y = layers.LeakyReLU()(y)   # anti-alias temporal

        y = layers.Conv2DTranspose(32, (4,4), strides=(2,4), padding='same')(y); y = layers.LeakyReLU()(y)
        y = layers.Conv2D(32, (5,1), padding='same', kernel_initializer=self.initializer)(y); y = layers.LeakyReLU()(y)   # anti-alias temporal

        y = layers.Conv2DTranspose(32, (4,4), strides=(2,2), padding='same')(y); y = layers.LeakyReLU()(y)
        y = layers.Conv2D(32, (5,1), padding='same', kernel_initializer=self.initializer)(y); y = layers.LeakyReLU()(y)   # anti-alias temporal

        # SALIDA FINAL (NO squeeze aquí; NO otra deconv)
        y = layers.Conv2DTranspose(1, (4,4), strides=(2,4), padding='same')(y)
        y = tf.keras.activations.tanh(y)  # o lineal si no escalas a [-1,1]
        y = tf.squeeze(y, axis=-1)    

        self.decoder = tf.keras.Model(z_in, y, name="decoder")

        self.file_writer = tf.summary.create_file_writer(tensorboard_logs)

    def call(self, inputs, training=None):
        z, z_mean, z_log_var = self.encoder(inputs, training=training)
        recon = self.decoder(z, training=training)
        return recon, z_mean, z_log_var

    @staticmethod
    def kl_normal(z_mean, z_log_var):
        return tf.reduce_mean(tf.reduce_sum(
            -0.5 * (1.0 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)), axis=1))
    
    def negative_corr_loss(self, y_true, y_pred):
        mu_x = tf.reduce_mean(y_true, axis=1, keepdims=True)
        mu_y = tf.reduce_mean(y_pred, axis=1, keepdims=True)
        x = y_true - mu_x
        y = y_pred - mu_y
        num = tf.reduce_sum(x * y, axis=1)
        den = tf.sqrt(tf.reduce_sum(tf.square(x), axis=1) * tf.reduce_sum(tf.square(y), axis=1))
        corr = num / (den + 1e-8)
        return 1.0 - tf.reduce_mean(corr)
    
    def _stft_logmag(self, x, n_fft=256, hop=None):
        # FIX: tiempo=eje 1 → transpón a (B*nodes, T)
        if hop is None: hop = n_fft // 4
        x = tf.cast(x, tf.float32)              # (B, T, N)
        B = tf.shape(x)[0]; T = tf.shape(x)[1]; N = tf.shape(x)[2]
        flat = tf.reshape(x, [B*N, T])          # (B*N, T)
        X = tf.signal.stft(flat, frame_length=n_fft, frame_step=hop,
                           window_fn=tf.signal.hann_window, pad_end=True)
        L = tf.math.log(tf.abs(X) + 1e-6)       # (B*N, F, frames)
        return L

    def stft_loss(self, y_true, y_pred, n_fft=256):
        # FIX: usa _stft_logmag que ya maneja tiempo en eje 1
        Lt = self._stft_logmag(y_true, n_fft)
        Lp = self._stft_logmag(y_pred, n_fft)
        return tf.reduce_mean(tf.abs(Lt - Lp))

    def _grad_l1(self, y_true, y_pred):
            # FIX: tiempo = eje 1 → diferencias en eje 1
            dy_t = y_true[:, 1:, :] - y_true[:, :-1, :]
            dy_p = y_pred[:, 1:, :] - y_pred[:, :-1, :]
            return tf.reduce_mean(tf.abs(dy_t - dy_p))

    def _hf_match_and_spurious(self, y_true, y_pred, n_fft=256, cutoff=0.4):
        # Espectros log-magnitude
        Lt = self._stft_logmag(y_true, n_fft)  # (B*, F, K)
        Lp = self._stft_logmag(y_pred, n_fft)  # (B*, F, K)

        Bf = tf.shape(Lt)[0]
        F  = tf.shape(Lt)[1]

        # Máscara de alta frecuencia (F_normalizado ∈ [0,1])
        freqs = tf.cast(tf.range(F), tf.float32) / tf.cast(tf.maximum(F-1, 1), tf.float32)
        mask_hf = tf.cast(freqs >= cutoff, tf.float32)              # (F,)
        mask_hf = tf.reshape(mask_hf, [1, -1, 1])                   # (1,F,1)

        # Peso proporcional a la energía AF de la real (w ≈ 0..1)
        # Normalizamos Lt en eje F por muestra/frame para que pese relativo
        Lt_hf = Lt * mask_hf
        # normalización robusta por-frame
        denom = tf.stop_gradient(tf.reduce_mean(tf.abs(Lt_hf), axis=1, keepdims=True) + 1e-6)
        w_t = tf.clip_by_value(tf.abs(Lt_hf) / denom, 0.0, 5.0)     # (B*,F,K)

        # 1) HF-match: acercar espectros donde la real tiene AF
        hf_match = tf.reduce_mean(w_t * tf.abs(Lp - Lt))

        # 2) HF-spurious: penalizar AF en pred donde la real NO tiene AF
        w_spur = mask_hf * tf.cast(w_t <= 0.2, tf.float32)          # baja energía en la real
        hf_spurious = tf.reduce_mean(w_spur * tf.nn.relu(Lp))

        return hf_match, hf_spurious

    def recon_loss_hf_preserve(self, y_true, y_pred):
        # Base global
        mse  = tf.reduce_mean(tf.square(y_true - y_pred))
        grad = self._grad_l1(y_true, y_pred)
        

        # Espectral: dos escalas para cubrir AF media y alta
        hf_m1, hf_s1 = self._hf_match_and_spurious(y_true, y_pred, n_fft=256, cutoff=0.40)
        hf_m2, hf_s2 = self._hf_match_and_spurious(y_true, y_pred, n_fft=512, cutoff=0.50)
        hf_match    = 0.5 * (hf_m1 + hf_m2)
        hf_spurious = 0.5 * (hf_s1 + hf_s2)

        # Mezcla estable: ajusta si te faltan picos (sube grad y hf_match) o sobra ruido (sube hf_spurious)
        return 0.35*mse + 0.30*grad + 0.25*hf_match + 0.10*hf_spurious

    
    '''def recon_loss(self,y_true, y_pred):
        mse  = tf.reduce_mean(tf.square(y_true - y_pred))

        grad = tf.reduce_mean(tf.abs((y_true[...,1:] - y_true[...,:-1]) -
                                    (y_pred[...,1:] - y_pred[...,:-1])))
        spec = (self.stft_loss(y_true, y_pred, 256) +
                self.stft_loss(y_true, y_pred, 512)) / 2.0
        
        print('grad', grad)
        print('spec', spec)

        return 0.5*mse + 0.3*grad + 0.2*spec'''

    def compute_loss_VAE(self, y_pred, y_true, z_mean, z_log_var, free_bits=0.5):
        tf.debugging.assert_all_finite(y_pred, "y_pred bad pre-loss")
        tf.debugging.assert_all_finite(y_true, "y_true bad pre-loss")

        recon = self.recon_loss_hf_preserve(y_true, y_pred)

        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        corr = self.negative_corr_loss(y_true, y_pred)
        tf.debugging.assert_all_finite(corr, "corr NaN/Inf")
        z_log_var_clipped = tf.clip_by_value(z_log_var, -10.0, 10.0)
        kl = tf.maximum(tf.reduce_mean(tf.reduce_sum(
            -0.5 * (1.0 + z_log_var_clipped - tf.square(z_mean) - tf.exp(z_log_var_clipped)), axis=1)), free_bits)
        tf.debugging.assert_all_finite(kl, "kl NaN/Inf")
        #recon = self.recon_loss(y_true, y_pred)

        #total = mse + self.beta * kl + 0.5 * corr
        total = recon + self.beta * kl + 0.5 * corr



        tf.debugging.assert_all_finite(total, "total NaN/Inf")
        return total, mse, kl

    def train_step(self, data):
        x, y_true = data
        tf.debugging.assert_all_finite(x, "Input x has NaN/Inf")
        tf.debugging.assert_all_finite(y_true, "Target y_true has NaN/Inf")
        step = int(self.optimizer.iterations)
        warm_steps = self.params.get("beta_warmup_steps", 3330)
        beta_max = float(self.params.get("beta_max", 3.0))
        frac = min(step / max(1, warm_steps), 1.0)
        self.beta.assign(frac * beta_max)

        with tf.GradientTape() as tape:
            y_pred, z_mean, z_log_var = self(x, training=True)
            tf.debugging.assert_all_finite(z_mean, "z_mean NaN/Inf")
            tf.debugging.assert_all_finite(z_log_var, "z_log_var NaN/Inf")
            tf.debugging.assert_all_finite(y_pred, "y_pred NaN/Inf")
            total, mse, kl = self.compute_loss_VAE(y_pred, y_true, z_mean, z_log_var)

        grads = tape.gradient(total, self.trainable_variables)
        for g in grads:
            if g is not None:
                tf.debugging.assert_all_finite(g, "Gradient NaN/Inf")
        grads, _ = tf.clip_by_global_norm(grads, 1.0)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
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
