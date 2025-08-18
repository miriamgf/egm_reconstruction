import tensorflow as tf
from keras import layers

class SamplingLayer(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        z_log_var = tf.clip_by_value(z_log_var, -10.0, 10.0)
        eps = tf.random.normal(tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * eps

class Gen_VAE_2D_v2_Cond(tf.keras.Model):
    """
    VAE 2D condicionado por clase (2 clases: 2 y 4).
    Encoder y decoder reciben el one-hot de clase.
    KL se calcula contra un prior gaussiano por clase aprendido.
    """
    def __init__(self, params, input_shape_, n_nodes, latent_dim=250, tensorboard_logs=None, num_classes=2):
        super().__init__()
        self.params = params
        self.input_shape_ = input_shape_
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.beta = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        self.initializer = tf.keras.initializers.HeNormal()

        # --- Inputs ---
        x_inp = layers.Input(shape=input_shape_, name="x")               # (B, 400, 2048)
        c_inp = layers.Input(shape=(num_classes,), name="class_onehot")  # (B, 2)

        # --- Encoder (condicionado) ---
        x = layers.Reshape((400, 2048, 1))(x_inp)
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
        x = layers.Dense(self.latent_dim, kernel_initializer=self.initializer)(x)
        x = layers.LeakyReLU()(x)

        # Condicionamos concatenando el one-hot
        x = layers.Concatenate()([x, c_inp])

        z_mean = layers.Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(self.latent_dim, name="z_log_var")(x)
        z = SamplingLayer()([z_mean, z_log_var])

        self.encoder = tf.keras.Model([x_inp, c_inp], [z, z_mean, z_log_var], name="encoder")

        # --- Decoder (condicionado) ---
        z_in = layers.Input(shape=(latent_dim,), name="z")
        c_in_dec = layers.Input(shape=(num_classes,), name="class_onehot_dec")
        dec_in = layers.Concatenate()([z_in, c_in_dec])  # concat cond

        y = layers.Dense(25*16*32)(dec_in); y = layers.LeakyReLU()(y)
        y = layers.Reshape((25,16,32))(y)

        y = layers.Conv2DTranspose(64, (4,4), strides=(2,4), padding='same')(y); y = layers.LeakyReLU()(y)
        y = layers.Conv2D(64, (5,1), padding='same', kernel_initializer=self.initializer)(y); y = layers.LeakyReLU()(y)

        y = layers.Conv2DTranspose(32, (4,4), strides=(2,4), padding='same')(y); y = layers.LeakyReLU()(y)
        y = layers.Conv2D(32, (5,1), padding='same', kernel_initializer=self.initializer)(y); y = layers.LeakyReLU()(y)

        y = layers.Conv2DTranspose(32, (4,4), strides=(2,2), padding='same')(y); y = layers.LeakyReLU()(y)
        y = layers.Conv2D(32, (5,1), padding='same', kernel_initializer=self.initializer)(y); y = layers.LeakyReLU()(y)

        y = layers.Conv2DTranspose(1, (4,4), strides=(2,4), padding='same')(y)
        y = tf.keras.activations.tanh(y)
        y = tf.squeeze(y, axis=-1)  # (B, 400, 2048)

        self.decoder = tf.keras.Model([z_in, c_in_dec], y, name="decoder")

        # --- Priors por clase (aprendidos) ---
        # Forma: (num_classes, latent_dim)
        self.prior_mu = self.add_weight(
            name="prior_mu", shape=(num_classes, latent_dim),
            initializer=tf.keras.initializers.Zeros(), trainable=True)
        self.prior_logvar = self.add_weight(
            name="prior_logvar", shape=(num_classes, latent_dim),
            initializer=tf.keras.initializers.Zeros(), trainable=True)

        self.file_writer = tf.summary.create_file_writer(tensorboard_logs)

    # ---------- utilidades de clase ----------
    @staticmethod
    def _labels_to_onehot(c_inp, num_classes):
        """
        Acepta: (B,) con valores {2,4} o (B,1) o escalar (clase única por batch),
        o ya (B,num_classes) one-hot.
        Devuelve: (B, num_classes) one-hot, y (B,) índices [0..num_classes-1]
        con el mapeo 2->0, 4->1.
        """
        c = c_inp
        c = tf.convert_to_tensor(c)
        # si ya es one-hot:
        if c.shape.rank == 2 and c.shape[-1] == num_classes:
            idx = tf.argmax(c, axis=-1, output_type=tf.int32)
            return tf.cast(c, tf.float32), idx

        # si es escalar o vector de etiquetas 2/4
        if c.shape.rank == 0:
            c = tf.reshape(c, [1])
        c = tf.cast(tf.reshape(c, [-1]), tf.int32)  # (B,)
        # mapeo {2->0, 4->1}
        idx = tf.where(tf.equal(c, 4), tf.ones_like(c), tf.zeros_like(c))
        onehot = tf.one_hot(idx, depth=num_classes, dtype=tf.float32)
        return onehot, tf.cast(idx, tf.int32)

    def call(self, inputs, training=None):
        x, c_inp = inputs  # x: (B, 400, 2048), c_inp: ver _labels_to_onehot
        c_oh, _ = self._labels_to_onehot(c_inp, self.num_classes)
        z, z_mean, z_log_var = self.encoder([x, c_oh], training=training)
        recon = self.decoder([z, c_oh], training=training)
        return recon, z_mean, z_log_var, c_oh

    # ---------- pérdidas auxiliares (idénticas a tu código) ----------
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
        if hop is None: hop = n_fft // 4
        x = tf.cast(x, tf.float32)              # (B, T, N)
        B = tf.shape(x)[0]; T = tf.shape(x)[1]; N = tf.shape(x)[2]
        flat = tf.reshape(x, [B*N, T])          # (B*N, T)
        X = tf.signal.stft(flat, frame_length=n_fft, frame_step=hop,
                           window_fn=tf.signal.hann_window, pad_end=True)
        L = tf.math.log(tf.abs(X) + 1e-6)       # (B*N, F, frames)
        return L

    def stft_loss(self, y_true, y_pred, n_fft=256):
        Lt = self._stft_logmag(y_true, n_fft)
        Lp = self._stft_logmag(y_pred, n_fft)
        return tf.reduce_mean(tf.abs(Lt - Lp))

    def _grad_l1(self, y_true, y_pred):
        dy_t = y_true[:, 1:, :] - y_true[:, :-1, :]
        dy_p = y_pred[:, 1:, :] - y_pred[:, :-1, :]
        return tf.reduce_mean(tf.abs(dy_t - dy_p))

    def _hf_match_and_spurious(self, y_true, y_pred, n_fft=256, cutoff=0.4):
        Lt = self._stft_logmag(y_true, n_fft)
        Lp = self._stft_logmag(y_pred, n_fft)

        F  = tf.shape(Lt)[1]
        freqs = tf.cast(tf.range(F), tf.float32) / tf.cast(tf.maximum(F-1, 1), tf.float32)
        mask_hf = tf.cast(freqs >= cutoff, tf.float32)
        mask_hf = tf.reshape(mask_hf, [1, -1, 1])

        Lt_hf = Lt * mask_hf
        denom = tf.stop_gradient(tf.reduce_mean(tf.abs(Lt_hf), axis=1, keepdims=True) + 1e-6)
        w_t = tf.clip_by_value(tf.abs(Lt_hf) / denom, 0.0, 5.0)

        hf_match = tf.reduce_mean(w_t * tf.abs(Lp - Lt))
        w_spur = mask_hf * tf.cast(w_t <= 0.2, tf.float32)
        hf_spurious = tf.reduce_mean(w_spur * tf.nn.relu(Lp))
        return hf_match, hf_spurious

    def recon_loss_hf_preserve(self, y_true, y_pred):
        mse  = tf.reduce_mean(tf.square(y_true - y_pred))
        grad = self._grad_l1(y_true, y_pred)
        hf_m1, hf_s1 = self._hf_match_and_spurious(y_true, y_pred, n_fft=256, cutoff=0.40)
        hf_m2, hf_s2 = self._hf_match_and_spurious(y_true, y_pred, n_fft=512, cutoff=0.50)
        hf_match    = 0.5 * (hf_m1 + hf_m2)
        hf_spurious = 0.5 * (hf_s1 + hf_s2)
        return 0.35*mse + 0.30*grad + 0.25*hf_match + 0.10*hf_spurious

    # ---------- KL contra prior por clase ----------
    @staticmethod
    def _kl_gaussians(mu_q, logvar_q, mu_p, logvar_p):
        """
        KL( N(mu_q, Sigma_q) || N(mu_p, Sigma_p) )
        mu*: (B, D), logvar*: (B, D)
        """
        var_q = tf.exp(tf.clip_by_value(logvar_q, -10.0, 10.0))
        var_p = tf.exp(tf.clip_by_value(logvar_p, -10.0, 10.0))
        term1 = tf.math.log(var_p) - tf.math.log(var_q)          # (B,D)
        term2 = (var_q + tf.square(mu_q - mu_p)) / var_p         # (B,D)
        kl = 0.5 * tf.reduce_sum(term1 - 1.0 + term2, axis=1)    # (B,)
        return tf.reduce_mean(kl)

    def compute_loss_VAE(self, y_pred, y_true, z_mean, z_log_var, c_idx, free_bits=0.5):
        recon = self.recon_loss_hf_preserve(y_true, y_pred)
        corr = self.negative_corr_loss(y_true, y_pred)

        # Prior por clase para cada muestra del batch
        mu_p = tf.gather(self.prior_mu, c_idx)          # (B, D)
        logvar_p = tf.gather(self.prior_logvar, c_idx)  # (B, D)

        kl = self._kl_gaussians(z_mean, z_log_var, mu_p, logvar_p)
        kl = tf.maximum(kl, free_bits)  # free-bits para estabilidad

        total = recon + self.beta * kl + 0.5 * corr
        return total, recon, kl

    # ---------- train/test ----------
    def train_step(self, data):
        # data puede venir como ((x, c_inp), y_true) o (x, y_true, c_inp)
        if isinstance(data[0], (tuple, list)) and len(data) == 2:
            (x, c_inp), y_true = data
        else:
            # fallback a (x, y_true, c_inp)
            x, y_true, c_inp = data

        # One-hot e índices por clase
        c_oh, c_idx = self._labels_to_onehot(c_inp, self.num_classes)

        tf.debugging.assert_all_finite(x, "Input x has NaN/Inf")
        tf.debugging.assert_all_finite(y_true, "Target y_true has NaN/Inf")

        step = int(self.optimizer.iterations)
        warm_steps = self.params.get("beta_warmup_steps", 3330)
        beta_max = float(self.params.get("beta_max", 3.0))
        frac = min(step / max(1, warm_steps), 1.0)
        self.beta.assign(frac * beta_max)

        with tf.GradientTape() as tape:
            y_pred, z_mean, z_log_var, _ = self([x, c_oh], training=True)
            tf.debugging.assert_all_finite(z_mean, "z_mean NaN/Inf")
            tf.debugging.assert_all_finite(z_log_var, "z_log_var NaN/Inf")
            tf.debugging.assert_all_finite(y_pred, "y_pred NaN/Inf")

            total, recon, kl = self.compute_loss_VAE(y_pred, y_true, z_mean, z_log_var, c_idx)

        grads = tape.gradient(total, self.trainable_variables)
        for g in grads:
            if g is not None:
                tf.debugging.assert_all_finite(g, "Gradient NaN/Inf")
        grads, _ = tf.clip_by_global_norm(grads, 1.0)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        for v in self.trainable_variables:
            tf.debugging.assert_all_finite(v, f"Weight {v.name} became NaN/Inf")

        return {"total_loss": total, "recon": recon, "kl_loss": kl, "beta": self.beta}

    def test_step(self, data):
        if isinstance(data[0], (tuple, list)) and len(data) == 2:
            (x, c_inp), y_true = data
        else:
            x, y_true, c_inp = data
        c_oh, c_idx = self._labels_to_onehot(c_inp, self.num_classes)

        y_pred, z_mean, z_log_var, _ = self([x, c_oh], training=False)
        total, recon, kl = self.compute_loss_VAE(y_pred, y_true, z_mean, z_log_var, c_idx)
        return {"total_loss": total, "recon": recon, "kl_loss": kl, "beta": self.beta}

    # ---------- helpers de uso ----------
    def encode(self, x, c_inp):
        c_oh, _ = self._labels_to_onehot(c_inp, self.num_classes)
        return self.encoder([x, c_oh], training=False)

    def decode(self, z, c_inp):
        c_oh, _ = self._labels_to_onehot(c_inp, self.num_classes)
        return self.decoder([z, c_oh], training=False)
