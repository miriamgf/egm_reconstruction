import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import tensorflow as tf

from tools_.tools_1 import normalize_array


class SyntheticDataGenerator:
    def __init__(self, vae_model, latent_dim, save_dir, experiment_dir, params,
                 rng_seed=42, sampling="guided", conditional_classes=None):
        """
        Parameters:
        - vae_model: instancia del VAE ya cargado (con pesos).
        - latent_dim: dimensión del espacio latente (int).
        - save_dir: carpeta donde se guardarán las señales generadas (.npy).
        - params: diccionario de hiperparámetros, se usa para el nombre de archivo.
        - rng_seed: semilla para reproducibilidad en muestreos.
        - sampling: 'random' | 'guided' | 'interpolated'
        - conditional_classes: lista de clases a generar (ej: [2,4]) si el VAE es condicional.
        """
        self.vae = vae_model
        self.latent_dim = int(latent_dim)
        self.save_dir = save_dir
        self.experiment_dir=experiment_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.model_name = params.get('experiment_name', 'experiment')
        self.params = params
        self.rng = np.random.default_rng(rng_seed)
        self.sampling = sampling
        self.conditional_classes = conditional_classes or []  # por defecto vacío

    # ---------------------------- Helpers API ----------------------------
    @staticmethod
    def _to_np(x):
        if isinstance(x, np.ndarray):
            return x
        try:
            return x.numpy()
        except Exception:
            return np.array(x)

    def _decode(self, z_batch, y_batch=None, num_classes=None):
        """
        Decodifica un batch latente z_batch -> señales.
        Maneja tanto VAEs estándar como condicionales.
        """
        try:
            if y_batch is not None:
                if num_classes is None:
                    raise ValueError("Debes especificar num_classes si usas VAE condicional.")

                # convertir clases enteras -> one-hot
                y_batch = tf.one_hot(y_batch, depth=num_classes, dtype=tf.float32)

                try:
                    x = self.vae.decode_from_latent([z_batch, y_batch])
                except Exception:
                    x = self.vae.decoder([z_batch, y_batch], training=False)
            else:
                try:
                    x = self.vae.decode_from_latent(z_batch)
                except Exception:
                    x = self.vae.decoder(z_batch, training=False)
        except Exception as e:
            raise RuntimeError(f"Decoder no aceptó las variantes condicionales ni estándar: {e}")
        return self._to_np(x)


    # ---------------------------- Sampling ------------------------------
    def sample_latent_vectors(self, num_samples, mu=None, sigma=None, z_mean_train=None):
        if self.sampling == "random":
            return self.rng.normal(loc=0.0, scale=1.0, size=(num_samples, self.latent_dim))

        elif self.sampling == "guided":
            if mu is None or sigma is None:
                raise ValueError("Para 'guided' debes pasar mu y sigma.")
            return self.rng.normal(loc=mu, scale=sigma, size=(num_samples, self.latent_dim))

        elif self.sampling == "interpolated":
            if z_mean_train is None or len(z_mean_train) < 2:
                raise ValueError("Para 'interpolated' necesitas z_mean_train con al menos 2 elementos.")
            idx1, idx2 = self.rng.choice(z_mean_train.shape[0], size=2, replace=False)
            z1, z2 = z_mean_train[idx1], z_mean_train[idx2]
            alphas = np.linspace(0, 1, num_samples)
            return np.stack([(1 - a) * z1 + a * z2 for a in alphas], axis=0)

        else:
            raise ValueError(f"sampling desconocido: {self.sampling}")

    # ---------------------------- Decoding ------------------------------
    def decode_latent_vectors(self, z_samples, y_class=None, batch_size=64,
                            squeeze=True, num_classes=None):
        """
        Decodifica vectores latentes a señales sintéticas.
        Si es condicional, convierte la clase a one-hot.
        """
        z_samples = np.asarray(z_samples, dtype=np.float32)
        n = z_samples.shape[0]
        out = []
        for i in range(0, n, batch_size):
            z_b = z_samples[i:i+batch_size]
            if y_class is not None:
                y_b = np.full((z_b.shape[0],), y_class, dtype=np.int32)
                xb = self._decode(z_b, y_batch=y_b, num_classes=num_classes)
            else:
                xb = self._decode(z_b)
            out.append(self._to_np(xb))
        X = np.concatenate(out, axis=0)
        if squeeze:
            X = np.squeeze(X)
        return X


    # ------------------------------ I/O --------------------------------
    def save_signals(self, signals, filename):
        save_path = os.path.join(self.save_dir, filename)
        np.save(save_path, signals)
        print(f"Saved {signals.shape[0]} synthetic signals at {save_path}")

    def plot_examples(self, signals, experiment_dir, class_complexity_list_train,
                      real_signals=None, num_examples=5,
                      max_nodes=5, prefix="synthetic", c=None):
        
        if c:
            classes_train=class_complexity_list_train[:].astype(np.int32)
            idx = np.where(classes_train == c)[0]
            # selecciona esas muestras de y_train
            y = real_signals[idx]
        else:
            y=real_signals

        signals = np.asarray(signals)
        n = min(num_examples, signals.shape[0])

        for i in range(n):
            sig = signals[i]
            sig_norm = normalize_array(sig, high=1, low=-1)

            plt.figure(figsize=(8, 4))
            plt.imshow(sig_norm, aspect="auto", cmap="viridis")
            plt.colorbar(label="Amplitude")
            plt.title(f"{prefix} example {i} (heatmap)")
            plt.xlabel("Nodes")
            plt.ylabel("Time")
            fpath = os.path.join(experiment_dir, f"{prefix}_heatmap_{i}.png")
            plt.savefig(fpath, dpi=150, bbox_inches="tight")
            plt.close()

            plt.figure(figsize=(10, 6))
            for node in range(min(max_nodes, sig.shape[1])):
                plt.plot(sig_norm[:, node], label=f"Synth node {node}", alpha=0.7)
                if y is not None:
                    plt.plot(y[0, :, node], "--", alpha=0.5, label=f"Real node {node}")
            plt.title(f"{prefix} example {i} (1D signals)")
            plt.xlabel("Time samples")
            plt.ylabel("Amplitude")
            plt.legend(loc="upper right", fontsize="x-small")
            if c:
                fpath = os.path.join(experiment_dir, f"{prefix}_1D_{i}_norm_{c}.png")
            else:
                fpath = os.path.join(experiment_dir, f"{prefix}_1D_{i}_norm.png")


            plt.savefig(fpath, dpi=150, bbox_inches="tight")
            plt.close()

        print(f"[INFO] Guardadas {n} figuras de ejemplos en {fpath}")
    
        # ----------------------------- Scoring ------------------------------
    def compute_rmse(self, synthetic_signals, real_signals,
                     chunk_size=16, show_progress=True):
        """
        Calcula el RMSE mínimo de cada señal sintética respecto al conjunto de reales.
        synthetic_signals: (S, T, N)
        real_signals: (R, T, N)
        return: lista de tuplas (idx_synthetic, min_rmse)
        """
        S, T, N = synthetic_signals.shape
        R = real_signals.shape[0]
        rmse_min_per_synth = []

        rng_iter = range(0, S, chunk_size)
        if show_progress:
            rng_iter = tqdm(rng_iter, total=int(np.ceil(S/chunk_size)), desc="RMSE")

        for s0 in rng_iter:
            s1 = min(s0 + chunk_size, S)
            syn = synthetic_signals[s0:s1]              # (s, T, N)
            diff = syn[:, None, :, :] - real_signals[None, :, :, :]  # (s,R,T,N)
            mse = np.mean(diff**2, axis=(2,3))          # (s,R)
            rmse = np.sqrt(mse)                         # (s,R)
            min_rmse = np.min(rmse, axis=1)             # (s,)
            rmse_min_per_synth.extend(min_rmse.tolist())

        return list(zip(range(S), rmse_min_per_synth))

    def select_best_signals(self, synthetic_signals, rmse_list, num_to_keep=100):
        """
        Selecciona las señales sintéticas con menor RMSE.
        """
        rmse_list_sorted = sorted(rmse_list, key=lambda x: x[1])
        best_idxs = [idx for idx, _ in rmse_list_sorted[:num_to_keep]]
        return synthetic_signals[best_idxs]

    # ---------------------------- Pipeline ------------------------------
    def generate_for_classes(self, num_per_class=100, z_mean_train=None,
                         decode_batch_size=64, save_prefix="synthetic",
                         plot_real_signals=None, class_labels=None,
                         select_best=False, num_selected=50,
                         num_classes=None):
        """
        Genera N señales por clase en self.conditional_classes.
        - select_best=True: selecciona las mejores señales según RMSE vs reales.
        - num_selected: nº de señales a conservar si select_best=True.
        - num_classes: nº de clases usado en el one-hot (ej: 10 si el VAE es sobre 0-9).
        """
        if not self.conditional_classes:
            raise ValueError("Debes definir conditional_classes en el constructor.")

        all_signals = {}
        for cls in self.conditional_classes:
            print(f"[INFO] Generando {num_per_class} señales para clase {cls}...")

            # estadísticas para sampling guided
            mu = np.mean(z_mean_train, axis=0) if z_mean_train is not None else None
            sigma = np.std(z_mean_train, axis=0) if z_mean_train is not None else None

            # sampleo en latente
            z_samples = self.sample_latent_vectors(
                num_per_class, mu=mu, sigma=sigma, z_mean_train=z_mean_train
            )

            # decodificación
            signals = self.decode_latent_vectors(
                z_samples, y_class=cls,
                batch_size=decode_batch_size,
                squeeze=True,
                num_classes=num_classes
            )

            # ---------------- selección opcional ----------------
            if select_best and (plot_real_signals is not None) and (class_labels is not None):
                idx_real = np.where(np.asarray(class_labels) == cls)[0]
                if idx_real.size > 0:
                    real_cls = plot_real_signals[idx_real]
                    rmse_list = self.compute_rmse(signals, real_cls)
                    signals = self.select_best_signals(signals, rmse_list, num_to_keep=num_selected)
                    print(f"[INFO] Seleccionadas {signals.shape[0]} mejores señales para clase {cls}")
            # -----------------------------------------------------

            # guardado
            fname = f"{save_prefix}_class{cls}_{self.model_name}_{self.sampling}.npy"
            self.save_signals(signals, fname)

            # ejemplos
            self.plot_examples(
                signals,
                self.experiment_dir,
                class_complexity_list_train=class_labels,
                real_signals=plot_real_signals,
                num_examples=5,
                prefix=f"{save_prefix}_class{cls}",
                c=cls
            )

            all_signals[cls] = signals

        return all_signals



    
    

