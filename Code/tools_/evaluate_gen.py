# tools_/evaluate_gen.py

import os
import json
import numpy as np
from numpy.fft import rfft
from scipy import linalg
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import silhouette_score

import tensorflow as tf


class EvaluateGen:
    """
    Evaluación para VAEs generativos de señales 2D (tiempo x nodos).

    Métricas:
      - Reconstrucción: MSE, correlación y Log-Spectral Distance (LSD) *nodo a nodo* y agregados globales
      - Generativo: FID-like y MMD (RBF) en embeddings (z_mean)
      - Diversidad/Cobertura: dispersión intra y cobertura en PCA(2D)
      - Latente: KL medio, Active Units (AU), opcionalmente silhouette/linear probe si hay labels

    Uso mínimo:
        evaluator = EvaluateGen(vae, latent_dim=250, save_dir=experiment_dir)
        results = evaluator.run_all(y_test, labels_test=class_complexity_list_test)

    Parámetros:
      - vae: objeto con .encoder y .decoder (o .decode_from_latent) y opcional .model
      - latent_dim: dim. espacio latente del VAE
      - save_dir: si no es None, guarda JSON con resultados
    """
    def __init__(self, vae, latent_dim, save_dir=None, rng_seed=42):
        self.vae = vae
        self.latent_dim = int(latent_dim)
        self.save_dir = save_dir
        self.rng = np.random.default_rng(rng_seed)

    # ----------------------- Helpers de compatibilidad -----------------------
    def _encode(self, x):
        """
        Devuelve (z, z_mean, z_log_var) o intenta derivarlos con las variantes disponibles.
        """
        # Variante build_encoder_module(x, input_shape)
        try:
            z, z_mean, z_log_var = self.vae.build_encoder_module(
                x, getattr(self.vae, "input_shape_", x.shape[1:])
            )
            return self._to_numpy(z), self._to_numpy(z_mean), self._to_numpy(z_log_var)
        except Exception:
            pass

        # Variante encoder(x)
        out = self.vae.encoder(x, training=False)
        if isinstance(out, (tuple, list)):
            if len(out) == 3:
                z, z_mean, z_log_var = out
            elif len(out) == 2:
                z, z_mean = out
                # Si no hay log_var, fabricamos algo razonable (cuidado con KL)
                z_log_var = tf.zeros_like(z_mean)
            else:
                # Un solo tensor -> asumimos z_mean
                z = out
                z_mean = out
                z_log_var = tf.zeros_like(z_mean)
        else:
            z = out
            z_mean = out
            z_log_var = tf.zeros_like(z_mean)
        return self._to_numpy(z), self._to_numpy(z_mean), self._to_numpy(z_log_var)

    def _decode_from_latent(self, z):
        try:
            x = self.vae.decode_from_latent(z)
            return self._to_numpy(x)
        except Exception:
            x = self.vae.decoder(z, training=False)
            return self._to_numpy(x)

    def _reconstruct(self, x):
        # Si el modelo completo existe
        try:
            xr = self.vae.model.predict(x, verbose=0)
            return self._to_numpy(xr)
        except Exception:
            # encode -> decode (determinista con z_mean)
            _, z_mean, _ = self._encode(x)
            return self._decode_from_latent(z_mean)

    @staticmethod
    def _to_numpy(x):
        if x is None:
            return None
        if isinstance(x, np.ndarray):
            return x
        try:
            return x.numpy()
        except Exception:
            return np.array(x)

    # ----------------------------- Métricas (globales antiguas) ----------------------------------
    # ----------------------------- Métricas (globales antiguas) ----------------------------------
    @staticmethod
    def _mmd_rbf(X, Y, sigma=None, use_biased=True, eps=1e-12):
        """
        MMD^2 con kernel RBF entre dos conjuntos de embeddings.
        X, Y: arrays (n_x, d) y (n_y, d).
        sigma: ancho del kernel. Si None -> heurística de la mediana.
        use_biased: True -> estimador sesgado (habitual). False -> no sesgado.
        Devuelve: (mmd2, sigma_usado)
        """
        X = np.asarray(X); Y = np.asarray(Y)
        nx, ny = len(X), len(Y)
        if nx == 0 or ny == 0:
            return float("nan"), float(1.0)

        Z = np.vstack([X, Y])
        D = pairwise_distances(Z, Z, metric="euclidean")
        D2 = D * D

        if sigma is None:
            iu = np.triu_indices_from(D, k=1)
            d_nonzero = D[iu]
            d_nonzero = d_nonzero[d_nonzero > 0]
            sigma = np.median(d_nonzero) if d_nonzero.size else 1.0

        gamma = 1.0 / (2.0 * sigma * sigma + eps)
        K = np.exp(-gamma * D2)

        Kxx = K[:nx, :nx]
        Kyy = K[nx:, nx:]
        Kxy = K[:nx, nx:]

        if use_biased:
            mmd2 = Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean()
        else:
            kxx = (Kxx.sum() - np.trace(Kxx)) / (nx * (nx - 1) + eps) if nx > 1 else 0.0
            kyy = (Kyy.sum() - np.trace(Kyy)) / (ny * (ny - 1) + eps) if ny > 1 else 0.0
            kxy = Kxy.mean()
            mmd2 = kxx + kyy - 2.0 * kxy

        return float(mmd2), float(sigma)


    @staticmethod
    def _mse_per_sample(X, Xr):
        """MSE global (promedio en tiempo y nodos) por muestra -> (B,)"""
        return np.mean((X - Xr) ** 2, axis=(1, 2))

    @staticmethod
    def _corr2d(a, b, eps=1e-8):
        """Correlación global 2D entre (T,N) y (T,N)"""
        am = a - a.mean()
        bm = b - b.mean()
        num = (am * bm).sum()
        den = np.sqrt((am ** 2).sum() * (bm ** 2).sum()) + eps
        return num / den

    def _corr_per_sample(self, X, Xr):
        """Correlación global por muestra -> (B,)"""
        return np.array([self._corr2d(X[i], Xr[i]) for i in range(len(X))])

    @staticmethod
    def _log_spectral_distance(x, y, eps=1e-8):
        """
        LSD global por muestra.
        x,y: (T x N). RFFT en eje tiempo; promedia en frecuencias y nodos.
        """
        X = np.abs(rfft(x, axis=0)) + eps
        Y = np.abs(rfft(y, axis=0)) + eps
        lsd = np.sqrt(((np.log(X) - np.log(Y)) ** 2).mean(axis=0))
        return float(lsd.mean())  # media sobre nodos

    def _lsd_per_sample(self, X, Xr):
        """LSD global por muestra -> (B,)"""
        return np.array([self._log_spectral_distance(X[i], Xr[i]) for i in range(len(X))])

    # ----------------------------- Métricas nodo a nodo (nuevas) ---------------------------------
    @staticmethod
    def _frechet_distance(mu1, cov1, mu2, cov2, eps=1e-6):
        diff = mu1 - mu2
        covmean, _ = linalg.sqrtm(
            (cov1 + np.eye(cov1.shape[0]) * eps) @ (cov2 + np.eye(cov2.shape[0]) * eps),
            disp=False
        )
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        return float(diff.dot(diff) + np.trace(cov1 + cov2 - 2 * covmean))
    
    @staticmethod
    def _mse_per_sample_per_node(X, Xr):
        """
        X, Xr: (B, T, N)  ->  (B, N) MSE por nodo (promediando en el eje tiempo).
        """
        X = np.asarray(X); Xr = np.asarray(Xr)
        return np.mean((X - Xr) ** 2, axis=1)  # (B, N)

    @staticmethod
    def _corr_per_sample_per_node(X, Xr, eps=1e-8):
        """
        Correlación Pearson por muestra y por nodo.
        X, Xr: (B, T, N)  ->  (B, N)
        """
        X = np.asarray(X); Xr = np.asarray(Xr)
        # centrar en el eje tiempo
        Xc  = X  - X.mean(axis=1, keepdims=True)
        Xrc = Xr - Xr.mean(axis=1, keepdims=True)
        num = (Xc * Xrc).sum(axis=1)                             # (B, N)
        den = np.sqrt((Xc**2).sum(axis=1) * (Xrc**2).sum(axis=1)) + eps  # (B, N)
        return num / den

    @staticmethod
    def _lsd_per_sample_per_node(X, Xr, eps=1e-8):
        """
        Log-Spectral Distance por muestra y por nodo.
        X, Xr: (B, T, N)  ->  (B, N)
        RFFT a lo largo del eje tiempo, luego promedio en frecuencias por nodo.
        """
        X = np.asarray(X); Xr = np.asarray(Xr)
        # RFFT en eje tiempo -> (B, F, N)
        FX  = np.abs(rfft(X,  axis=1)) + eps
        FXr = np.abs(rfft(Xr, axis=1)) + eps
        # LSD por nodo para cada muestra: sqrt(mean_f (log(FX)-log(FXr))^2)
        lsd_bn = np.sqrt(((np.log(FX) - np.log(FXr)) ** 2).mean(axis=1))  # (B, N)
        return lsd_bn

    @staticmethod
    def _grid_coverage(Pxy, bins=20):
        H, _, _ = np.histogram2d(Pxy[:, 0], Pxy[:, 1], bins=bins)
        return float((H > 0).mean())

    # ----------------------------- Bloques -----------------------------------
    def reconstruction_metrics(self, X):
        """
        X: y_test (B x T x N)
        Devuelve métricas nodo a nodo (+ agregados globales).
        """
        X = self._to_numpy(X)
        Xr = self._reconstruct(X)  # (B, T, N)

        # --- Nodo a nodo por muestra (B, N) ---
        mse_bn  = self._mse_per_sample_per_node(X, Xr)            # (B, N)
        corr_bn = self._corr_per_sample_per_node(X, Xr)           # (B, N)
        lsd_bn  = self._lsd_per_sample_per_node(X, Xr)            # (B, N)

        # --- Agregamos sobre el batch -> por nodo ---
        mse_node_mean  = mse_bn.mean(axis=0)                      # (N,)
        mse_node_std   = mse_bn.std(axis=0)                       # (N,)
        corr_node_mean = corr_bn.mean(axis=0)                     # (N,)
        corr_node_std  = corr_bn.std(axis=0)                      # (N,)
        lsd_node_mean  = lsd_bn.mean(axis=0)                      # (N,)
        lsd_node_std   = lsd_bn.std(axis=0)                       # (N,)

        # (Opcional) métricas globales (promediando nodos de los promedios por nodo)
        mse_global_mean  = float(mse_node_mean.mean())
        mse_global_std   = float(mse_node_mean.std())
        corr_global_mean = float(corr_node_mean.mean())
        corr_global_std  = float(corr_node_mean.std())
        lsd_global_mean  = float(lsd_node_mean.mean())
        lsd_global_std   = float(lsd_node_mean.std())

        # Para JSON: convertir arrays a listas
        return {

            # globales opcionales
            "mse_global_mean":  mse_global_mean,
            "mse_global_std":   mse_global_std,
            "corr_global_mean": corr_global_mean,
            "corr_global_std":  corr_global_std,
            "lsd_global_mean":  lsd_global_mean,
            "lsd_global_std":   lsd_global_std,
        }
    
    

    def generative_metrics(self, X, num_generated=None):
        """
        Compara reales vs generados en embeddings (z_mean).
        """
        if num_generated is None:
            num_generated = len(X)

        # Embeddings reales
        _, z_mean_real, _ = self._encode(X)
        # Genera desde el prior
        z = tf.random.normal((num_generated, self.latent_dim))
        Xg = self._decode_from_latent(z)
        # Embeddings de sintéticas
        _, z_mean_fake, _ = self._encode(Xg)

        # FID-like
        m1, C1 = z_mean_real.mean(axis=0), np.cov(z_mean_real, rowvar=False)
        m2, C2 = z_mean_fake.mean(axis=0), np.cov(z_mean_fake, rowvar=False)
        fid_like = self._frechet_distance(m1, C1, m2, C2)

        # MMD
        mmd2, sigma = self._mmd_rbf(z_mean_real, z_mean_fake)

        return {
            "fid_like_zmean": float(fid_like),
            "mmd2_zmean": float(mmd2),
            "mmd_sigma_used": float(sigma),
        }

    def diversity_coverage_metrics(self, X, num_generated=None, pca_bins=20):
        if num_generated is None:
            num_generated = len(X)

        # Embeddings reales y sintéticos
        _, z_mean_real, _ = self._encode(X)
        z = tf.random.normal((num_generated, self.latent_dim))
        Xg = self._decode_from_latent(z)
        _, z_mean_fake, _ = self._encode(Xg)

        # Dispersión intra
        d_real = pairwise_distances(z_mean_real).mean()
        d_fake = pairwise_distances(z_mean_fake).mean()

        # Cobertura en PCA(2)
        P = PCA(n_components=2).fit_transform(
            np.vstack([z_mean_real, z_mean_fake])
        )
        Pr, Pf = P[:len(z_mean_real)], P[len(z_mean_real):]
        cov_r = self._grid_coverage(Pr, bins=pca_bins)
        cov_f = self._grid_coverage(Pf, bins=pca_bins)

        return {
            "dispersion_real": float(d_real),
            "dispersion_fake": float(d_fake),
            "pca2_coverage_real": float(cov_r),
            "pca2_coverage_fake": float(cov_f),
        }

    def latent_metrics(self, X, labels=None):
        """
        KL medio, Active Units y (opcional) silhouette/linear probe si labels no es None.
        """
        _, z_mean, z_log_var = self._encode(X)

        # KL por muestra
        logvar = np.clip(z_log_var, -10.0, 10.0)
        kl = -0.5 * np.sum(1 + logvar - np.square(z_mean) - np.exp(logvar), axis=1)

        # Active Units
        var_zj = np.var(z_mean, axis=0)
        active_units = int((var_zj > 1e-2).sum())

        out = {
            "kl_mean": float(kl.mean()),
            "kl_std": float(kl.std()),
            "active_units": active_units,
            "latent_dim": int(z_mean.shape[1]),
        }

        # Métricas supervisadas opcionales
        if labels is not None:
            labels = np.asarray(labels).reshape(-1)
            try:
                sil = silhouette_score(z_mean, labels, metric="euclidean")
                out["silhouette_zmean"] = float(sil)
            except Exception:
                pass
            # Linear probe simple
            try:
                clf = LogisticRegression(max_iter=200, n_jobs=1)
                n = len(z_mean)
                idx = np.arange(n)
                self.rng.shuffle(idx)
                split = int(0.8 * n)
                tr, te = idx[:split], idx[split:]
                clf.fit(z_mean[tr], labels[tr])
                pred = clf.predict(z_mean[te])
                out["linear_probe_acc"] = float(accuracy_score(labels[te], pred))
            except Exception:
                pass

        return out

    # ----------------------------- Runner ------------------------------------
    def run_all(self, y_test, labels_test=None, num_generated=None, pca_bins=20, save_json_name="eval_results.json"):
        """
        Ejecuta todos los bloques y devuelve un diccionario con resultados.
        """
        results = {}

        # 1) Reconstrucción (incluye nodo a nodo y globales)
        try:
            results["reconstruction"] = self.reconstruction_metrics(y_test)
        except Exception as e:
            results["reconstruction_error"] = str(e)

        # 2) Generativo (FID-like + MMD)
        try:
            results["generative"] = self.generative_metrics(y_test, num_generated=num_generated)
        except Exception as e:
            results["generative_error"] = str(e)

        # 3) Diversidad/Cobertura
        try:
            results["diversity_coverage"] = self.diversity_coverage_metrics(
                y_test, num_generated=num_generated, pca_bins=pca_bins
            )
        except Exception as e:
            results["diversity_coverage_error"] = str(e)

        # 4) Latente
        try:
            results["latent"] = self.latent_metrics(y_test, labels=labels_test)
        except Exception as e:
            results["latent_error"] = str(e)

        # Guardar JSON
        if self.save_dir is not None:
            try:
                os.makedirs(self.save_dir, exist_ok=True)
                with open(os.path.join(self.save_dir, save_json_name), "w") as f:
                    json.dump(results, f, indent=2)
            except Exception:
                pass

        return results
