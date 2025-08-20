# tools_/evaluate_gen.py
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
from numpy.fft import rfft
from scipy import linalg
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import tensorflow as tf


# ============================== Utils ==============================

def _as_numpy(x):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    try:
        return x.numpy()
    except Exception:
        return np.array(x)

def _onehot_from_int(y_int_zero_based, num_classes):
    y = np.asarray(y_int_zero_based).reshape(-1).astype(int)
    oh = np.zeros((len(y), int(num_classes)), dtype=np.float32)
    oh[np.arange(len(y)), y] = 1.0
    return oh

# ============================== Evaluador ==============================

class EvaluateGen:
    """
    Evaluación para VAEs generativos (condicional/no condicional).
    Corre métricas por-clase para las clases indicadas (por defecto 2 y 4).
    """
    def __init__(self, vae, latent_dim, save_dir=None, rng_seed=42,
                 conditional_classes=(2, 4)):
        self.vae = vae
        self.latent_dim = int(latent_dim)
        self.save_dir = save_dir
        self.rng = np.random.default_rng(rng_seed)

        # ---- Remapeo de clases externas -> índices [0..K-1] para el modelo ----
        self.conditional_classes = tuple(conditional_classes)   # p.ej. (2,4)
        self.class_map = {c: i for i, c in enumerate(self.conditional_classes)}  # {2:0, 4:1}
        self.inv_class_map = {i: c for c, i in self.class_map.items()}
        self.num_labels = len(self.conditional_classes)  # tamaño del one-hot

    # ----------------------- Compatibilidad encoder/decoder -----------------------
    def _map_labels_zero_based(self, y_ext):
        """Convierte etiquetas externas (p.ej. {2,4}) a índices {0,1}."""
        y_ext = np.asarray(y_ext).reshape(-1)
        try:
            y_zero = np.vectorize(self.class_map.__getitem__)(y_ext)
        except KeyError as e:
            raise ValueError(f"Etiqueta externa no reconocida {e.args[0]}; "
                             f"esperaba una de {list(self.class_map.keys())}")
        return y_zero.astype(int)

    def _encode(self, x, y_ext=None):
        """
        Llama al encoder con o sin condición.
        y_ext: etiquetas externas (2/4), se remapean a 0/1 si no es None.
        """
        if y_ext is not None:
            y_zero = self._map_labels_zero_based(y_ext)
            y_oh = _onehot_from_int(y_zero, self.num_labels)
            # probar firmas comunes
            for lbl in (y_oh, y_zero):
                try:
                    out = self.vae.encoder([x, lbl], training=False); break
                except Exception:
                    try:
                        out = self.vae.encoder(x, lbl, training=False); break
                    except Exception:
                        continue
            else:
                raise RuntimeError("encoder condicional requiere 2 entradas, no aceptó ninguna variante.")
        else:
            out = self.vae.encoder(x, training=False)

        if isinstance(out, (tuple, list)):
            if len(out) == 3:
                z, z_mean, z_log_var = out
            elif len(out) == 2:
                z, z_mean = out; z_log_var = tf.zeros_like(z_mean)
            else:
                z = out; z_mean = out; z_log_var = tf.zeros_like(z)
        else:
            z = out; z_mean = out; z_log_var = tf.zeros_like(z)
        return _as_numpy(z), _as_numpy(z_mean), _as_numpy(z_log_var)

    def _decode_from_latent(self, z, y_ext=None):
        """
        Decodificación condicional robusta:
        - Intenta dos entradas: [z, y] o (z, y)
        - Si falla, concatena z || onehot(y) y llama con una sola entrada.
        - Si no hay y_ext, usa el decoder no condicional.
        """
        z = _as_numpy(z)

        # --- caso no condicional ---
        if y_ext is None:
            # decode_from_latent -> decoder
            try:
                x = self.vae.decode_from_latent(z); return _as_numpy(x)
            except Exception:
                x = self.vae.decoder(z, training=False); return _as_numpy(x)

        # --- preparar etiquetas (remapeo 2/4 -> 0/1 y one-hot) ---
        y_zero = self._map_labels_zero_based(y_ext)                 # (B,)
        y_oh   = _onehot_from_int(y_zero, self.num_labels).astype(z.dtype)  # (B,num_labels)

        # --- 1) intentos con dos entradas ---
        for lbl in (y_oh, y_zero):
            try:
                x = self.vae.decode_from_latent([z, lbl]); return _as_numpy(x)
            except Exception:
                try:
                    x = self.vae.decode_from_latent(z, lbl); return _as_numpy(x)
                except Exception:
                    try:
                        x = self.vae.decoder([z, lbl], training=False); return _as_numpy(x)
                    except Exception:
                        try:
                            x = self.vae.decoder(z, lbl, training=False); return _as_numpy(x)
                        except Exception:
                            pass  # seguimos probando

        # --- 2) fallback: concatenar (z || onehot(y)) y pasar una sola entrada ---
        try:
            z_cat = np.concatenate([z, y_oh], axis=1)  # (B, latent_dim + num_labels)

            # Heurística: si el decoder espera exactamente ese tamaño, úsalo.
            # (si falla la inspección, igualmente intentamos ambos)
            tried_err = []

            try:
                x = self.vae.decode_from_latent(z_cat); return _as_numpy(x)
            except Exception as e:
                tried_err.append(str(e))
                try:
                    x = self.vae.decoder(z_cat, training=False); return _as_numpy(x)
                except Exception as e2:
                    tried_err.append(str(e2))
                    # intento final: algunos modelos esperan float32 estrictamente
                    if z_cat.dtype != np.float32:
                        z_cat32 = z_cat.astype(np.float32)
                        try:
                            x = self.vae.decode_from_latent(z_cat32); return _as_numpy(x)
                        except Exception:
                            x = self.vae.decoder(z_cat32, training=False); return _as_numpy(x)
                    else:
                        raise RuntimeError(
                            "decoder condicional no aceptó ninguna variante ([z,y]/(z,y)/concat). "
                            f"Errores: {tried_err}"
                        )
        except Exception:
            raise RuntimeError("decoder condicional no aceptó ninguna variante ([z,y] / (z,y)).")


    def _reconstruct(self, x, y_ext=None):
        # Usa .model si acepta entradas condicionales
        try:
            if y_ext is not None:
                y_zero = self._map_labels_zero_based(y_ext)
                y_oh = _onehot_from_int(y_zero, self.num_labels)
                out = self.vae.model.predict([x, y_oh], verbose=0)
            else:
                out = self.vae.model.predict(x, verbose=0)
            return _as_numpy(out)
        except Exception:
            # encode -> decode determinista con z_mean
            _, z_mean, _ = self._encode(x, y_ext=y_ext)
            return self._decode_from_latent(z_mean, y_ext=y_ext)

    # ----------------------------- Métricas -----------------------------------
    @staticmethod
    def _mmd_rbf(X, Y, sigma=None, use_biased=True, eps=1e-12):
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
        Kxx = K[:nx, :nx]; Kyy = K[nx:, nx:]; Kxy = K[:nx, nx:]
        if use_biased:
            mmd2 = Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean()
        else:
            kxx = (Kxx.sum() - np.trace(Kxx)) / (nx * (nx - 1) + eps) if nx > 1 else 0.0
            kyy = (Kyy.sum() - np.trace(Kyy)) / (ny * (ny - 1) + eps) if ny > 1 else 0.0
            kxy = Kxy.mean()
            mmd2 = kxx + kyy - 2.0 * kxy
        return float(mmd2), float(sigma)

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

    # ----------------------------- Reconstrucción -----------------------------------
    def reconstruction_metrics(self, X, labels=None):
        """
        Ahora acepta labels y se las pasa al modelo/encoder (obligatorio para VAE cond.).
        """
        X = _as_numpy(X)
        Xr = self._reconstruct(X, y_ext=labels)

        # (B,N) MSE por nodo
        mse_bn  = np.mean((X - Xr) ** 2, axis=1)

        # correlación por nodo
        eps = 1e-8
        Xc  = X  - X.mean(axis=1, keepdims=True)
        Xrc = Xr - Xr.mean(axis=1, keepdims=True)
        num = (Xc * Xrc).sum(axis=1)
        den = np.sqrt((Xc**2).sum(axis=1) * (Xrc**2).sum(axis=1)) + eps
        corr_bn = num / den

        # LSD por nodo
        FX  = np.abs(rfft(X,  axis=1)) + eps
        FXr = np.abs(rfft(Xr, axis=1)) + eps
        lsd_bn = np.sqrt(((np.log(FX) - np.log(FXr)) ** 2).mean(axis=1))

        # agregados
        mse_node_mean  = mse_bn.mean(axis=0)
        corr_node_mean = corr_bn.mean(axis=0)
        lsd_node_mean  = lsd_bn.mean(axis=0)

        return {
            "mse_global_mean":  float(mse_node_mean.mean()),
            "mse_global_std":   float(mse_node_mean.std()),
            "corr_global_mean": float(corr_node_mean.mean()),
            "corr_global_std":  float(corr_node_mean.std()),
            "lsd_global_mean":  float(lsd_node_mean.mean()),
            "lsd_global_std":   float(lsd_node_mean.std()),
        }

    # ----------------------------- Generativo (por clase) ----------------------------
    def generative_metrics(self, X, labels=None, num_generated=None):
        if num_generated is None:
            num_generated = len(X)

        out = {}
        classes = self.conditional_classes if labels is not None else (None,)

        for c in classes:
            if c is None:
                X_real = X; y_real = None
            else:
                mask = (np.asarray(labels).reshape(-1) == c)
                X_real = X[mask]
                y_real = np.full((len(X_real),), c, dtype=int)

            _, z_mean_real, _ = self._encode(X_real, y_ext=y_real)

            # genera condicionando la clase
            z = tf.random.normal((num_generated, self.latent_dim))
            y_gen = None if c is None else np.full((num_generated,), c, dtype=int)
            Xg = self._decode_from_latent(z, y_ext=y_gen)
            _, z_mean_fake, _ = self._encode(Xg, y_ext=y_gen)

            m1, C1 = z_mean_real.mean(axis=0), np.cov(z_mean_real, rowvar=False)
            m2, C2 = z_mean_fake.mean(axis=0), np.cov(z_mean_fake, rowvar=False)
            fid_like = self._frechet_distance(m1, C1, m2, C2)
            mmd2, sigma = self._mmd_rbf(z_mean_real, z_mean_fake)

            key = f"class_{c}" if c is not None else "all"
            out[key] = {
                "fid_like_zmean": float(fid_like),
                "mmd2_zmean": float(mmd2),
                "mmd_sigma_used": float(sigma),
                "n_real": int(len(z_mean_real)),
                "n_fake": int(len(z_mean_fake)),
            }
        return out

    # ----------------------------- Diversidad/Cobertura (por clase) -----------------
    def diversity_coverage_metrics(self, X, labels=None, num_generated=None, pca_bins=20):
        if num_generated is None:
            num_generated = len(X)

        out = {}
        classes = self.conditional_classes if labels is not None else (None,)

        for c in classes:
            if c is None:
                X_real = X; y_real = None
            else:
                mask = (np.asarray(labels).reshape(-1) == c)
                X_real = X[mask]
                y_real = np.full((len(X_real),), c, dtype=int)

            _, z_mean_real, _ = self._encode(X_real, y_ext=y_real)

            z = tf.random.normal((num_generated, self.latent_dim))
            y_gen = None if c is None else np.full((num_generated,), c, dtype=int)
            Xg = self._decode_from_latent(z, y_ext=y_gen)
            _, z_mean_fake, _ = self._encode(Xg, y_ext=y_gen)

            d_real = pairwise_distances(z_mean_real).mean()
            d_fake = pairwise_distances(z_mean_fake).mean()

            P = PCA(n_components=2).fit_transform(np.vstack([z_mean_real, z_mean_fake]))
            Pr, Pf = P[:len(z_mean_real)], P[len(z_mean_real):]
            H_r, _, _ = np.histogram2d(Pr[:, 0], Pr[:, 1], bins=pca_bins)
            H_f, _, _ = np.histogram2d(Pf[:, 0], Pf[:, 1], bins=pca_bins)
            cov_r = float((H_r > 0).mean())
            cov_f = float((H_f > 0).mean())

            key = f"class_{c}" if c is not None else "all"
            out[key] = {
                "dispersion_real": float(d_real),
                "dispersion_fake": float(d_fake),
                "pca2_coverage_real": cov_r,
                "pca2_coverage_fake": cov_f,
            }
        return out

    # ----------------------------- Latente (opcionalmente supervisado) --------------
    def latent_metrics(self, X, labels=None):
        _, z_mean, z_log_var = self._encode(X, y_ext=labels)

        logvar = np.clip(z_log_var, -10.0, 10.0)
        kl = -0.5 * np.sum(1 + logvar - np.square(z_mean) - np.exp(logvar), axis=1)
        var_zj = np.var(z_mean, axis=0)
        active_units = int((var_zj > 1e-2).sum())

        out = {
            "kl_mean": float(kl.mean()),
            "kl_std": float(kl.std()),
            "active_units": active_units,
            "latent_dim": int(z_mean.shape[1]),
        }

        if labels is not None:
            labels = np.asarray(labels).reshape(-1)
            try:
                sil = silhouette_score(z_mean, labels, metric="euclidean")
                out["silhouette_zmean"] = float(sil)
            except Exception:
                pass
            try:
                clf = LogisticRegression(max_iter=200, n_jobs=1)
                n = len(z_mean)
                idx = np.arange(n); np.random.shuffle(idx)
                split = int(0.8 * n)
                tr, te = idx[:split], idx[split:]
                clf.fit(z_mean[tr], labels[tr])
                pred = clf.predict(z_mean[te])
                out["linear_probe_acc"] = float(accuracy_score(labels[te], pred))
            except Exception:
                pass

        return out

    # ----------------------------- Runner ------------------------------------------
    def run_all(self, y_test, labels_test=None, num_generated=None, pca_bins=20,
                save_json_name="eval_results.json"):
        results = {}

        try:
            results["reconstruction"] = self.reconstruction_metrics(y_test, labels=labels_test)
        except Exception as e:
            results["reconstruction_error"] = str(e)

        try:
            results["generative"] = self.generative_metrics(
                y_test, labels=labels_test, num_generated=num_generated
            )
        except Exception as e:
            results["generative_error"] = str(e)

        try:
            results["diversity_coverage"] = self.diversity_coverage_metrics(
                y_test, labels=labels_test, num_generated=num_generated, pca_bins=pca_bins
            )
        except Exception as e:
            results["diversity_coverage_error"] = str(e)

        try:
            results["latent"] = self.latent_metrics(y_test, labels=labels_test)
        except Exception as e:
            results["latent_error"] = str(e)

        if self.save_dir is not None:
            try:
                os.makedirs(self.save_dir, exist_ok=True)
                with open(os.path.join(self.save_dir, save_json_name), "w") as f:
                    json.dump(results, f, indent=2)
            except Exception:
                pass

        return results


# ============================== Plots condicionales ==============================

def _encode_zmean_conditional(vae, X, y_ext, class_map, num_labels):
    # remapea externas -> 0..K-1 y hace one-hot
    y_zero = np.vectorize(class_map.__getitem__)(np.asarray(y_ext).reshape(-1))
    y_oh = _onehot_from_int(y_zero, num_labels)
    # probar firmas
    try:
        out = vae.encoder([X, y_oh], training=False)
    except Exception:
        out = vae.encoder(X, y_oh, training=False)
    if isinstance(out, (tuple, list)):
        z_mean = out[1] if len(out) >= 2 else out[0]
    else:
        z_mean = out
    return _as_numpy(z_mean)

def plot_pca_conditional(vae, X, class_labels, experiment_dir,
                         classes=(2, 4), filename='pca_guided_by_class_named.png'):
    class_map = {c:i for i, c in enumerate(classes)}
    num_labels = len(classes)

    Z_list, y_plot = [], []
    for c in classes:
        y_c = np.full((len(X),), c, dtype=int)
        z_mean_c = _encode_zmean_conditional(vae, X, y_c, class_map, num_labels)
        Z_list.append(z_mean_c); y_plot.append(y_c)
    Z_all = np.vstack(Z_list); y_all = np.concatenate(y_plot)

    pca = PCA(n_components=2)
    P = pca.fit_transform(Z_all)

    plt.figure(figsize=(6, 6), tight_layout=True)
    for cls in classes:
        m = (y_all == cls)
        plt.scatter(P[m, 0], P[m, 1], alpha=0.8, s=40, label=class_labels.get(cls, f"Clase {cls}"))
    ev = pca.explained_variance_ratio_
    plt.title(f"z_mean (PCA cond.) — PC1 {ev[0]*100:.1f}% · PC2 {ev[1]*100:.1f}%")
    plt.xlabel("PC1"); plt.ylabel("PC2"); plt.grid(True, linewidth=0.5); plt.legend(frameon=True)
    out = os.path.join(experiment_dir, filename)
    plt.savefig(out, dpi=160, bbox_inches='tight'); plt.close(); print(out)

def plot_tsne_conditional(vae, X, class_labels, experiment_dir,
                          classes=(2, 4), filename='tsne_guided_by_class_named.png'):
    class_map = {c:i for i, c in enumerate(classes)}
    num_labels = len(classes)

    Z_list, y_plot = [], []
    for c in classes:
        y_c = np.full((len(X),), c, dtype=int)
        z_mean_c = _encode_zmean_conditional(vae, X, y_c, class_map, num_labels)
        Z_list.append(z_mean_c); y_plot.append(y_c)
    Z_all = np.vstack(Z_list); y_all = np.concatenate(y_plot)

    Xs = StandardScaler().fit_transform(Z_all)
    perplexity = min(30, max(5, (len(Xs) - 1) // 3))
    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate="auto", init="pca", random_state=0)
    T = tsne.fit_transform(Xs)

    plt.figure(figsize=(9, 6), tight_layout=True)
    markers = ['o', 's', '^', 'D', 'P', 'X']
    for i, cls in enumerate(classes):
        m = (y_all == cls)
        plt.scatter(T[m, 0], T[m, 1], alpha=0.8, s=40, marker=markers[i % len(markers)],
                    label=class_labels.get(cls, f"Clase {cls}"))
    plt.title("z_mean (t-SNE cond.)")
    plt.xlabel("t-SNE 1"); plt.ylabel("t-SNE 2"); plt.grid(True, linewidth=0.5); plt.legend(frameon=True)
    out = os.path.join(experiment_dir, filename)
    plt.savefig(out, dpi=160, bbox_inches='tight'); plt.close(); print(out)


# ============================== Ejemplo de uso ==============================
if __name__ == "__main__":
    """
    Rellena con tus objetos/datos reales:

    vae: objeto con .encoder/.decoder/.model que aceptan [x, y_onehot]
    y_test: np.array (B,T,N)
    labels_test: np.array (B,) con valores {2,4}
    class_labels = {2: "Clase 2", 4: "Sinusal"}
    experiment_dir = "./experimentos/"
    """

    # evaluator = EvaluateGen(vae, latent_dim=250, save_dir=experiment_dir,
    #                         conditional_classes=(2, 4))
    # results = evaluator.run_all(y_test, labels_test=labels_test, num_generated=len(y_test))
    # print(json.dumps(results, indent=2))

    # plot_pca_conditional(vae, y_test, class_labels, experiment_dir, classes=(2,4))
    # plot_tsne_conditional(vae, y_test, class_labels, experiment_dir, classes=(2,4))
    pass
