import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.stats import ks_2samp, skew, kurtosis, entropy
import os

# ---------- utilidades ----------
def _ensure_2d(x):
    x = np.asarray(x)
    if x.ndim == 3:   # (N, C, T) -> usa canal 0
        x = x[:, 0, :]
    elif x.ndim == 1:
        x = x[None, :]
    return x.astype(np.float32)

def feats_time(x):
    mean = x.mean(axis=1)
    std  = x.std(axis=1)
    rms  = np.sqrt((x**2).mean(axis=1))
    p2p  = x.max(axis=1) - x.min(axis=1)
    sk   = skew(x, axis=1, bias=False)
    ku   = kurtosis(x, axis=1, fisher=True, bias=False)
    return np.stack([mean, std, rms, p2p, sk, ku], axis=1), ["mean","std","rms","p2p","skew","kurt"]

def feats_freq(x, fs, nperseg=256):
    f, Pxx = welch(x, fs=fs, nperseg=min(nperseg, x.shape[1]))
    def bandpower(P, f, fmin, fmax):
        idx = (f>=fmin) & (f<fmax)
        return np.trapz(P[:, idx], f[idx], axis=1)
    bp1 = bandpower(Pxx, f, 0.0, 5.0)
    bp2 = bandpower(Pxx, f, 5.0, 15.0)
    bp3 = bandpower(Pxx, f, 15.0, 40.0)
    total = bp1 + bp2 + bp3 + 1e-12
    rel = np.stack([bp1/total, bp2/total, bp3/total], axis=1)
    return (f, Pxx), rel, ["relPow_0-5","relPow_5-15","relPow_15-40"]

def mmd_rbf(x, y, gamma=None):
    X = x; Y = y
    if gamma is None:
        Z = np.vstack([X, Y])
        dists = np.sum(Z**2,1,keepdims=True) - 2*Z@Z.T + np.sum(Z**2,1)[None,:]
        med = np.median(dists[dists>0])
        gamma = 1.0/(2*(med+1e-12))
    def k(a,b):
        aa = np.sum(a**2,1,keepdims=True); bb = np.sum(b**2,1,keepdims=True)
        d2 = aa - 2*a@b.T + bb.T
        return np.exp(-gamma*d2)
    kxx = k(X,X); kyy = k(Y,Y); kxy = k(X,Y)
    return kxx.mean() + kyy.mean() - 2*kxy.mean()

def hist_1d(a, b, bins=60, range_=None, eps=1e-12):
    h1, edges = np.histogram(a, bins=bins, range=range_, density=True)
    h2, _     = np.histogram(b, bins=bins, range=range_, density=True)
    h1 = h1 + eps; h2 = h2 + eps
    h1 = h1 / h1.sum(); h2 = h2 / h2.sum()
    centers = 0.5*(edges[1:]+edges[:-1])
    return centers, h1, h2

def plot_and_save(fig, fname, outdir):
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, fname), bbox_inches="tight", dpi=150)
    plt.show()

def plot_hist_pair(values_real, values_synth, title, xlabel, outdir, idx):
    centers, h1, h2 = hist_1d(values_real, values_synth)
    fig = plt.figure(figsize=(7,4.5))
    plt.plot(centers, h1, label="Real")
    plt.plot(centers, h2, label="Sintético")
    plt.title(title)
    plt.xlabel(xlabel); plt.ylabel("Densidad")
    plt.legend(); plt.grid(True, alpha=0.3)
    plot_and_save(fig, f"hist_{xlabel}_{idx}.png", outdir)

# ---------- pipeline principal ----------
def compare_distributions(X_1channel, synt_X_1channel, fs,
                          outdir="/home/profes/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/Figures/Dataset_augmentation_vae"):
    X = _ensure_2d(X_1channel)
    S = _ensure_2d(synt_X_1channel)

    # --- Features en tiempo ---
    Ft_r, names_t = feats_time(X)
    Ft_s, _       = feats_time(S)

    for i, name in enumerate(names_t):
        plot_hist_pair(Ft_r[:, i], Ft_s[:, i], f"Distribución {name}", name, outdir, i)

    # --- Frecuencia ---
    (f_r, Pxx_r), Ff_r, names_f = feats_freq(X, fs)
    (f_s, Pxx_s), Ff_s, _       = feats_freq(S, fs)

    # PSD promedio Real
    fig = plt.figure(figsize=(7,4.5))
    mean_psd = Pxx_r.mean(axis=0); se_psd = Pxx_r.std(axis=0)/np.sqrt(Pxx_r.shape[0])
    plt.semilogy(f_r, mean_psd)
    plt.fill_between(f_r, np.maximum(mean_psd-se_psd, 1e-12), mean_psd+se_psd, alpha=0.2)
    plt.title("PSD promedio (Real)")
    plt.xlabel("Frecuencia [Hz]"); plt.ylabel("PSD")
    plt.grid(True, which="both", alpha=0.3)
    plot_and_save(fig, "PSD_real.png", outdir)

    # PSD promedio Sintético
    fig = plt.figure(figsize=(7,4.5))
    mean_psd = Pxx_s.mean(axis=0); se_psd = Pxx_s.std(axis=0)/np.sqrt(Pxx_s.shape[0])
    plt.semilogy(f_s, mean_psd)
    plt.fill_between(f_s, np.maximum(mean_psd-se_psd, 1e-12), mean_psd+se_psd, alpha=0.2)
    plt.title("PSD promedio (Sintético)")
    plt.xlabel("Frecuencia [Hz]"); plt.ylabel("PSD")
    plt.grid(True, which="both", alpha=0.3)
    plot_and_save(fig, "PSD_sintetico.png", outdir)

    # Comparación directa
    fig = plt.figure(figsize=(7,4.5))
    plt.semilogy(f_r, Pxx_r.mean(axis=0), label="Real")
    plt.semilogy(f_s, Pxx_s.mean(axis=0), label="Sintético")
    plt.title("PSD promedio: Real vs Sintético")
    plt.xlabel("Frecuencia [Hz]"); plt.ylabel("PSD")
    plt.grid(True, which="both", alpha=0.3); plt.legend()
    plot_and_save(fig, "PSD_real_vs_sintetico.png", outdir)

    # Distribuciones de bandas relativas
    for i, name in enumerate(names_f):
        plot_hist_pair(Ff_r[:, i], Ff_s[:, i], f"Distribución {name}", name, outdir, f"freq{i}")

    # --- PCA en features ---
    Feat_r = np.hstack([Ft_r, Ff_r])
    Feat_s = np.hstack([Ft_s, Ff_s])
    XZ = np.vstack([Feat_r, Feat_s])
    Z = XZ - XZ.mean(axis=0, keepdims=True)
    U, Ssvd, Vt = np.linalg.svd(Z, full_matrices=False)
    PC = Z @ Vt[:2].T
    n_r = Feat_r.shape[0]
    PC_r, PC_s = PC[:n_r], PC[n_r:]

    fig = plt.figure(figsize=(6.2,5.6))
    plt.scatter(PC_r[:,0], PC_r[:,1], s=10, alpha=0.5, label="Real")
    plt.scatter(PC_s[:,0], PC_s[:,1], s=10, alpha=0.5, label="Sintético")
    plt.title("PCA de features (tiempo + frecuencia)")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.legend(); plt.grid(True, alpha=0.3)
    plot_and_save(fig, "PCA_features.png", outdir)

    # --- Métricas globales ---
    mmd_val = mmd_rbf(Feat_r, Feat_s, gamma=None)
    print(f"MMD-RBF (features tiempo+frecuencia): {mmd_val:.4f}")
    ks_std = ks_2samp(Feat_r[:,1], Feat_s[:,1]).statistic
    ks_band = ks_2samp(Ff_r[:,0], Ff_s[:,0]).statistic
    print(f"KS(std): {ks_std:.3f}   KS(relPow_0-5Hz): {ks_band:.3f}")
