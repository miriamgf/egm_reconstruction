import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.stats import skew, kurtosis, ks_2samp, entropy
import os

OUTDIR = "/home/profes/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/Figures/Dataset_augmentation_vae"

# ---------- utilidades ----------
def _ensure_3d(x):
    x = np.asarray(x)
    if x.ndim != 3:
        raise ValueError(f"Se esperaba (T,H,W), recibido {x.shape}")
    return x.astype(np.float32)

def _save_show(fig, outpath):
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath, bbox_inches="tight", dpi=150)
    plt.show()

def _hist_pair(a, b, bins=60, eps=1e-12):
    lo = float(min(np.nanmin(a), np.nanmin(b)))
    hi = float(max(np.nanmax(a), np.nanmax(b)))
    h1, edges = np.histogram(a, bins=bins, range=(lo, hi), density=True)
    h2, _     = np.histogram(b, bins=bins, range=(lo, hi), density=True)
    h1 = h1 + eps; h2 = h2 + eps
    h1 = h1 / h1.sum(); h2 = h2 / h2.sum()
    centers = 0.5*(edges[1:] + edges[:-1])
    return centers, h1, h2

def _plot_hist(values_real, values_synth, title, xlabel, outpath):
    centers, h1, h2 = _hist_pair(values_real, values_synth)
    fig = plt.figure(figsize=(7,4.5))
    plt.plot(centers, h1, label="Real")
    plt.plot(centers, h2, label="Sintético")
    plt.title(title)
    plt.xlabel(xlabel); plt.ylabel("Densidad")
    plt.grid(True, alpha=0.3); plt.legend()
    _save_show(fig, outpath)
    ks = ks_2samp(values_real.ravel(), values_synth.ravel()).statistic
    kl = float(entropy(h1, h2))
    print(f"{title} | KS={ks:.3f}  KL(real||synth)={kl:.3f}")

def _heatmap(mat, title, outpath, vmin=None, vmax=None):
    fig = plt.figure(figsize=(6.6,4.6))
    plt.imshow(mat, aspect='auto', origin='upper', vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.colorbar()
    plt.xlabel("W (32)"); plt.ylabel("H (12)")
    plt.grid(False)
    _save_show(fig, outpath)

# ---------- 1) features temporales por electrodo ----------
def electrode_time_features(X):
    """
    X: (T,H,W). Devuelve dict de mapas (H,W) con estadísticas a lo largo de T.
    """
    T, H, W = X.shape
    # (T,H,W) -> (H*W, T)
    Z = X.transpose(1,2,0).reshape(H*W, T)
    feats = {}
    feats["mean"] = Z.mean(axis=1).reshape(H,W)
    feats["std"]  = Z.std(axis=1).reshape(H,W)
    feats["rms"]  = np.sqrt((Z**2).mean(axis=1)).reshape(H,W)
    feats["p2p"]  = (Z.max(axis=1) - Z.min(axis=1)).reshape(H,W)
    feats["skew"] = skew(Z, axis=1, bias=False).reshape(H,W)
    feats["kurt"] = kurtosis(Z, axis=1, fisher=True, bias=False).reshape(H,W)
    return feats  # cada valor es (H,W)

# ---------- 2) potencias temporales por banda (por electrodo) ----------
def electrode_bandpowers(X, fs, bands=((0,5),(5,15),(15,40)), nperseg=None):
    """
    X: (T,H,W). Welch por electrodo -> potencias relativas por banda en (H,W,len(bands))
    """
    T, H, W = X.shape
    if nperseg is None:
        nperseg = min(256, T)
    out = np.zeros((H, W, len(bands)), dtype=np.float32)
    # Procesamos cada electrodo (H*W señales)
    for r in range(H):
        for c in range(W):
            f, Pxx = welch(X[:, r, c], fs=fs, nperseg=nperseg)
            bp = []
            for (fmin, fmax) in bands:
                idx = (f >= fmin) & (f < fmax)
                bp.append(np.trapz(Pxx[idx], f[idx]) if np.any(idx) else 0.0)
            bp = np.array(bp, dtype=np.float64)
            s = bp.sum() + 1e-12
            out[r, c, :] = (bp / s).astype(np.float32)
    names = [f"relPow_{int(a)}-{int(b)}Hz" for (a,b) in bands]
    return out, names  # (H,W,B)

# ---------- 3) espectro espacial (FFT2 de la imagen 12x32) ----------
def spatial_spectrum_profiles(X, n_samples=64, seed=123):
    """
    X: (T,H,W). Toma n_samples instantes aleatorios, FFT2(|·|) y promedia perfil radial.
    Devuelve (r, prof_mean).
    """
    rng = np.random.default_rng(seed)
    T, H, W = X.shape
    idx = rng.choice(T, size=min(n_samples, T), replace=False)
    acc = None; cnt = 0
    # Precomputo radios
    cy, cx = H//2, W//2
    yy, xx = np.ogrid[:H, :W]
    r = np.sqrt((yy - cy)**2 + (xx - cx)**2)
    r_int = r.astype(int)
    r_max = int(r_int.max())

    for t in idx:
        img = X[t]  # (H,W)
        F = np.fft.fftshift(np.fft.fft2(img)) / (H*W)
        mag = np.abs(F)
        power = np.bincount(r_int.ravel(), weights=mag.ravel(), minlength=r_max+1)
        counts = np.bincount(r_int.ravel(), minlength=r_max+1)
        counts[counts==0] = 1
        prof = power / counts
        if acc is None:
            acc = np.zeros_like(prof, dtype=np.float64)
        L = min(acc.size, prof.size)
        acc[:L] += prof[:L]
        cnt += 1
    prof_mean = acc / max(cnt,1)
    return np.arange(prof_mean.size), prof_mean

# ---------- 4) MMD ----------
def mmd_rbf(X, Y, gamma=None):
    X = np.asarray(X); Y = np.asarray(Y)
    if gamma is None:
        Z = np.vstack([X, Y])
        d2 = np.sum(Z**2,1,keepdims=True) - 2*Z@Z.T + np.sum(Z**2,1)[None,:]
        med = np.median(d2[d2>0])
        gamma = 1.0/(2*(med + 1e-12))
    def k(a,b):
        aa = np.sum(a**2,1,keepdims=True); bb = np.sum(b**2,1,keepdims=True)
        d2 = aa - 2*a@b.T + bb.T
        return np.exp(-gamma*d2)
    return float(k(X,X).mean() + k(Y,Y).mean() - 2*k(X,Y).mean())

# ---------- 5) pipeline principal ----------
def compare_real_vs_synth_gridtime(
    X_real, X_synth, fs,
    outdir=OUTDIR,
    bands=((0,5),(5,15),(15,40)),
    nperseg=None,
    n_spatial_samples=64
):
    Xr = _ensure_3d(X_real)  # (T,H,W)
    Xs = _ensure_3d(X_synth)

    T_r, H, W = Xr.shape
    T_s, Hs, Ws = Xs.shape
    assert (H==Hs and W==Ws), "Reales y sintéticos deben tener misma rejilla H×W"

    # --- A) features temporales por electrodo ---
    Fr = electrode_time_features(Xr)   # dict de (H,W)
    Fs = electrode_time_features(Xs)

    for name in ["mean","std","rms","p2p","skew","kurt"]:
        '''_heatmap(Fr[name], f"{name} (Real) a lo largo de T", os.path.join(outdir, f"map_{name}_real.png"))
        _heatmap(Fs[name], f"{name} (Synth) a lo largo de T", os.path.join(outdir, f"map_{name}_synth.png"))
        delta = Fr[name] - Fs[name]
        v = np.nanmax(np.abs(delta))
        _heatmap(delta, f"Δ {name} (Real - Synth)", os.path.join(outdir, f"map_delta_{name}.png"), vmin=-v, vmax=+v)'''

        # histogramas (aplanar H×W)
        _plot_hist(Fr[name].ravel(), Fs[name].ravel(),
                   title=f"Hist {name} por electrodo",
                   xlabel=name,
                   outpath=os.path.join(outdir, f"hist_{name}.png"))

    # --- B) potencias por banda temporales por electrodo ---
    Rr, band_names = electrode_bandpowers(Xr, fs, bands=bands, nperseg=nperseg)  # (H,W,B)
    Rs, _          = electrode_bandpowers(Xs, fs, bands=bands, nperseg=nperseg)

    for b in range(Rr.shape[-1]):
        name = band_names[b]
        '''_heatmap(Rr[:,:,b], f"{name} (Real)", os.path.join(outdir, f"map_{name}_real.png"))
        _heatmap(Rs[:,:,b], f"{name} (Synth)", os.path.join(outdir, f"map_{name}_synth.png"))
        delta = Rr[:,:,b] - Rs[:,:,b]
        v = np.nanmax(np.abs(delta))
        _heatmap(delta, f"Δ {name} (Real - Synth)", os.path.join(outdir, f"map_delta_{name}.png"), vmin=-v, vmax=+v)'''

        _plot_hist(Rr[:,:,b].ravel(), Rs[:,:,b].ravel(),
                   title=f"Hist {name} por electrodo",
                   xlabel=name,
                   outpath=os.path.join(outdir, f"hist_{name}.png"))

    # --- C) espectro espacial (imágenes 12x32) ---
    rR, pR = spatial_spectrum_profiles(Xr, n_samples=n_spatial_samples, seed=123)
    rS, pS = spatial_spectrum_profiles(Xs, n_samples=n_spatial_samples, seed=321)
    fig = plt.figure(figsize=(7,4.5))
    plt.plot(rR, pR, label="Real")
    plt.plot(rS, pS, label="Sintético")
    plt.title("Perfil radial del espectro espacial (promedio)")
    plt.xlabel("Radio (frecuencia espacial)"); plt.ylabel("|FFT2| promedio")
    plt.grid(True, alpha=0.3); plt.legend()
    _save_show(fig, os.path.join(outdir, "spatial_fft2_radial_profile.png"))

    # --- D) MMD + KS (globales en features por electrodo) ---
    # Construyo vectores de features por electrodo concatenando (mean,std,rms,p2p,skew,kurt, bandas…)
    Fstack_r = np.stack([Fr["mean"], Fr["std"], Fr["rms"], Fr["p2p"], Fr["skew"], Fr["kurt"]], axis=-1)  # (H,W,6)
    Fstack_s = np.stack([Fs["mean"], Fs["std"], Fs["rms"], Fs["p2p"], Fs["skew"], Fs["kurt"]], axis=-1)

    Fband_r  = Rr  # (H,W,B)
    Fband_s  = Rs

    V_r = np.concatenate([Fstack_r.reshape(-1, Fstack_r.shape[-1]),
                          Fband_r.reshape(-1, Fband_r.shape[-1])], axis=1)  # (H*W, 6+B)
    V_s = np.concatenate([Fstack_s.reshape(-1, Fstack_s.shape[-1]),
                          Fband_s.reshape(-1, Fband_s.shape[-1])], axis=1)

    mmd_val = mmd_rbf(V_r, V_s, gamma=None)
    ks_std  = ks_2samp(Fr["std"].ravel(), Fs["std"].ravel()).statistic
    ks_low  = ks_2samp(Rr[:,:,0].ravel(), Rs[:,:,0].ravel()).statistic  # primera banda

    print(f"[Resumen] MMD-RBF (features por electrodo): {mmd_val:.4f}")
    print(f"[Resumen] KS(std)={ks_std:.3f}  KS({band_names[0]})={ks_low:.3f}")

    # Guardar métricas
    try:
        import csv
        csv_path = os.path.join(outdir, "distribution_metrics_gridtime.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["metric","value"])
            w.writerow(["MMD_RBF_electrode_features", f"{mmd_val:.6f}"])
            w.writerow(["KS_std", f"{ks_std:.6f}"])
            w.writerow([f"KS_{band_names[0]}", f"{ks_low:.6f}"])
        print(f"Métricas guardadas en: {csv_path}")
    except Exception as e:
        print("No se pudieron guardar métricas CSV:", e)
