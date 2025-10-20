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

def _js_divergence(p, q):
    """Jensen–Shannon divergence (simétrica y acotada)."""
    m = 0.5*(p+q)
    return 0.5*entropy(p, m) + 0.5*entropy(q, m)

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
    js = float(_js_divergence(h1, h2))
    print(f"{title} | KS={ks:.3f}  KL(real||synth)={kl:.3f}  JS={js:.3f}")
    return ks, kl, js

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
    Z = X.transpose(1,2,0).reshape(H*W, T)  # (H*W, T)
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

# ---------- 4) helpers para MMD robusto ----------
def _standardize(Xr, Xs, eps=1e-8):
    """Estandariza usando estadísticas del conjunto REAL."""
    mu = Xr.mean(axis=0, keepdims=True)
    sd = Xr.std(axis=0, keepdims=True) + eps
    return (Xr - mu)/sd, (Xs - mu)/sd

def _match_sizes(X, Y, seed=0):
    """Submuestra sin reemplazo para igualar tamaños."""
    rng = np.random.default_rng(seed)
    n = min(len(X), len(Y))
    Xi = rng.choice(len(X), size=n, replace=False)
    Yi = rng.choice(len(Y), size=n, replace=False)
    return X[Xi], Y[Yi]

def mmd_rbf_unbiased(X, Y, gammas=None):
    """
    MMD^2 no sesgado (U-statistic) con suma de kernels RBF.
    X, Y: (n,d) con n igualado previamente.
    """
    if gammas is None:
        Z = np.vstack([X, Y])
        d2 = np.sum(Z**2,1,keepdims=True) - 2*Z@Z.T + np.sum(Z**2,1)[None,:]
        sigma2_med = np.median(d2[d2>0])
        # multi-kernel alrededor de la mediana
        sigmas2 = [sigma2_med/2, sigma2_med, 2*sigma2_med] if sigma2_med > 0 else [1.0]
        gammas = [1.0/(2*s) for s in sigmas2]

    def ksum(A, B):
        AA = np.sum(A**2,1,keepdims=True)
        BB = np.sum(B**2,1,keepdims=True)
        d2 = AA - 2*A@B.T + BB.T
        K = sum(np.exp(-g*d2) for g in gammas)
        return K

    n = X.shape[0]
    Kxx = ksum(X, X); np.fill_diagonal(Kxx, 0.0)
    Kyy = ksum(Y, Y); np.fill_diagonal(Kyy, 0.0)
    Kxy = ksum(X, Y)
    mmd2 = Kxx.sum()/(n*(n-1)) + Kyy.sum()/(n*(n-1)) - 2*Kxy.mean()
    return float(mmd2)

def mmd_bootstrap(Xr, Xs, n_boot=200, seed=123):
    """
    Repite emparejado+estandarización+MMD varias veces para media e IC95%.
    Devuelve (mean, low95, high95, std).
    """
    rng = np.random.default_rng(seed)
    vals = []
    for b in range(n_boot):
        # cambiamos semilla de submuestreo en cada bootstrap
        Xr_m, Xs_m = _match_sizes(Xr, Xs, seed=int(rng.integers(0, 1_000_000)))
        Xr_z, Xs_z = _standardize(Xr_m, Xs_m)
        vals.append(mmd_rbf_unbiased(Xr_z, Xs_z))
    vals = np.array(vals, dtype=float)
    mean = float(np.mean(vals))
    std  = float(np.std(vals, ddof=1))
    low  = float(np.percentile(vals, 2.5))
    high = float(np.percentile(vals, 97.5))
    return mean, low, high, std

# ---------- 5) pipeline principal ----------
def compare_real_vs_synth_gridtime(
    X_real, X_synth, fs,
    outdir=OUTDIR,
    bands=((0,5),(5,15),(15,40)),
    nperseg=None,
    n_spatial_samples=64,
    mmd_n_boot=0,          # >=1 para activar bootstrap
    bootstrap_seed=123
):
    Xr = _ensure_3d(X_real)  # (T,H,W)
    Xs = _ensure_3d(X_synth)

    T_r, H, W = Xr.shape
    T_s, Hs, Ws = Xs.shape
    assert (H==Hs and W==Ws), "Reales y sintéticos deben tener misma rejilla H×W"

    # --- A) features temporales por electrodo ---
    Fr = electrode_time_features(Xr)   # dict de (H,W)
    Fs = electrode_time_features(Xs)

    # histogramas (aplanar H×W) + métricas
    hist_metrics = []
    for name in ["mean","std","rms","p2p","skew","kurt"]:
        ks, kl, js = _plot_hist(
            Fr[name].ravel(), Fs[name].ravel(),
            title=f"Hist {name} por electrodo",
            xlabel=name,
            outpath=os.path.join(outdir, f"hist_{name}.png")
        )
        hist_metrics.append((f"KS_{name}", ks))
        hist_metrics.append((f"KL_{name}", kl))
        hist_metrics.append((f"JS_{name}", js))

    # --- B) potencias por banda temporales por electrodo ---
    Rr, band_names = electrode_bandpowers(Xr, fs, bands=bands, nperseg=nperseg)  # (H,W,B)
    Rs, _          = electrode_bandpowers(Xs, fs, bands=bands, nperseg=nperseg)

    for b in range(Rr.shape[-1]):
        name = band_names[b]
        ks, kl, js = _plot_hist(
            Rr[:,:,b].ravel(), Rs[:,:,b].ravel(),
            title=f"Hist {name} por electrodo",
            xlabel=name,
            outpath=os.path.join(outdir, f"hist_{name}.png")
        )
        hist_metrics.append((f"KS_{name}", ks))
        hist_metrics.append((f"KL_{name}", kl))
        hist_metrics.append((f"JS_{name}", js))

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

    # Emparejado + estandarización + MMD no sesgado con multi-kernel
    V_r_m, V_s_m = _match_sizes(V_r, V_s, seed=bootstrap_seed)
    V_r_z, V_s_z = _standardize(V_r_m, V_s_m)
    mmd_val = mmd_rbf_unbiased(V_r_z, V_s_z)

    # Bootstrap opcional
    mmd_mean = mmd_low = mmd_high = mmd_std = None
    if mmd_n_boot and mmd_n_boot > 0:
        mmd_mean, mmd_low, mmd_high, mmd_std = mmd_bootstrap(V_r, V_s, n_boot=mmd_n_boot, seed=bootstrap_seed)

    ks_std  = ks_2samp(Fr["std"].ravel(), Fs["std"].ravel()).statistic
    ks_low  = ks_2samp(Rr[:,:,0].ravel(), Rs[:,:,0].ravel()).statistic  # primera banda

    print(f"[Resumen] MMD-RBF (unbiased, multi-kernel): {mmd_val:.4f}")
    if mmd_mean is not None:
        print(f"[Bootstrap] MMD mean={mmd_mean:.4f}  95%CI=({mmd_low:.4f},{mmd_high:.4f})  std={mmd_std:.4f}")
    print(f"[Resumen] KS(std)={ks_std:.3f}  KS({band_names[0]})={ks_low:.3f}")

    # Guardar métricas
    try:
        import csv
        csv_path = os.path.join(outdir, "distribution_metrics_gridtime.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["metric","value"])
            w.writerow(["MMD_RBF_electrode_features_unbiased", f"{mmd_val:.6f}"])
            if mmd_mean is not None:
                w.writerow(["MMD_RBF_bootstrap_mean", f"{mmd_mean:.6f}"])
                w.writerow(["MMD_RBF_bootstrap_ci95_low", f"{mmd_low:.6f}"])
                w.writerow(["MMD_RBF_bootstrap_ci95_high", f"{mmd_high:.6f}"])
                w.writerow(["MMD_RBF_bootstrap_std", f"{mmd_std:.6f}"])
            w.writerow(["KS_std", f"{ks_std:.6f}"])
            w.writerow([f"KS_{band_names[0]}", f"{ks_low:.6f}"])
            # añade también métricas de histogramas resumidas
            for key, val in hist_metrics:
                w.writerow([key, f"{val:.6f}"])
        print(f"Métricas guardadas en: {csv_path}")
    except Exception as e:
        print("No se pudieron guardar métricas CSV:", e)

    # Devuelve un diccionario útil por si quieres usarlo programáticamente
    out = {
        "MMD_unbiased": mmd_val,
        "MMD_bootstrap": {
            "mean": mmd_mean, "ci95_low": mmd_low, "ci95_high": mmd_high, "std": mmd_std
        },
        "KS_std": ks_std,
        f"KS_{band_names[0]}": ks_low,
        "hist_metrics": dict(hist_metrics)
    }
    return out
