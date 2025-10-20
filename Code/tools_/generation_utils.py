from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

def plot_pca(z_mean_train,batch_classes, class_labels, experiment_dir):

    pca = PCA(n_components=2)
    z_proj = pca.fit_transform(z_mean_train)  

    plt.figure(figsize=(6, 6))
    for class_id in np.unique(batch_classes):
        mask = batch_classes == class_id
        plt.scatter(
            z_proj[mask, 0], z_proj[mask, 1],
            label=class_labels.get(class_id, f"Clase {class_id}"),
            alpha=0.7
        )

    plt.title("z_mean (PCA)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(experiment_dir + 'pca_guided_by_class_named.png')
    print(experiment_dir + 'pca_guided_by_class_named.png')
    plt.close()

def plot_tsne(z_mean_train,batch_classes, class_labels, experiment_dir):

    #tsne
    # --- Datos/etiquetas ---
    X = z_mean_train                      # (n_samples, latent_dim)
    y = np.asarray(batch_classes).ravel() # (n_samples,)

    # --- Estandariza (recomendado para t-SNE) ---
    Xs = StandardScaler().fit_transform(X)

    # --- Perplexity válida según tu n de muestras ---
    n = Xs.shape[0]
    perplexity = min(30, max(5, (n - 1) // 3))

    # --- t-SNE ---
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=0
    )
    Z = tsne.fit_transform(Xs)  # (n_samples, 2)

    # --- Plot ---
    plt.figure(figsize=(9, 6), tight_layout=True)
    markers = ['o', 's', '^', 'D', 'P', 'X']
    for i, cls in enumerate(np.unique(y)):
        m = (y == cls)
        label = class_labels.get(cls, f"Clase {cls}")
        plt.scatter(Z[m, 0], Z[m, 1], alpha=0.8, s=40, marker=markers[i % len(markers)], label=label)

    plt.title("z_mean (t-SNE)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.grid(True, linewidth=0.5)
    plt.legend(frameon=True)

    # --- Guardar junto al PCA ---
    out_path = experiment_dir + "tsne_guided_by_class_named.png"
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(out_path)