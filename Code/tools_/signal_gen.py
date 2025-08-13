import numpy as np
from tqdm import tqdm
import os

class SyntheticDataGenerator:
    def __init__(self, vae_model, latent_dim, save_dir, params):
        """
        Parameters:
        - vae_model: tu instancia del VAE ya cargado (con weights).
        - latent_dim: dimensión del espacio latente.
        - experiment_dir: carpeta donde se guardarán las señales generadas.
        """
        self.vae = vae_model
        self.latent_dim = latent_dim
        self.save_dir = save_dir
        self.model_name = params['experiment_name']
        self.params = params

    def sample_latent_vectors(self, num_samples, mu, sigma, z_mean_train, sampling="random"):
        """Genera muestras latentes desde una distribución normal guiada por mu y sigma."""
        if sampling=="random":
            mu=0
            sigma=1
            return np.random.normal(loc=mu, scale=sigma, size=(num_samples, self.latent_dim))
        elif sampling=="guided":
            return np.random.normal(loc=mu, scale=sigma, size=(num_samples, self.latent_dim))
        elif sampling=="interpolated":

            idx1, idx2 = np.random.choice(z_mean_train.shape[0], size=2, replace=False)
            z1 = z_mean_train[idx1]
            z2 = z_mean_train[idx2]

            # Genera interpolaciones lineales entre z1 y z2
            alphas = np.linspace(0, 1, num_samples)
            return np.array([(1 - alpha) * z1 + alpha * z2 for alpha in alphas])
        
    def decode_latent_vectors(self, z_samples):
        """Decodifica un conjunto de vectores latentes a señales sintéticas."""
        synthetic_signals = self.vae.decode_from_latent(z_samples)
        return np.array(synthetic_signals).squeeze()

    def compute_rmse(self, synthetic_signals, real_signals):
        """Calcula el RMSE de cada señal sintética respecto al conjunto de señales reales."""
        rmse_list = []
        for idx in tqdm(range(synthetic_signals.shape[0])):
            syn_signal = synthetic_signals[idx]  # Shape: (400, 2048)
            
            # Broadcasting to subtract from all real signals at once
            diff = real_signals - syn_signal  # Shape: (333, 400, 2048)
            
            # Compute MSE per real sample (axis=(1,2): time & nodes)
            mse = np.mean(diff**2, axis=(1,2))  # Shape: (333,)
            
            # RMSE
            rmse = np.sqrt(mse)  # Shape: (333,)
            
            # Take the minimal RMSE (the real signal most similar to this synthetic one)
            min_rmse = np.min(rmse)
            
            rmse_list.append((idx, min_rmse))
        return rmse_list

    def select_best_signals(self, synthetic_signals, rmse_list, num_to_keep=100):
        """Selecciona las señales sintéticas con menor RMSE."""
        rmse_list_sorted = sorted(rmse_list, key=lambda x: x[1])
        best_idxs = [idx for idx, _ in rmse_list_sorted[:num_to_keep]]
        return synthetic_signals[best_idxs]

    def save_signals(self, signals, filename):
        """Guarda las señales sintéticas en archivo .npy."""
        save_path = os.path.join(self.save_dir, filename)
        np.save(save_path, signals)
        print(f"Saved {signals.shape[0]} synthetic signals at {save_path}")

    def generate_and_select(self, real_signals, z_mean_train, num_generated=10, num_selected=5):
        """
        Pipeline completo: genera señales, selecciona las mejores y las guarda.
        """
        mu = np.mean(z_mean_train, axis=0)
        sigma = np.std(z_mean_train, axis=0)

        z_samples = self.sample_latent_vectors(num_generated, mu, sigma, z_mean_train, sampling='random')
        synthetic_signals = self.decode_latent_vectors(z_samples)

        rmse_list = self.compute_rmse(synthetic_signals, real_signals)
        best_synthetic_signals = self.select_best_signals(synthetic_signals, rmse_list, num_selected)
        self.save_signals(best_synthetic_signals, "synt_" + self.model_name + ".npy")

        return best_synthetic_signals
