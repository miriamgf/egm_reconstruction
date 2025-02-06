import sys
sys.path.append("../Code")
import argparse
import math
import os
import sys
import time
import matplotlib.pyplot as plt

import scripts.Tikhonov.forward_inverse_problem as fip
import numpy as np
import scripts.Tikhonov.precompute_matrix as pre_m
from numpy import reshape
import tools_.tools as tools
from scripts.evaluation.tools_evaluate import downsampling

path_output_l_curva= f"/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/ZOT_L_curva/"

class TikhonovReconstruction:

    def __init__(self,bsp,transfer_matrix, order, path_figs=path_output_l_curva):

        self.transfer_matrix=transfer_matrix
        self.bsp=bsp
        self.order=order
        self.path_figs=path_figs

    def __call__(self, plot_L_curve=False):
        
        A = np.array(self.transfer_matrix)  # Only one torso
        AA, L, LL = pre_m.precompute_matrix(A, self.order)
        print('Computing Tikhonov')
        lambda_test = np.logspace(-6, 2, 200)  # Ampliar el rango de lambda

        x_hat, lambda_opt, magnitude_term, error_term, maxcurve_index = (
        fip.classical_tikhonov_noiter_global(A, AA, L, LL, self.bsp, lambda_test=lambda_test))
        print('opt lamda: ', lambda_opt)
        if plot_L_curve:
            self.plot_L_curve(magnitude_term, error_term, maxcurve_index )
        return x_hat
    
    def tik_post_process_to_plot(self,tik_rec, fs, divisible_rows, n_batch):
        '''
        This function postprocess ZOT reconstructions: downsamples and changes batch size according
        to experiment_ID to enable a fair evaluation and visualization

        
        '''
        tik_rec_mod = tik_rec[:, :-1]
        tik_rec_mod_T = tik_rec_mod.T
        tik_rec_down =downsampling(tik_rec_mod_T, fs)
        tik_trunc=tik_rec_down[:divisible_rows]
        tik_batches = reshape(
                        tik_trunc,
                        (
                            int(len(tik_trunc) / n_batch),
                            n_batch,
                            tik_trunc.shape[1],
                            1,
                        ),
                    )
        return tik_batches
    
    def plot_L_curve(self,magnitude_term, error_term, maxcurve_index ):
 

        # Supongamos que ya tienes los datos de `error_term` y `magnitude_term`:
        # error_term: normas de los residuos (||Ax - y||_2^2)
        # magnitude_term: normas de la solución regularizada (||Lx||_2^2)
        # lambda_test: valores de lambda probados

        # Convertir a escala logarítmica
        x_term = np.log10(error_term)
        z_term = np.log10(magnitude_term)

        # Pintar la L-curve
        plt.figure(figsize=(8, 6))
        plt.plot(x_term, z_term, '-o', markersize=4, label='L-Curve')
        plt.xlabel(r'$\log_{10}(\|Ax - y\|_2^2)$ (Norma de residuos)')
        plt.ylabel(r'$\log_{10}(\|Lx\|_2^2)$ (Norma de la solución)')
        plt.title('L-Curve para Tikhonov Regularization')
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.legend()

        # Si se seleccionó lambda_opt, marcarlo en la curva
        if 'maxcurve_index' in locals():
            plt.scatter(x_term[maxcurve_index], z_term[maxcurve_index], color='red', label=r'$\lambda_{\text{opt}}$')
            plt.legend()

        plt.savefig(self.path_figs + 'L_curve.png')
        print('L-curve plot saved in: ', self.path_figs + 'L_curve.png')
        plt.close()

