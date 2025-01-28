import numpy as np
import scipy.io as sio
import vtk
from scipy.interpolate import interp1d
from vtk.util.numpy_support import numpy_to_vtk

import os
import cv2
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../..", "Code")))
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import pandas as pd


class ReportMetrics():
    def __init__(self,  algorithm_list=None):
        self.path_experiments= "/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/experiments/experiments_VAE/"

        if algorithm_list is None:
            self.algorithm_list= ['OMAMI_repeated', 'OMAMI_VAE_Optuna_1', 'OMAMI_no_filt', 'OMAMI_VAE_no_filt' ]
        else:
            self.algorithm_list= algorithm_list
        self.algorithm_ID_tik_filt="OMAMI_VAE_Optuna_1"
        self.algorithm_ID_tik_no_filt="OMAMI_no_filt"

    def load_evaluation_dataframes(self, algorithm_ID):

        csv_path_dl=f"{self.path_experiments}{algorithm_ID}/metrics_dl.csv"
        csv_path_tik=f"{self.path_experiments}{algorithm_ID}/metrics_tik.csv"
        df_dl = pd.read_csv(csv_path_dl)
        df_tik = pd.read_csv(csv_path_tik)

        #Preprocess column names
        df_dl.columns = df_dl.columns.str.replace('_', '', regex=True)  # Elimina los guiones bajos
        df_tik_filt.columns = df_tik_filt.columns.str.replace('_', '', regex=True)  # Elimina los guiones bajos


        csv_dl=f"/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/experiments/experiments_VAE/{algorithm_ID}/metrics_dl.csv"
        df_dl = pd.read_csv(csv_dl)
        df_dl = df_dl.sort_values(by='Correlation', ascending=False)
        df_dl.columns = df_dl.columns.str.replace('_', '', regex=True)  # Elimina los guiones bajos


        csv_tik_filt=f"/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/experiments/experiments_VAE/{self.algorithm_ID_tik_filt}/metrics_tik.csv"
        df_tik_filt = pd.read_csv(csv_tik_filt)
        df_tik_filt.columns = df_tik_filt.columns.str.replace('_', '', regex=True)  # Elimina los guiones bajos



        csv_tik_no_filt=f"/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/experiments/experiments_VAE/{self.algorithm_ID_tik_no_filt}/metrics_tik.csv"
        df_tik_no_filt = pd.read_csv(csv_tik_no_filt)
        df_tik_no_filt.columns = df_tik_no_filt.columns.str.replace('_', '', regex=True)  # Elimina los guiones bajos

        return


        
    def __call__(self, *args, **kwds):

        algorithm_mod_name = [algo.replace('_', '') for algo in self.algorithm_list]
        cont=0
        for algorithm_ID in self.algorithm_list:
            algorithm_ID_mod=algorithm_mod_name[cont]
            cont+=1
            # Cargar el archivo CSV
            self.load_evaluation_dataframes(algorithm_ID_mod)
            csv_dl = f"/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/experiments/experiments_VAE/{algorithm_ID}/metrics_dl.csv"
            df = pd.read_csv(csv_dl)
            df.columns = df.columns.str.replace('_', '', regex=True)  # Elimina los guiones bajos
            

            # Agregar sufijo al DataFrame actual
            df = df.add_suffix(f"_{algorithm_ID_mod}")
            df.rename(columns={f'name_{algorithm_ID_mod}': 'name'}, inplace=True)

            # Si `merged_df` no está inicializado, usar el primer DataFrame como base
            if merged_df is None:
                merged_df = df
            else:
                # Fusionar con el DataFrame base
                merged_df = pd.merge(merged_df, df, on='name', how='outer')

        merged_df_dl=merged_df.copy()


if __name__ == "__main__":
    ReportMetrics()()