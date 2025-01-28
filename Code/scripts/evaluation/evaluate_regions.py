import sys
sys.path.append("../Code")
import os
from scipy.io import loadmat
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import numpy as np
import ast

class EvaluateRegions:
    def __init__(self, experiment_ID_list):
        # Ruta del archivo .mat
        self.path_regions = "/home/pdi/miriamgf/tesis/Autoencoders/Regions/extended_regions.mat"
        self.regions = loadmat(self.path_regions)['regions'][0].astype(int)  # Convertir a enteros
        self.experiment_ID_list = experiment_ID_list
        self.path_output= "/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/experiments/experiments_VAE/"

    def evaluate_metric_per_region(self):
        '''

        This function loads metrics from specified algorithm and outputs a dataframe
        with the mean value of the metrics according to 7 anatomical regions

        '''
        # Obtener las clases únicas de regiones
        range_region_classes = list(np.unique(self.regions))

        # Diccionario para almacenar resultados globales
        aggregated_results = []

        # Iterar sobre los experimentos
        for experiment_ID in self.experiment_ID_list:
            path = f"{self.path_output}{experiment_ID}/metrics_all_nodes_dl.csv"
            print('Loading CSV of:', path)
            csv_metrics = pd.read_csv(path)
            column_names = list(csv_metrics.columns)  # Obtener los nombres de las columnas

            # Iterar por cada métrica
            for metric in column_names:
                # Saltar la columna 'name'
                if metric == 'name':
                    continue
                
                # Inicializar un diccionario por métrica y región
                metric_results = {region_class: [] for region_class in range_region_classes if region_class != 0}

                # Iterar por cada paciente
                for patient in range(len(csv_metrics)):
                    # Obtener la fila del paciente
                    patient_row = csv_metrics.iloc[patient]

                    # Obtener la métrica para el paciente
                    column_metric = patient_row[metric]

                    # Convertir a array NumPy si es un string
                    if isinstance(column_metric, str):
                        try:
                            metric_array = np.array(ast.literal_eval(column_metric))
                        except Exception as e:
                            print(f"Error al procesar la columna {metric}: {e}")
                            continue
                    else:
                        metric_array = np.array(column_metric)

                    # Calcular métricas por `region_class`
                    for region_class in range_region_classes:
                        if region_class == 0:  # Excluir la clase 0 (extendidos)
                            continue
                        
                        # Obtener los índices de la región
                        indices_class = np.where(self.regions == region_class)[0]
                        
                        # Extraer métricas para la región
                        metric_region_i = metric_array[indices_class]
                        metric_results[region_class].extend(metric_region_i)

                # Calcular la media y std global por métrica y región
                for region_class, values in metric_results.items():
                    if len(values) > 0:
                        global_mean = np.mean(values)
                        global_std = np.std(values)
                        aggregated_results.append({
                            'Metric': metric,
                            'Region Class': region_class,
                            'Mean': global_mean,
                            'STD': global_std
                        })

        # Crear un DataFrame para guardar los resultados
        results_df = pd.DataFrame(aggregated_results)
        results_df.to_csv(self.path_output+'global_metrics_per_region.csv', index=False)
        #dic_patient.to_csv(self.path_output+"metrics_per_regions.csv", index=False)
        print('Saving in: ', self.path_output+'global_metrics_per_region.csv')
    
                    

    def __call__(self, *args, **kwds):

        self.evaluate_metric_per_region()

if __name__ == "__main__":
    experiment_ID_list=["OMAMI_repeated", "OMAMI_VAE_Optuna_1"]
    EvaluateRegionsObj=EvaluateRegions(experiment_ID_list=experiment_ID_list)()



    
