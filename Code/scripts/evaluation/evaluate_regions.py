import sys
sys.path.append("../Code")
import os
from scipy.io import loadmat
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import seaborn as sns

class EvaluateRegions:
    def __init__(self, experiment_ID_list, test_id='0'):
        # Ruta del archivo .mat
        self.path_regions = "/home/pdi/miriamgf/tesis/Autoencoders/Regions/extended_regions.mat"
        self.regions = loadmat(self.path_regions)['regions'][0].astype(int)  # Convertir a enteros
        self.experiment_ID_list = experiment_ID_list
        self.path_output= "/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/experiments/experiments_VAE/"
        self.path_output_figs = "/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/evaluation/Regions/region_boxplot"
        self.test_id=test_id

    def evaluate_metric_per_region(self, algorithm, filename, tik):
        '''

        This function loads metrics from specified algorithm and outputs a dataframe
        with the mean value and full vector of the metrics according to 7 anatomical regions

        '''
        filename=filename[0]
        # Obtener las clases únicas de regiones
        range_region_classes = list(np.unique(self.regions))

        aggregated_results = []

        for experiment_ID in algorithm: #For each algorithm, a dataframe with aggregated_results will be saved 
            
            path = f"{self.path_output}{experiment_ID}/{filename}"
            print('Loading CSV of:', path)
            csv_metrics = pd.read_csv(path)
            column_names = list(csv_metrics.columns)  

            # Iterar por cada métrica
            for metric in column_names:

                if metric == 'name' or metric == "Peak_detector_Error":
                    continue
                
                # Inicializar un diccionario por métrica y región
                metric_results = {region_class: [] for region_class in range_region_classes if region_class != 0}

                # Iterar por cada paciente
                for patient in range(len(csv_metrics)):
                    print('Patient:', patient)
                    # Obtener la fila del paciente
                    patient_row = csv_metrics.iloc[patient]

                    # Obtener la métrica para el paciente
                    column_metric = patient_row[metric]

                    # Convertir a array NumPy si es un string
                    if isinstance(column_metric, str):
                        try:
                            column_metric = column_metric.replace('\n', '')
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
                            'STD': global_std,
                            'Array': np.array(values)
                        })

            # Crear un DataFrame para guardar los resultados
            results_df = pd.DataFrame(aggregated_results)
            if tik:
                results_df.to_csv(f"{self.path_output}global_metrics_per_region_{algorithm[0]}_tik_testing.csv", index=False)
                print('Saving in: ', f"{self.path_output}global_metrics_per_region_{algorithm[0]}_tik_testing.csv")
            else:
                results_df.to_csv(f"{self.path_output}global_metrics_per_region_{algorithm[0]}_dl_testing.csv", index=False)
                print('Saving in: ', f"{self.path_output}global_metrics_per_region_{algorithm[0]}_dl_testing.csv")


        #dic_patient.to_csv(self.path_output+"metrics_per_regions.csv", index=False)
        
        return results_df
    
    def plot_boxplot_per_algorithm(self, df, algorithm, tik ):
        '''
        This function takes as input the computed metrics by region in self.evaluate_metric_per_region
        and plots barplots and boxplots representing the distribution of the metrics as the average for all patients for 1 algorithm
        
        '''
        algorithm=algorithm[0]

        # Configurar estilo de gráficos
        sns.set(style="whitegrid")

        # Obtener todas las métricas únicas
        metricas = df["Metric"].unique()

        # Crear un barplot para cada métrica
        for metric in metricas:

            region_colors = {
                1: "red",
                2: "green",
                3: "blue",
                4: "yellow",
                5: "cyan",
                6: "deeppink",
                7: "gray"
            }

            # Filtrar por métrica
            df_subset = df[df["Metric"] == metric]
            plt.figure(figsize=(10, 6))
            sns.barplot(
                data=df_subset, 
                x="Region Class", 
                y="Mean", 
                palette=[region_colors[region] for region in sorted(df_subset['Region Class'].unique())]
            )

            plt.title(f"Media de {metric} por Region Class")
            plt.xlabel("Region Class")
            plt.ylabel("Mean")
            if tik:
                path = f"{self.path_output_figs}/{algorithm}/barplot_regions_{metric}_tik_2.png"
            else:
                path = f"{self.path_output_figs}/{algorithm}/barplot_regions_{metric}_dl_2.png"

            os.makedirs(os.path.dirname(path), exist_ok=True)
            plt.savefig(path)
            print('Barplot of regions saved at:', path)
            plt.close()

            #############BOXPLOT############

            # Explode para descomponer listas en filas individuales
            df_exploded = df_subset.explode('Array')

            plt.figure(figsize=(10, 6))
            sns.boxplot(
                data=df_exploded, 
                x='Region Class', 
                y='Array', 
                palette=[region_colors[region] for region in sorted(df_exploded['Region Class'].unique())]
            )
            plt.xlabel("Región")
            plt.ylabel("Valores")
            plt.title("Boxplot representing distribution of metrics in each region for all patients")
            plt.xticks(rotation=45)  
            if tik:
                path = f"{self.path_output_figs}/{algorithm}/boxplot_regions_{metric}_tik_2.png"
            else:
                path = f"{self.path_output_figs}/{algorithm}/boxplot_regions_{metric}_dl_2.png"
            os.makedirs(os.path.dirname(path), exist_ok=True)
            plt.savefig(path)
            print('Barplot of regions saved at:', path)
            plt.close()

    def __call__(self, *args, **kwds):

        list_filenames=[[f"metrics_all_nodes_dl.csv"],
                        [f"metrics_all_nodes_tik.csv"]]
        
        for algorithm in self.experiment_ID_list:
            tik=False
            print('algorithm ', algorithm)

            for filename in list_filenames:
             
                if filename== ["metrics_all_nodes_tik.csv"]:
                    tik=True
                
                print('To save regions of', filename)

                results_df=self.evaluate_metric_per_region(algorithm, filename, tik)
                self.plot_boxplot_per_algorithm(results_df, algorithm, tik)

if __name__ == "__main__":


    experiment_ID_list=[["OMAMI_repeated"], ["OMAMI_VAE_Optuna_1"], ['OMAMI_no_filt'], ['OMAMI_VAE_no_filt']]
    EvaluateRegionsObj=EvaluateRegions(experiment_ID_list=experiment_ID_list)()



    
