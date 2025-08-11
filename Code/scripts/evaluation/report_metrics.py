import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../..", "Code")))
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as mpatches  # Para arreglar la leyenda
import seaborn as sns
pd.set_option('display.max_columns', None)


class ReportMetrics():
    '''
    This class provides a global reporting on a set of <algorithm_list>

    It computes global metrics algorithm wise and patient wise

    All algorithms to be analyzed must have been evaluated first (evaluate_main.py)
    
    
    
    '''
    def __init__(self,  algorithm_list=None, test_id=0, name='default', stratified=False):
        self.path_experiments= "/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/experiments/experiments_VAE/"

        if algorithm_list is None:
            self.algorithm_list= ['OMAMI_repeated', 'OMAMI_VAE_Optuna_1', 'OMAMI_no_filt', 'OMAMI_VAE_no_filt' ]
        else:
            self.algorithm_list= algorithm_list
        #self.algorithm_ID_tik_filt="OMAMI_VAE_Optuna_1"
        self.algorithm_ID_tik_no_filt="OMAMI_no_filt_Optuna_bs_fs_Optuna"
        self.name_results = "".join(self.algorithm_list)
        self.path_to_save=f"/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/evaluation/Global_results/agg_results_figs/{self.name_results}/"
        self.path_to_save_summary= f"/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/evaluation/Global_results/agg_results_figs/{self.name_results}/"
        os.makedirs(self.path_to_save, exist_ok=True)
        os.makedirs(self.path_to_save_summary, exist_ok=True)
        self.name=name
        self.test_id=test_id
        self.stratified=stratified
        if self.stratified:
            self.class_df = "/home/profes/miriamgf/tesis/Autoencoders/Data/annotations.csv"
    
    def load_class_df(self):
        '''
        This function loads the class df for stratified evaluation
        '''
        # Load
        df_annotation_complexity= pd.read_csv(self.class_df, delimiter=';')
        df_annotation_complexity.rename(columns={'Simulation_Name': 'name'}, inplace=True)

        # Select only patients with complexity in classes_to_strat
        df_annotation_complexity.head()
        return df_annotation_complexity
    
    def load_evaluation_dataframes(self, algorithm_ID):

        '''
        This function loads the selected DL model and all the ZOT variations*
        *Zot variations at January 2025:
            - Egm filtered
            - Egm not filtered
        
        '''

        try:
            csv_dl=f"{self.path_experiments}{algorithm_ID}/metrics_dl_{self.test_id}.csv"
            df_dl = pd.read_csv(csv_dl)

        except:
            csv_dl=f"{self.path_experiments}{algorithm_ID}/metrics_dl__str_test.csv"
            df_dl = pd.read_csv(csv_dl)

        df_dl = df_dl.sort_values(by='Correlation', ascending=False)
        df_dl.columns = df_dl.columns.str.replace('_', '', regex=True)  # Elimina los guiones bajos
        df_dl_or = df_dl.copy()

        '''
        csv_tik_filt=f"{self.path_experiments}{self.algorithm_ID_tik_filt}/metrics_tik_{self.test_id}.csv"
        df_tik_filt = pd.read_csv(csv_tik_filt)
        df_tik_filt.columns = df_tik_filt.columns.str.replace('_', '', regex=True)  # Elimina los guiones bajos
        '''
        try:
            csv_tik_no_filt=f"{self.path_experiments}{self.algorithm_ID_tik_no_filt}/metrics_tik.csv"
            df_tik_no_filt = pd.read_csv(csv_tik_no_filt)

        except:
            csv_tik_no_filt=f"{self.path_experiments}{self.algorithm_ID_tik_no_filt}/metrics_tik_0.csv"
            df_tik_no_filt = pd.read_csv(csv_tik_no_filt)

        df_tik_no_filt = pd.read_csv(csv_tik_no_filt)
        df_tik_no_filt.columns = df_tik_no_filt.columns.str.replace('_', '', regex=True)  # Elimina los guiones bajos

        return df_dl, df_tik_no_filt, df_dl_or
    
    def summary_per_algorithm_stratified_tocsv(self, merged_df, df_dl, df_groups, algorithm_mod_name):
        """
        Compute mean ± std of metrics per algorithm, stratified by group/class.
        Saves one CSV per group.
        """

        # Merge with group information
        df_with_groups = pd.merge(merged_df, df_groups, on='name')  # Ajusta el 'on' si el identificador es otro


        # Extraer nombres únicos de métricas
        metric_names = list(set(col.split("_OMAMI")[0] for col in merged_df.columns if "_OMAMI" in col))

        for group, group_df in df_with_groups.groupby("Complexity"):
            metrics_summary = {}

            for algorithm in algorithm_mod_name:
                
                metric_values = {}
                for metric in metric_names:
                    algorithm_cols = [col for col in group_df.columns if metric in col and algorithm in col]
                    if len(algorithm_cols)==0:
                        raise ValueError(f"Algorithm columns not found for {algorithm} and {metric}") 
                    if algorithm_cols:

                        std_value = group_df[algorithm_cols].std(numeric_only=True).mean()

                        if pd.isna(std_value):
                            # Si la desviación estándar es NaN, asignar 0
                            metric_values[metric] = {
                                "Mean": group_df[algorithm_cols].mean(numeric_only=True).mean(),
                                "Std": 0
                            }

                        else:
                            metric_values[metric] = {
                                "Mean": group_df[algorithm_cols].mean(numeric_only=True).mean(),
                                "Std": group_df[algorithm_cols].std(numeric_only=True).mean()
                            }

                metrics_summary[algorithm] = pd.DataFrame(metric_values).T

            summary_df = pd.concat(metrics_summary, axis=1).T
            df_formatted = summary_df.copy()

            if isinstance(df_formatted.index, pd.MultiIndex):
                df_mean = np.round(df_formatted.xs('Mean', level=1), 4)
                df_std = np.round(df_formatted.xs('Std', level=1), 2)
                df_result = df_mean.astype(str) + " ± " + df_std.astype(str)
            else:
                print(f"Error: El DataFrame para el grupo {group} no tiene un MultiIndex.")

            # Reordenar columnas como df_dl
            #common_columns = [col for col in df_dl.columns if col in df_result.columns]
            #df_result = df_result[common_columns]

            # Guardar resultado
            path = f"{self.path_to_save_summary}/group_{group}_results_{self.name}.csv"
            df_result.to_csv(path, index=True)
            print(f"CSV saved for group {group} at: {path}")

    
    def summary_per_algorithm_tocsv(self,merged_df,df_dl,algorithm_mod_name):
        '''
        This function computes the mean and std of all metrics for each algorithm and saves it to a csv file
        
        
        '''


        # Extraer nombres únicos de métricas eliminando los nombres de algoritmos
        metric_names = list(set(col.split("_OMAMI")[0] for col in merged_df.columns if "_OMAMI" in col))

        # Crear un diccionario vacío para almacenar los resultados
        metrics_summary = {}

        # Iterar sobre cada algoritmo y calcular la media y std para cada métrica
        for algorithm in algorithm_mod_name:
            metric_values = {}
            for metric in metric_names:
                # Filtrar columnas que contienen la métrica y el algoritmo
                algorithm_cols = [col for col in merged_df.columns if metric in col and algorithm in col]
                
                # Calcular la media y std solo para estas columnas
                if algorithm_cols:
                    metric_values[metric] = {
                        "Mean": merged_df[algorithm_cols].mean(numeric_only=True).mean(),
                        "Std": merged_df[algorithm_cols].std(numeric_only=True).mean()
                    }

            # Guardar en el diccionario con el nombre del algoritmo
            metrics_summary[algorithm] = pd.DataFrame(metric_values).T

        # Concatenar los DataFrames de cada algoritmo con MultiIndex
        summary_df = pd.concat(metrics_summary, axis=1)
        summary_df=summary_df.T

        df_formatted = summary_df.copy()

        # Asegurar que el índice sea un MultiIndex con [algoritmo, estadística (Mean/Std)]
        if isinstance(df_formatted.index, pd.MultiIndex):
            # Aplicar la transformación Mean ± Std por cada métrica
            df_mean = np.round(df_formatted.xs('Mean', level=1),4)  # Extraer solo la fila de "Mean"
            df_std = np.round(df_formatted.xs('Std', level=1),2)  # Extraer solo la fila de "Std"

            # Combinar los valores en el formato "mean ± std"
            df_result = df_mean.astype(str) + " ± " + df_std.astype(str)

            # Mostrar el nuevo DataFrame
            

        else:
            print("Error: El DataFrame no tiene un MultiIndex con niveles (algoritmo, estadística).")

        #reorder columns
        #common_columns = [col for col in df_dl.columns if col in df_result.columns]
        #df_result = df_result[common_columns]

        path=f"{self.path_to_save_summary}/global_results_{self.name}.csv"
        df_result.to_csv(path, index=True)
        print('csv saved at: ', path)

        df_result.head(20)
    
    def summary_per_patient_tocsv(self, merged_df):
        '''
        This function computes the mean of all metrics for each patient and saves it to a csv file
        
        '''
        # Paso 1: Extraer los nombres de las métricas sin el sufijo del algoritmo
        metric_columns = [col for col in merged_df.columns if col != "name"]
        metric_dict = {col: col.split("_")[0] for col in metric_columns}  # Extraer solo la métrica

        # Paso 2: Renombrar las columnas para agrupar métricas con el mismo nombre
        df_metrics = merged_df.rename(columns=metric_dict)

        # EXCLUIR la columna "name" antes de calcular la media
        df_numeric = df_metrics.drop(columns=["name"])

        # Paso 3: Agrupar columnas con el mismo nombre y calcular la media en cada fila
        df_mean_metrics = df_numeric.groupby(axis=1, level=0).mean()

        # Paso 4: Reinsertar la columna "name" para identificar a cada paciente
        df_mean_metrics.insert(0, "name", merged_df["name"])

        df_mean_metrics.head(20)

        path=f"{self.path_to_save_summary}results_per_patient_{self.name}.csv"
        df_mean_metrics.to_csv(path, index=True)
        print('csv saved at: ', path)

        return df_mean_metrics
    
    def plot_individual_barplots_per_algorithm(self, df, algorithm_ID):


        algorithm_ID=algorithm_ID.replace("_", "")
        # Elegir la métrica a visualizar
        metrica = f"Correlation_{algorithm_ID}"  # Cambia esto por la métrica que te interese

        # Graficar
        plt.figure(figsize=(12, 6))
        plt.bar(df["name"], df[metrica], color="skyblue")
        plt.xticks(rotation=90, ha="right")  # Rotar etiquetas para mejor visibilidad
        plt.xlabel("Nombre")
        plt.ylabel(metrica)
        plt.title(f"Diagrama de barras de {metrica}")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.ylim([0, 0.8])
        plt.savefig(f"{self.path_to_save}/barplot_corr_{algorithm}.png")
        print(f"{self.path_to_save}/barplot_corr_{algorithm}.png")
        plt.close()


    def classification_rotor_complexity_tocsv(self, df):
        '''
        This function classifies each patient as Sinusal, Simple Rotor or Complex Rotor
        provide the mean metrics for each class (of all DL algorithms) and saves it to a csv file
        
        '''
        sinusal = {
            "Simulation_01_200316_001_  7", "Simulation_01_210209_001_002",
            "Simulation_01_210205_001_002", "Simulation_01_200428_001_001",
            "Simulation_01_200316_001_ 5", "Simulation_01_200428_001_005",
            "Simulation_01_200212_001_  7", "Simulation_01_210119_001_001",
            "Simulation_01_210205_001_003", "Simulation_01_201223_001_002",
            "Simulation_01_200212_001_  1", "Simulation_01_200428_001_003",
            "Simulation_01_200316_001_  3", "Simulation_01_200316_001_  4",
            "Sinusal_150629", "RA_RAA_141216", "Simulation_01_200428_001_002",
            "Simulation_01_210210_001_001", "Simulation_01_210208_001_002",
            "Simulation_01_200212_001_  9", "Simulation_01_200316_001_  1",
            "Simulation_01_200212_001_  6", "Simulation_01_200212_001_  5",
            "Simulation_01_210209_001_003", "Simulation_01_200316_001_  9",
            "Simulation_01_210209_001_001", "Simulation_01_200428_001_006",
            "Simulation_01_200316_001_  5",
        }

        rotor_simple = {
            "Simulation_01_200212_001_  2", "Simulation_01_191001_001_002",
            "Simulation_01_190717_001_001", "Simulation_01_191001_001_007",
            "Simulation_01_190717_001_004", "Simulation_01_190619_001_003",
            "Simulation_01_190619_001_004", "Simulation_01_190717_001_003",
            "Simulation_01_190502_001_006", "Simulation_01_190502_001_004",
            "Simulation_01_190717_001_002", "Simulation_01_190619_001_001",
            "Simulation_01_200316_001_  8", "Simulation_01_200428_001_004",
            "Simulation_01_200212_001_  8", "Simulation_01_200212_001_10",
            "Simulation_01_191001_001_001", "Simulation_01_200428_001_007",
            "Simulation_01_200428_001_009", "Simulation_01_200428_001_010",
            "Simulation_01_200316_001_  2", "Simulation_01_200316_001_  6",
            "Simulation_01_200316_001_10", "Simulation_01_190502_001_003",
            "Simulation_01_190619_001_002", "Simulation_01_191001_001_005",
            "Simulation_01_200212_001_  4", "Simulation_01_200212_001_ 10",
            "Simulation_01_200428_001_008", "Simulation_01_190502_001_005"

        }

        rotor_complejo = {
            "LA_RSPV_CAF_150115", "LA_RSPV_150113'", "LA_PLAW_140612",
            "LA_LSPV_150203", "LA_LSPV_150113", "RA_RAFW_140807",
            "RA_RAA_141230", "TwoRotors_181219", "LA_PLAW_140711_arm",
            "RA_RAFW_SAF_140730", "LA_RIPV_150121", "LA_LIPV_150119"
        }

        # Función para clasificar cada fila
        def classify_row(name):
            if name in sinusal:
                return 'Frente de Onda Sinusal'
            elif name in rotor_simple:
                return 'Rotor Simple'
            elif name in rotor_complejo:
                return 'Rotor Complejo'
            else:
                return 'Desconocido'
        
        df['classification'] = df['name'].apply(classify_row)
        path=f"{self.path_to_save_summary}/classification_all_metrics_{self.name}.csv"

        df.to_csv(path, index=True)
        print('csv saved at: ', path)

        return df
    
    def boxplot_per_algorithm(self, merged_df, algorithm_mod_name):
        '''
        This function plots boxplots that shows the mean distribution of DL algorithms and
        ZOT algorithms across algorithms.
        
        '''
        # Obtener todas las métricas únicas
        #metrics = ['Correlation', 'RMSE', 'PeakdetectorRecall', 'PeakdetectorPrecision', 'DTW', 'Coherence']
        #algorithms = algorithm_mod_name

        metrics = ['Correlation', 'RMSE', 'PeakdetectorRecall', 'PeakdetectorPrecision', 'DTW', 'Coherence']
        algorithms = algorithm_mod_name
        algorithms.append('Zot')  # Añadir ZOT como un algoritmo más

        # Reorganizar el DataFrame al formato largo
        long_df = merged_df.melt(
            id_vars=['name'],  # Mantener la columna 'name'
            value_vars=[f"{metric}_{algo}" for algo in algorithms for metric in metrics],
            var_name='metric_algorithm',  # Nueva columna que combina métricas y algoritmos
            value_name='value'  # Columna para los valores de las métricas
        )

        # Dividir 'metric_algorithm' en 'metric' y 'algorithm'
        long_df[['metric', 'algorithm']] = long_df['metric_algorithm'].str.rsplit('_', n=1, expand=True)

        # Generar un gráfico por cada métrica usando todos los valores de los pacientes
        for metric in metrics:
            plt.figure(figsize=(8, 5))  # Crear una nueva figura para cada métrica
            
            # Filtrar datos para la métrica actual (TODOS los valores de los pacientes)
            metric_data = long_df[long_df['metric'] == metric]

            
            # Crear el boxplot con la distribución real de los valores
            sns.boxplot(
                data=metric_data,
                x='algorithm',
                y='value',
                palette="tab10"  # Diferenciar algoritmos con colores
            )
            
            # Agregar puntos individuales (jitter) para ver la dispersión de los pacientes
            sns.stripplot(
                data=metric_data,
                x='algorithm',
                y='value',
                color="black",
                alpha=0.5, 
                jitter=True 
            )
            
            plt.title(f'{metric} Comparison Across Algorithms (All Patients)')
            plt.ylabel('Metric Value')
            plt.xlabel('Algorithm')
            plt.xticks(rotation=45)  

            plt.tight_layout()
            
            plt.savefig(f"{self.path_to_save}/boxplot_across_algorithms_{metric}.png")

            plt.close()




    
    def boxplot_per_patient(self, merged_df_dl, df_tik_merged):
        '''
        This function plots boxplots that shows the mean distribution of DL algorithms and
        ZOT algorithms across patients.
        
        '''

        # Extraer las columnas de cada dataset
        dl_columns = [col for col in merged_df_dl.columns if col != "name"]
        tik_columns = [col for col in df_tik_merged.columns if col != "name"]

        # Extraer las métricas comunes
        dl_metrics = {col.split("_")[0] for col in dl_columns}
        tik_metrics = {col.split("_")[0] for col in tik_columns}
        common_metrics = dl_metrics.intersection(tik_metrics)

        # Agrupar columnas por métrica
        dl_grouped = {metric: [col for col in dl_columns if col.startswith(metric)] for metric in common_metrics}
        tik_grouped = {metric: [col for col in tik_columns if col.startswith(metric)] for metric in common_metrics}


        # Iterar sobre cada métrica
        for metric in common_metrics:
            if metric not in dl_grouped or metric not in tik_grouped:
                continue  # Ignorar métricas no comunes

            dl_columns = dl_grouped[metric]
            tik_columns = tik_grouped[metric]

            dl_data = merged_df_dl[dl_columns]  # Todas las métricas DL para la métrica actual
            zot_data = df_tik_merged[tik_columns]  # Todas las métricas ZOT para la métrica actual

            plt.figure(figsize=(12, 6))

            num_pacientes = len(merged_df_dl)
            dl_positions = np.arange(num_pacientes) * 3  # Aumentamos la separación entre pacientes
            zot_positions = dl_positions + 0.6  # Acercamos DL y ZOT

            bp_dl = plt.boxplot(
                dl_data.values.T,  
                positions=dl_positions,
                widths=0.35,  
                patch_artist=True,
                boxprops=dict(facecolor="blue", color="blue"),
                medianprops=dict(color="white"),
                whiskerprops=dict(color="blue"),
                capprops=dict(color="blue"),
                flierprops=dict(markerfacecolor="blue", markeredgecolor="blue"),
            )

            bp_zot = plt.boxplot(
                zot_data.values.T,
                positions=zot_positions,
                widths=0.35,  
                patch_artist=True,
                boxprops=dict(facecolor="red", color="red"),
                medianprops=dict(color="white"),
                whiskerprops=dict(color="red"),
                capprops=dict(color="red"),
                flierprops=dict(markerfacecolor="red", markeredgecolor="red"),
            )

            x_labels = merged_df_dl["name"]  
            plt.xticks(dl_positions + 0.3, x_labels, rotation=90)  

            plt.title(f"Distribución de {metric} por Paciente (DL vs ZOT)")
            plt.xlabel("Pacientes")
            plt.ylabel(metric)
            
            dl_patch = mpatches.Patch(color="blue", label="DL")
            zot_patch = mpatches.Patch(color="red", label="ZOT")
            plt.legend(handles=[dl_patch, zot_patch], loc="upper right")

            plt.grid(axis="y", linestyle="--", alpha=0.7)

            path_to_save=self.path_to_save
            path=f"{path_to_save}/boxplot_per_patient_{metric}.png"
            os.makedirs(os.path.dirname(path), exist_ok=True)
            plt.savefig(path)
            print(f"Figure saved at: {path}")
    
    def barplot_algorithm(self, merged_df,algorithm_mod_name, patient_name='None' ):
        '''
        This function gives results for a specific patient as a barplot
        
        
        '''

        if not None:
            # Suponiendo que `merged_df` ya está creado
            # Seleccionar las métricas y los algoritmos
            metrics = ['Correlation', 'RMSE', 'PeakdetectorRecall', 'PeakdetectorPrecision', 'DTW', 'Coherence']
            algorithms = algorithm_mod_name

            # Reorganizar el DataFrame al formato largo
            long_df = merged_df.melt(
                id_vars=['name'],  # Mantener la columna 'name'
                value_vars=[f"{metric}_{algo}" for algo in algorithms for metric in metrics],
                var_name='metric_algorithm',  # Nueva columna que combina métricas y algoritmos
                value_name='value'  # Columna para los valores de las métricas
            )

            # Dividir 'metric_algorithm' en 'metric' y 'algorithm'
            long_df[['metric', 'algorithm']] = long_df['metric_algorithm'].str.rsplit('_', n=1, expand=True)

            # Seleccionar un paciente específico
            patient_data = long_df[long_df['name'] == patient_name]

            # Ajustar el tamaño de la figura para cada métrica
            plt.figure(figsize=(8, 5))

            # Iterar por cada métrica en el conjunto de métricas
            for metric in metrics:
                # Filtrar los datos para la métrica actual
                data_for_metric = patient_data[patient_data['metric'] == metric]
                
                # Crear un gráfico de barras para la métrica actual
                plt.figure()
                sns.barplot(
                    data=data_for_metric,
                    x='algorithm',  # Algoritmo en el eje X
                    y='value'       # Valor en el eje Y
                )
                
                # Configurar el título y las etiquetas
                plt.title(f'{metric} Comparison for Patient: {patient_name}')
                plt.ylabel('Metric Value')
                plt.xlabel('Algorithm')
                plt.xticks(rotation=45)  # Rotar las etiquetas en el eje X
                plt.tight_layout()       # Ajustar el diseño para evitar solapamientos
                
                path_to_save=self.path_to_save
                path=f"{path_to_save}/{patient_name}/barplot_per_patient_{metric}1.png"
                os.makedirs(os.path.dirname(path), exist_ok=True)
                plt.savefig(path)
                print(f"Figure saved at: {path}")
                plt.close()
    
    def stratified_evaluation():
        pass


        
    def __call__(self, *args, **kwds):

        algorithm_mod_name = [algo.replace('_', '') for algo in self.algorithm_list]
        cont=0
        merged_df=None
        for algorithm_ID in self.algorithm_list:
            algorithm_ID_mod=algorithm_mod_name[cont]
            cont+=1
            # Cargar el archivo CSV
            df_dl, df_tik_no_filt, df_dl_or = self.load_evaluation_dataframes(algorithm_ID)
            
            # Agregar sufijo al DataFrame actual
            df_dl = df_dl.add_suffix(f"_{algorithm_ID_mod}")
            df_dl.rename(columns={f'name_{algorithm_ID_mod}': 'name'}, inplace=True)

            # Si `merged_df` no está inicializado, usar el primer DataFrame como base
            if merged_df is None:
                merged_df = df_dl
            else:
                # Fusionar con el DataFrame base
                merged_df = pd.merge(merged_df, df_dl, on='name', how='outer')

        merged_df_dl=merged_df.copy() #ONLY DL ALGORITHMS

        # Add here Tikhonov variations names
        #algorithm_mod_name+=['ZotFilt']
        #algorithm_mod_name+=['ZotNoFilt']
        
        #Add ZOT results
        df = df_tik_no_filt.add_suffix(f"_Zot")
        df_filt = df
        df.rename(columns={f'name_Zot': 'name'}, inplace=True)
        merged_df = pd.merge(merged_df, df, on='name', how='outer')
        
        #Save merged
        merged_df.to_csv(f"{self.path_to_save_summary}merged_df_{self.name}.csv")

        #Save metric summaries
        if self.stratified:
            df_groups = self.load_class_df()
            self.summary_per_algorithm_stratified_tocsv(merged_df, df_dl, df_groups, algorithm_mod_name)
        
        # Save global results
        self.summary_per_algorithm_tocsv(merged_df,df_dl,algorithm_mod_name)
        df_summary_per_patient=self.summary_per_patient_tocsv(merged_df)

        #Figures
        #self.boxplot_per_patient(merged_df_dl, df_tik_no_filt)
        self.barplot_algorithm(merged_df,algorithm_mod_name, patient_name='LA_RSPV_CAF_150115' )
        self.boxplot_per_algorithm(merged_df, algorithm_mod_name)
        self.classification_rotor_complexity_tocsv(df_summary_per_patient)
        self.plot_individual_barplots_per_algorithm(df_dl, algorithm_ID=algorithm_ID)


if __name__ == "__main__":

    '''algorithm_list= ["OMAMI_VAE_no_filt_testing_repeated_no_filt_l2_strat_5_class_overs",
                    "OMAMI_VAE_no_filt_testing_repeated_no_filt_l2_strat_2_class_overs",
                    "OMAMI_VAE_no_filt_testing_repeated_no_filt_l2_tm_strat_2_class_overs_det",
                    "OMAMI_VAE_no_filt_testing_repeated_no_filt_l2_tm_strat_2_class_overs",
                    "OMAMI_no_filt_testing2_repeated_no_filt_l2_strat_5_class_overs",
                    "OMAMI_no_filt_testing2_repeated_no_filt_l2_strat_2_class_overs",
                    "OMAMI_no_filt_testing2_repeated_no_filt_l2_tm_strat_2_class_overs_det",
                    "OMAMI_no_filt_testing2_repeated_no_filt_l2_tm_strat_2_class_overs",
                    ]'''
    
    algorithm_list= ["OMAMI_VAE_no_filt_testing_repeated_no_filt_l2_strat_2_class_overs"]

    
    for algorithm in algorithm_list:

        ReportMetrics(name= 'stratified_evaluation',
                    algorithm_list=[algorithm],
                    test_id='0', 
                    stratified=True)()