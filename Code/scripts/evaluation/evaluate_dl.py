import sys
sys.path.append("../Code")
import os
import json
import argparse
import matplotlib.pyplot as plt
import pandas as pd
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
import time
import tensorflow as tf

import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from scipy.stats import pearsonr, spearmanr
from tools_.preprocess_data import Preprocess_Dataset
from scripts.config import ParseHiperparams
from models.multioutput_VAE import MultiOutput_VAE, SamplingLayer
from models.attention import LuongAttention
#from scripts.evaluation.tools_evaluate import normalize_array, downsampling
import tools_.tools as tools
from scripts.evaluation.metrics import Metrics
from tools_.load_dataset import LoadDataset_BSPS
from tools_.tools_inference import postprocess_prediction
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')



from scripts.Tikhonov.compute_tik import TikhonovReconstruction
from scripts.evaluate_function import *
from tools_.tools import corr_pearson_cols
from tools_.tools_inference import *
from tools_ import freq_phase_analysis as freq_pha

test_patients = [
            "LA_PLAW_140711_arm", "LA_RSPV_CAF_150115",
            "Simulation_01_200212_001_  5", "Simulation_01_200212_001_ 10",
            "Simulation_01_200316_001_  3", "Simulation_01_200316_001_  4",
            "Simulation_01_200316_001_  8", "Simulation_01_200428_001_004",
            "Simulation_01_200428_001_008", "Simulation_01_200428_001_010",
            "Simulation_01_210119_001_001", "Simulation_01_210208_001_002"
        ]

class EvaluateDL:


    def __init__(self, algorithm_ID, test_patients, torso_num = 2, test_id=''):
        self.algorithm_ID = algorithm_ID
        self.start_time = time.time()
        self.torso_num = torso_num
        self.test_patients=test_patients
        self.test_id=test_id


    def configure(self):
        # Configuración inicial
        self.torso_num = 1
        self.time_duration = 500

        self.geom_path_CF = "/home/pdi/miriamgf/tesis/Autoencoders/geometries/Atria_geom/Modelos_computacionales_Carlos_Fambuena/Atria.mat"
        self.geom_path_edgar = "/home/pdi/miriamgf/tesis/Autoencoders/geometries/Atria_geom/Modelos_Edgar/Atria.mat"
        self.data_dir = "/home/profes/miriamgf/tesis/Autoencoders/Data/"
        self.torsos_dir = "/home/profes/miriamgf/tesis/Autoencoders/Labeled_torsos/"

        self.experiment_dir = f"/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/experiments/experiments_VAE/{self.algorithm_ID}/"
        self.model_path_DL = self.experiment_dir + f"reconstructions_by_model_{self.algorithm_ID}.mat"
        self.weights_path = self.experiment_dir + "model_weights.h5"
        self.params_path = self.experiment_dir + 'hyperparams.json'

        # Cargar parámetros del modelo
        with open(self.params_path) as file:
            self.params = json.load(file)
        
        self.n_batch=self.params["batch_size"]
        self.fs=self.params["fs_sub"]

        '''
        if self.params["algorithm"] == "OMAMI_VAE":
            self.fs = 100
            self.n_batch = 200
        elif self.params["algorithm"] == "OMAMI":
            self.fs = 200
            self.n_batch = 400
        '''



        # Cargar nombres de torsos
        self.all_torsos_names = [
            file for _, _, files in os.walk(self.torsos_dir) for file in files if file.endswith(".mat")
        ]

        self.SNR_em_noise = None
        try:
            self.SNR_white_noise = self.params["SNR_white_noise"]
        except:
            self.SNR_white_noise = 100

        self.patches_oclussion = "PT"
        self.unfold_code = 1

        # Cargar nombres de torsos
        self.all_torsos_names = [
            file for _, _, files in os.walk(self.torsos_dir) for file in files if file.endswith(".mat")
        ]

        self.SNR_em_noise = None
        self.SNR_white_noise = 100
        self.patches_oclussion = "PT"
        self.unfold_code = 1
        

    def load_and_process_patient(self, patient, cont):
        print(f"LOADING PATIENT {cont}/{len(self.test_patients)}")
        model_name = [patient]

        # Cargar dataset
        (
            X_1channel,
            Y,
            Y_model,
            egm_tensor,
            length_list,
            AF_models,
            all_model_names,
            transfer_matrices,
            y_list,
        ) = LoadDataset_BSPS(
            self.params,
            directory=self.data_dir,
            data_type="1channelTensor",
            n_classes=self.params["n_classes"],
            downsampling=False,
            fs=self.params["fs"],
            norm=False,
            SR=True,
            n_batch=self.params["batch_size"],
            sinusoid=False,
            SNR_em_noise=self.SNR_em_noise,
            SNR_white_noise=self.SNR_white_noise,
            patches_oclussion=self.patches_oclussion,
            unfold_code=self.unfold_code,
            inference=False,
            select_model=model_name,
        )()

        # Preparar datos para inferencia
        torso_index = self.all_torsos_names.index(f"Torso{self.torso_num}_mod.mat")
        bspm_signal = y_list[torso_index]["y"]
        transfer_matrix = transfer_matrices[torso_index][0]
        egm_single = np.split(egm_tensor, 10)[torso_index]


        #Select only specified torso signals
        egm_single=np.split(egm_tensor, 10)[torso_index]
        X_1channel_single=np.split(X_1channel, 10)[torso_index]
        AF_models_single=np.split(np.array(AF_models), 10)[torso_index]
        Y_model_single=np.split(np.array(Y_model), 10)[torso_index]

        plt.figure(figsize=(20, 10))
        plt.subplot(2, 1, 1)
        plt.plot(egm_single[0:1000, 0], label='egm')
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(X_1channel_single[0:1000, 0, 0], label='bspm')
        plt.legend()
        plt.savefig("output/figures/evaluation_trash/X_1channel_loaded_ev.png")
        print("output/figures/evaluation_trash/X_1channel_loaded_ev.png")
        plt.close()

        #normalize 
        #bspm_signal_norm = normalize_array(bspm_signal.T, high=1, low=-1, axis_n=1) 
        #egm_single_norm = normalize_array(egm_single, high=1, low=-1, axis_n=0) 

        print(X_1channel.shape, egm_tensor.shape, Y_model.shape)    

        dic_vars={}
        (
        X_1channel, egm_tensor, AF_models, Y_model
        ) = Preprocess_Dataset(
            self.params,
            X_1channel_single,
            egm_single,
            list(AF_models_single),
            Y_model,
            dic_vars,
            Y,
            all_model_names,
            transfer_matrices,
            self.experiment_dir,
            norm_egm=True,
            inference=True
        )()

        rows = X_1channel.shape[0]
        n_batch=self.params["batch_size"]
        divisible_rows = (rows // n_batch) * n_batch
        #batch gen
        bsps_batches = reshape(
                        X_1channel,
                        (
                            int(len(X_1channel) / n_batch),
                            n_batch,
                            X_1channel.shape[1],
                            X_1channel.shape[2],
                            1,
                        ),
                    )
        egm_batches = reshape(
                        egm_tensor,
                        (
                            int(len(egm_tensor) / n_batch),
                            n_batch,
                            egm_tensor.shape[1],
                            1,
                        ),
                    )



        print(X_1channel.shape)

        return bsps_batches, egm_batches, Y_model, transfer_matrix, bspm_signal, egm_single

    def run_inference(self, X_1channel, egm_tensor, Y_model):
        # Cargar modelo
        try:
            model = load_model(self.weights_path)
        except:
            try:
                model = load_model(self.weights_path, 
                                custom_objects={'LuongAttention': LuongAttention, 'SamplingLayer': SamplingLayer})
                print('loading sampling layer')
            except:
                model = load_model(self.weights_path, 
                custom_objects={'SamplingLayer': SamplingLayer})
                print('loading sampling layer')


        # Inferencia
        try:
            prediction_array = model.predict(X_1channel, batch_size=1)
            _, prediction = prediction_array[0], prediction_array[1]

        except:

            def convert_model_to_float32(model):

                model_config = model.get_config()
                for layer in model_config["layers"]:
                    if "dtype" in layer["config"]:
                        layer["config"]["dtype"] = "float32"
                new_model = tf.keras.Model.from_config(model_config)
                new_model.set_weights([tf.cast(w, tf.float32) for w in model.get_weights()])
                return new_model
            
            print('float32 conversion')

            # Convertir el modelo
            model_32 = convert_model_to_float32(model)

            # Convertir datos de entrada a float32
            X_1channel_32 = tf.cast(X_1channel, tf.float32)

            # Inferencia con el modelo en float32
            prediction = model_32.predict(X_1channel_32, batch_size=1)[1]
        
        np.save("output/figures/evaluation_trash/X_1channel_inference.npy", X_1channel)
        X_1channel=np.squeeze(X_1channel, axis=-1)
        y_test_flat = reshape_tensor(X_1channel, n_dim_input=X_1channel.ndim, n_dim_output=2)

        reconstruction_flat_test = reshape_tensor(
            prediction, n_dim_input=prediction.ndim, n_dim_output=2
        )

        estimate_egms_n = reconstruction_flat_test

        #AF_models_test = AF_models_test[0:len(reconstruction_flat_test)]
        #estimate_egms_n = normalize_by_models(reconstruction_flat_test, AF_models_test)

        estimate_egms_n=tools.normalize_array(reconstruction_flat_test, high=1, low=-1, axis_n=0)

        def correlation_by_node(array1, array2):
            """
            Calcula la correlación de Spearman entre las columnas de dos arrays.

            Args:
                array1: un array de numpy de dimensión (n,m)
                array2: otro array de numpy de dimensión (n,m)

            Returns:
                Un array de numpy de dimensión (m,) que contiene la correlación de Spearman
                de las columnas de array1 y array2.
            """

            # Verificar si ambos arrays tienen las mismas dimensiones
            assert (
                array1.shape == array2.shape
            ), "Los arrays deben tener las mismas dimensiones."

            # Calcular la correlación de Spearman de las columnas de ambos arrays
            n_cols = array1.shape[1]
            print('Computing correlation in :', n_cols, 'nodes')
            corr = np.zeros(n_cols)
            for i in range(n_cols):
                corr[i], _ = spearmanr(array1[:, i], array2[:, i]) # or pearsonr

            return corr

        prediction_flat = prediction.reshape(prediction.shape[0] * prediction.shape[1], prediction.shape[2])
        egm_flat = egm_tensor.reshape(prediction.shape[0] * prediction.shape[1], prediction.shape[2])



        return prediction_flat, egm_flat
    
    def postprocess_data_DL(self, prediction):
        '''
        This function applies postprocessing to enchance AI predictions including

            - removing DC component + detrending
            - Low pass filter to remove noise (noise considered f> FPA_cutoff)
            - normalizing -1 and 1
        '''
        print('Applying postprocessing...')
        prediction_post=postprocess_prediction(prediction, fs=self.fs, FPA_cutoff=15,cutoff_DC=1, axis=0)
        
        try:
            assert prediction_post[:, 0].max() == 1
        except AssertionError:
            print('Prediction not normalized!')
            sys.exit()


        return prediction_post
    
    def run(self):

        self.configure()

        df_metrics_all_patients = []
        all_nodes_list = []

        for cont, patient in enumerate(self.test_patients, start=1):
            
            X_1channel, egm_tensor, Y_model, _, _, _ = self.load_and_process_patient(patient, cont)
            prediction, y_label = self.run_inference(X_1channel, egm_tensor, Y_model)
            prediction=self.postprocess_data_DL(prediction)
            MetricsObj = Metrics(algorithm_ID=self.algorithm_ID, model_name=patient, tik=False)
            df_metrics, metrics_all_nodes = MetricsObj.compute_metrics(prediction, y_label, fs=self.fs)
            df_metrics_all_patients.append(df_metrics)
            all_nodes_list.append(metrics_all_nodes)

        # Guardar métricas
        #df = pd.DataFrame({"name": self.test_patients, "mean correlation": corr_list, "mean RMSE": rmse_list})
        df=pd.DataFrame(df_metrics_all_patients)
        df_all_nodes=pd.DataFrame(all_nodes_list)
        
        output_path1 = self.experiment_dir + f"metrics_dl_{self.test_id}.csv"
        df.to_csv(output_path1, index=False)
        output_path2 = self.experiment_dir + f"metrics_all_nodes_dl_{self.test_id}.csv"
        df_all_nodes.to_csv(output_path2, index=False)

        print("Metrics saved in", output_path1)
        print("Metrics lists saved in", output_path2)
        print("Execution time of DL evaluation:", time.time() - self.start_time, "sec")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DL script")
    parser.add_argument("--algorithm_ID", type=str, help="experiment name", default="OMAMI_VAE_no_filt")
    args = parser.parse_args()

    evaluator = EvaluateDL(algorithm_ID=args.algorithm_ID, test_patients=test_patients)
    evaluator.run()
