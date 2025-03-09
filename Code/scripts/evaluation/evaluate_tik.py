import sys
sys.path.append("../Code")
import os
import json
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import time

from tools_.preprocess_data import Preprocess_Dataset
from tools_.load_dataset import LoadDataset_BSPS
from scripts.evaluation.metrics import Metrics
from scripts.Tikhonov.compute_tik import TikhonovReconstruction
from tools_.tools_inference import postprocess_prediction
from tools_.tools_inference import *
import tools_.tools as tools

test_patients = [
            "LA_PLAW_140711_arm", "LA_RSPV_CAF_150115",
            "Simulation_01_200212_001_  5", "Simulation_01_200212_001_ 10",
            "Simulation_01_200316_001_  3", "Simulation_01_200316_001_  4",
            "Simulation_01_200316_001_  8", "Simulation_01_200428_001_004",
            "Simulation_01_200428_001_008", "Simulation_01_200428_001_010",
            "Simulation_01_210119_001_001", "Simulation_01_210208_001_002"
        ]

class EvaluateTikhonov:
    def __init__(self, algorithm_ID, test_patients, test_id='', torso_num = 2):

        self.algorithm_ID = algorithm_ID
        self.start_time = time.time()
        self.torso_num = torso_num
        self.test_patients=test_patients
        self.name = ""
        self.test_id=test_id


    def configure(self):
        self.torso_num = 2
        self.data_dir = "/home/profes/miriamgf/tesis/Autoencoders/Data/"
        self.torsos_dir = "/home/profes/miriamgf/tesis/Autoencoders/Labeled_torsos/"
        self.experiment_dir = f"/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/experiments/experiments_VAE/{self.algorithm_ID}/"
        self.path_output_l_curva = f"/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/output/ZOT_L_curva/{self.algorithm_ID}/"
        os.makedirs(self.path_output_l_curva, exist_ok=True)
        self.output_directory = self.experiment_dir
        self.params_path = self.experiment_dir + 'hyperparams.json'
        self.tik= True
    


        with open(self.params_path) as file:
            self.params = json.load(file)
        
        
        self.n_batch=self.params["batch_size"]
        self.fs=self.params["fs"]
        self.fs_sub=self.params["fs_sub"]

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

    def load_and_process_patient(self, patient, cont):
        '''
        This function loads and preprocess test data for ZOT reconstruction
        Preprocessing includes:
            - BSPM reduced from the original number of nodes to 64 leads 
            - BSPM norm
            - EGM norm
            - EGM downsampled and batch split (only for visualization purposes, not for reconstruction)
        
        '''
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

        #Unpack
        torso_name = f"Torso{self.torso_num}_mod.mat"
        torso_index = self.all_torsos_names.index(torso_name)
        bspm_signal = y_list[torso_index]['y']
        transfer_matrix= transfer_matrices[torso_index]#[0]
        transfer_matrix_flat=transfer_matrices[torso_index][0]
        transfer_matrix_64=transfer_matrix[0][transfer_matrix[1].ravel(), :]
        bspm_signal_64=bspm_signal[transfer_matrix[1].ravel(), :]

        #Select only specified torso signals
        egm_single=np.split(egm_tensor, 10)[torso_index]
        X_1channel_single=np.split(X_1channel, 10)[torso_index]
        AF_models_single=np.split(np.array(AF_models), 10)[torso_index]
        Y_model_single=np.split(np.array(Y_model), 10)[torso_index]

        #normalize 
        bspm_signal_norm = normalize_array(bspm_signal_64.T, high=1, low=-1, axis_n=1) 
        egm_single_norm = normalize_array(egm_single, high=1, low=-1, axis_n=0) 

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
        self.divisible_rows = (X_1channel.shape[0] // self.n_batch) * self.n_batch


        return bspm_signal_norm, egm_tensor, Y_model, transfer_matrix_64


    def inference_tik(self, bspm_signal_norm, egm_flat, Y_model, transfer_matrix_64, patient):
        

        ObjTik = TikhonovReconstruction(bspm_signal_norm.T, transfer_matrix_64,
                                        order=0, path_figs=self.path_output_l_curva)
        tik_rec = ObjTik(plot_L_curve=True)
        tik_batches = ObjTik.tik_post_process_to_plot(tik_rec, self.fs_sub, self.divisible_rows, self.n_batch)
        tik_flat = tik_batches.reshape((tik_batches.shape[0] * tik_batches.shape[1], tik_batches.shape[2]))

        #save for plotting
        tik_dict = {
            "tik_rec": tik_rec.tolist()
        }
        path_to_save=f"{self.experiment_dir}{patient}_tik_array.json"
        with open(path_to_save, "w") as f:
            json.dump(tik_dict, f)
        print("Saved ZOT at:", path_to_save)

        return tik_flat, egm_flat
    
    def postprocess_tik(self, prediction):
        '''
        This
        '''
        prediction=prediction-np.mean(prediction)
        tik_rec_norm = normalize_array(prediction, high=1, low=-1, axis_n=0)

        return tik_rec_norm


    
    
    def run(self):

        self.configure()

        df_metrics_all_patients = []
        all_nodes_list = []

        for cont, patient in enumerate(self.test_patients, start=1):
            bspm_signal_norm, egm_flat, Y_model, transfer_matrix_64= self.load_and_process_patient(patient, cont)
            prediction, y_label = self.inference_tik(bspm_signal_norm, egm_flat, Y_model, transfer_matrix_64, patient)   
            prediction=self.postprocess_tik(prediction)         
            MetricsObj = Metrics(algorithm_ID=self.algorithm_ID, model_name=patient, tik=self.tik)
            df_metrics, metrics_all_nodes = MetricsObj.compute_metrics(prediction, y_label, fs=self.fs_sub)
            df_metrics_all_patients.append(df_metrics)
            all_nodes_list.append(metrics_all_nodes)

        # Guardar métricas
        #df = pd.DataFrame({"name": self.test_patients, "mean correlation": corr_list, "mean RMSE": rmse_list})
        df=pd.DataFrame(df_metrics_all_patients)
        df_all_nodes=pd.DataFrame(all_nodes_list)
        output_path1 = self.experiment_dir + f"metrics_tik_{self.test_id}.csv"
        df.to_csv(output_path1, index=False)
        output_path2 = self.experiment_dir + f"metrics_all_nodes_tik_{self.test_id}.csv"
        df_all_nodes.to_csv(output_path2, index=False)

        print("Metrics saved in", output_path1)
        print("Metrics lists saved in", output_path2)

        print("Execution time of DL evaluation:", time.time() - self.start_time, "sec")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Tikhonov script")
    parser.add_argument("--algorithm_ID", type=str, help="experiment name", default="OMAMI_VAE_no_filt")
    args = parser.parse_args()

    evaluator = EvaluateTikhonov(args.algorithm_ID, test_patients=test_patients)
    evaluator.run()


