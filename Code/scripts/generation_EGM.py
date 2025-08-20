# This script was developed Miriam Gutiérrez Fernández

import sys
sys.path.append("../Code")
import argparse
import datetime
import os
import pickle
import json
import random
import time


import matplotlib.pyplot as plt
#import mlflow
import scipy
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from numpy import *
from scipy.io import savemat
import h5py
from config import str_to_bool

import tools_
import tools_.tools
 
from models.gen_vae_2d import Gen_VAE_2D
from models.gen_vae_2d_v2_cond import Gen_VAE_2D_v2_Cond
from models.gen_vae_2d_v2 import Gen_VAE_2D_v2

from tools_.df_mapping import *
from tools_.tools import *
from tools_.tools_1 import normalize_array
from tools_.signal_gen import SyntheticDataGenerator
from tools_.evaluate_gen import EvaluateGen
from tools_.generation_utils import plot_pca, plot_tsne



tf.random.set_seed(42)
import argparse
import datetime
import time

import tensorflow as tf
from config import ParseHiperparams
from keras import backend as K

from tools_.load_dataset import LoadDataset
from tools_.preprocess_data import Preprocess_Dataset
from tools_.preprocessing_compression import *
from tools_.train_model_gen import TrainModelGen

# Clear GPU
K.clear_session()
tf.keras.backend.clear_session()
tf.compat.v1.reset_default_graph()

#PRECONFIG
params = ParseHiperparams().parse_default_hyperparams()


params["oversampling"] = False
params["discard_classes"] =[0,1, 3, 5]
params["classes_to_oversample"]= []
params["latent_dim"]=50
params['early_stopping_patience']=30
params["beta_warmup_epochs"]= 20
params["select_classes"]= [2,4]
params["filter_EGM"]=False

if len(params['select_classes'])>1:
    params["split_mode"] = "stratified"
else:
    params["split_mode"] = "random"

try:
    print("parsing")
    parser = argparse.ArgumentParser(description="Noise params")
    parser.add_argument("--algorithm", type=str, help="experiment name", required=False)
    parser.add_argument("--optuna", type=str_to_bool, help="True or False", required=False)
    parser.add_argument("--n_nodes", type=int, help="682, 1024", required=False)
    parser.add_argument("--evaluation", type=str_to_bool, help="evaluation", required=False)

    args = parser.parse_args()

    # Lee argumentos con fallback a params si no se pasan
    algorithm = args.algorithm if args.algorithm is not None else params.get("algorithm")
    n_nodes = args.n_nodes if args.n_nodes is not None else params.get("n_nodes_regression")

    # Normaliza booleanos
    optuna = bool(args.optuna) if args.optuna is not None else False
    evaluation_ = bool(args.evaluation) if args.evaluation is not None else True

    # Mensaje de evaluación
    if evaluation_:
        print("Evaluation mode activated")

    # Actualiza params
    if algorithm is not None:
        params["algorithm"] = algorithm
    if n_nodes is not None:
        params["n_nodes_regression"] = n_nodes
    params["optuna_optimization"] = optuna

except Exception as e:
    # Manejo limpio del error y valores de reserva
    print(f"Error al parsear argumentos: {e}")
    algorithm = params.get("algorithm")

evaluation_ = True  
dataset_generator_=False
train_=False


print('Params to train: ', params)

SNR_em_noise = None
SNR_white_noise = 100
patches_oclussion = "PT"
experiment_number = 0
unfold_code = 1
experiment_name = algorithm

experiment_name="Gen_VAE_2D_v2"

experiment_name=experiment_name+'_develop_cond'
params["experiment_name"] = experiment_name

root_logdir = "output/logs/"
log_dir = root_logdir + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
data_dir = "/home/profes/miriamgf/tesis/Autoencoders/Data/"
torsos_dir = "../../../../Labeled_torsos/"
figs_dir = "output/figures/"
models_dir = "output/model/"
dict_var_dir = "output/variables/"
dict_results_dir = "output/results/"
experiment_dir = f"output/experiments/synthetic_generation/{experiment_name}/"


if not os.path.exists(experiment_dir):
    os.makedirs(experiment_dir)    
print('Experiment dir', experiment_dir)

dic_vars = {}
dict_results = {}

# GPU Configuration
physical_devices = tf.config.list_physical_devices("GPU")
print("Num GPUs:", len(physical_devices))
for gpu in tf.config.experimental.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

start = time.time()

all_torsos_names = []
for subdir, dirs, files in os.walk(torsos_dir):
    for file in files:
        if file.endswith(".mat"):
            all_torsos_names.append(file)


all_model_names = []
directory = data_dir
for subdir, dirs, files in os.walk(directory):
    # print(subdir, directory, files)

    if subdir != directory:
        model_name = subdir.split("/")[-1]
        all_model_names.append(model_name)

all_model_names = sorted(all_model_names)
print(all_model_names)


#mlflow.set_tracking_uri(uri="http://10.110.100.78:5000")
#mlflow.autolog()

# Load data
if params["fs"] == params["fs_sub"]:
    params["fs"] = params["fs_sub"]

Transfer_model = False  # Transfer learning from sinusoids
sinusoids = False

#CONFIGURE
params["test_source"]=["Simulation_01_200316_001_  5",
                    "Simulation_01_200212_001_  7",
                    'Simulation_01_210205_001_003',
                    'Simulation_01_200428_001_009',
                    'Simulation_01_210209_001_003']
params["val_source"]=None

# Load data
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
    class_complexity_list
) = LoadDataset(
    params,
    directory=directory,
    data_type="1channelTensor",
    n_classes=params["n_classes"],
    downsampling=False, #deprecated
    fs=params["fs"],
    norm=False,
    SR=True,
    n_batch=params["batch_size"],
    SNR_em_noise=SNR_em_noise,
    SNR_white_noise=SNR_white_noise,
    patches_oclussion=patches_oclussion,
    unfold_code=unfold_code,
    inference=False,
    all_classes=False
)()


(
    x_train,
    x_test,
    x_val,
    y_train,
    y_test,
    y_val,
    dic_vars,
    BSPM_train,
    BSPM_test,
    BSPM_val,
    AF_models_train,
    AF_models_test,
    AF_models_val,
    class_complexity_list_train,
    class_complexity_list_test,
    class_complexity_list_val,
    train_models,
    test_models,
    val_models,
) = Preprocess_Dataset(
    params,
    X_1channel,
    egm_tensor,
    AF_models,
    class_complexity_list,
    Y_model,
    dic_vars,
    Y,
    all_model_names,
    transfer_matrices,
    experiment_dir,
    split_mode=params["split_mode"],
    norm_egm=True,
    shuffle_patient= True, 
)()

class_complexity_list_train=class_complexity_list_train[:, 0]
class_complexity_list_test=class_complexity_list_test[:, 0]
class_complexity_list_val=class_complexity_list_val[:, 0]


params["algorithm"] = "gen_VAE_2D_v2_cond"

print("Algorithm selected:", params["algorithm"])

if params["algorithm"] == "gen_VAE_3D":

    y_train=y_train.reshape(y_train.shape[0], y_train.shape[1], 32, 64)
    y_val=y_val.reshape(y_val.shape[0], y_val.shape[1], 32, 64)
    y_test=y_test.reshape(y_test.shape[0], y_test.shape[1], 32, 64)


if train_:
    
    print('Starting training...')
    params["n_epochs"] = 50

    # Guardar parámetros como JSON en experiment_dir
    params_path = os.path.join(experiment_dir, "params.json")
    with open(params_path, "w") as f:
        json.dump(params, f, indent=4)  

    if params['algorithm']=="gen_VAE_2D_v2_cond":
        model, history = TrainModelGen(
            params, y_train, y_test, y_val, y_train, y_test, y_val, models_dir, experiment_dir, 
            c_train=class_complexity_list_train,   
            c_val=class_complexity_list_val
        )()
    
    else:
        model, history = TrainModelGen(
            params, y_train, y_test, y_val, y_train, y_test, y_val, models_dir, experiment_dir
        )()

    print(f"Parámetros guardados en {params_path}")

if evaluation_:

    if params['algorithm']=="gen_VAE_2D_v2_cond":
        vae = Gen_VAE_2D_v2_Cond(
                params,
                input_shape_=y_train.shape[1:],   # (400, 2048)
                n_nodes=2048,
                tensorboard_logs=experiment_dir + "tb_logs/",
                latent_dim=params["latent_dim"],
                num_classes=2)
        vae.build(input_shape=[(None,) + y_train.shape[1:], (None, 2)])  # (x, one-hot)

    elif params['algorithm']=="gen_VAE_2D_warmup":
        vae = Gen_VAE_2D(
            params=params,
            input_shape_=y_test.shape[1:], 
            n_nodes=2048,
            latent_dim=params["latent_dim"],
            tensorboard_logs=experiment_dir + "tb_logs/")
    elif params['algorithm']=="gen_VAE_2D_v2":
            vae = Gen_VAE_2D_v2(params,
                                y_train.shape[1:],
                                n_nodes=2048,
                                tensorboard_logs=experiment_dir + "tb_logs/",
                                latent_dim=params["latent_dim"])
            
            vae.build(input_shape=(None,) + y_train.shape[1:])   # create variables
            vae.load_weights(model_name + "model_weights.h5")

    
    #model_name
    if train_:
        model_name=f"/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/experiments/synthetic_generation/{experiment_name}/"
        experiment_dir=model_name
    else:
        model_name="/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/experiments/synthetic_generation/Gen_VAE_2D_v2_develop_cond/"
        experiment_dir=model_name
    
    #load weights
    try:
        vae.model.load_weights(model_name + "model_weights.h5")
    except:
        #vae.build(input_shape=(None,) + y_train.shape[1:])   # create variables
        vae.load_weights(model_name + "model_weights.h5")


    N=100
    x_real= y_train[:N, :, :]
    label_real=class_complexity_list_train[:N].astype(np.int32)
    batch_classes=class_complexity_list_train[:N].astype(np.int32)

    class_labels = {
    4: "Sinusal",
    3: "AF"}

    if params["algorithm"]=="gen_VAE_2D_v2_cond":
            z, z_mean_train, z_log_var = vae.encode(x_real,label_real)
    else:
        # Latent exploration 
        try:
            z, z_mean_train, _ = vae.build_encoder_module(x_real, vae.input_shape_)
        except:
            z, z_mean_train, z_log_var = vae.encoder(x_real, training=False)

    #evaluate
    labels_test = class_complexity_list_test if "class_complexity_list_test" in locals() else None
    evaluator = EvaluateGen(vae, latent_dim=params["latent_dim"], save_dir=experiment_dir,
                         conditional_classes=(2, 4))
    results = evaluator.run_all(x_real, labels_test=label_real, num_generated=len(y_test), pca_bins=20)
    print(json.dumps(results, indent=2))
    #Plots condicionados (misma X real, forzando c=2 y c=4):

    
    
    # Usa labels si quieres silhouette/linear probe (en tu script ya tienes class_complexity_list_test)

    results = evaluator.run_all(
        y_test=y_test,
        labels_test=labels_test,
        num_generated=len(y_test),   # o pon un número menor para ir más rápido
        pca_bins=20
    )

    print("=== Resultados de Evaluación ===")
    for k, v in results.items():
        print(k, "->", v)

    save_dir="/home/pdi/miriamgf/tesis/Autoencoders/Data_generated"
    if dataset_generator_:
        gen = SyntheticDataGenerator(
            vae_model=vae,
            latent_dim=50,
            save_dir="./synthetic_out",
            experiment_dir=experiment_dir,
            params={"experiment_name": "vae_cond"},
            sampling="guided",
            conditional_classes=[2,4]
        )

        signals_by_class = gen.generate_for_classes(
            num_per_class=200,
            save_prefix=experiment_name,
            z_mean_train=z_mean_train,
            plot_real_signals=y_train,
            class_labels=class_complexity_list_train,
            select_best=True,
            num_selected=25,
            num_classes=2   
        )


        '''gen.plot_examples(signals_by_class[2], 
                            experiment_dir=experiment_dir,
                            class_complexity_list_train=class_complexity_list_train,
                            real_signals=y_train,
                            num_examples=3, max_nodes=3)'''

    # Visualize latent space
    plot_pca(z_mean_train,batch_classes, class_labels, experiment_dir)
    plot_tsne(z_mean_train,batch_classes, class_labels, experiment_dir)


    # Samplear del espacio latente: random, guided, interpolation, reconstruction
    random_sampling = True
    guided_sampling = True
    reconstruction = True
    reconstruction_2D = True
    interpol = True
    manifold = True

    if random_sampling: 
        z = tf.random.normal((1, vae.latent_dim))  # 1 muestras aleatorias
        if params["algorithm"]=="gen_VAE_2D_v2_cond":
            c = np.int32(2)
            synthetic = vae.decode(z,c)
        else:
            c = np.int32(4)

            try:
                synthetic = vae.decode_from_latent(z)
            except:
                synthetic = vae.decoder(z, training=False)

        synthetic=np.array(synthetic)
        synthetic = synthetic.squeeze()
        synthetic_norm = np.zeros(synthetic.shape)
        for i in range(synthetic.shape[1]):
            synthetic_norm[ :, i] = normalize_array(synthetic[ :, i], axis_n=0, high=1.0, low=-1.0)
        

        classes_train=class_complexity_list_train[:].astype(np.int32)
        idx = np.where(classes_train == c)[0]
        # selecciona esas muestras de y_train
        y = y_train[idx]
        
        plt.figure(figsize=(8,6), tight_layout=True)
        plt.subplot(2, 1, 1)
        plt.title("Synthetic signal")
        plt.imshow(synthetic_norm[:, : ], aspect="auto")
        plt.subplot(2, 1, 2)
        plt.title("Real signal")
        plt.imshow(y[0, :, :], aspect="auto")
        plt.savefig(
             f"{experiment_dir}synthetic_signal_random_sampling_c_{c}.png", dpi=300, bbox_inches="tight"
        )
        print(f"{experiment_dir}synthetic_signal_random_sampling_c_{c}.png")
        plt.close()

        for node in range(0, 2047, 500):


            plt.figure(figsize=(8,6), tight_layout=True)
            plt.subplot(2, 1, 1)
            plt.title("Synthetic signal")
            plt.plot(synthetic[0:400, node])
            plt.subplot(2, 1, 2)
            plt.title("Real signal")
            plt.plot(y[0,0:400, node ])
            plt.savefig(
                experiment_dir + f"random_sampling_1D_{node}_c{c}.png", dpi=300, bbox_inches="tight"
            )
            print(experiment_dir + f"random_sampling_1D_{node}_c{c}.png")
            plt.close()

        plt.figure(figsize=(8,6), tight_layout=True)

        plt.hist(z.numpy().flatten(), bins=100)
        plt.title("Distribución de z_mean")
        plt.savefig(
            experiment_dir + "histogram_rand.png", dpi=300, bbox_inches="tight"
        )
        plt.close()



    if guided_sampling:

        if params["algorithm"]=="gen_VAE_2D_v2_cond":
            z, z_mean_train, z_log_var = vae.encode(x_real,label_real)
        else:
            try:
                z, z_mean_train, _ = vae.build_encoder_module(x_real, vae.input_shape_)
            except:
                z, z_mean_train, z_log_var = vae.encoder(x_real, training=False)

        mu = np.mean(z_mean_train, axis=0)
        sigma = np.std(z_mean_train, axis=0)

        z = np.random.normal(loc=mu, scale=sigma, size=(10, params["latent_dim"]))

        if params["algorithm"]=="gen_VAE_2D_v2_cond":
            c=2
            c_=np.ones(len(z))*c
            synthetica = vae.decode(z, c_)
        else:

            try:
                synthetica = vae.decode_from_latent(z)
            except:
                synthetica = vae.decoder(z, training=False)

        for example in range(0,10):

            classes_train=class_complexity_list_train[:].astype(np.int32)
            idx = np.where(classes_train == c)[0]
            # selecciona esas muestras de y_train
            y = y_train[idx]
            one_synthetic = synthetica[example, :, :]

            synthetic=np.array(one_synthetic)
            synthetic = synthetic.squeeze()
            synthetic_norm = np.zeros(synthetic.shape)
            for i in range(synthetic.shape[1]):
                synthetic_norm[ :, i] = normalize_array(synthetic[ :, i], axis_n=0, high=1.0, low=-1.0)
            
            #plot 2d

            plt.figure(tight_layout=True)
            plt.subplot(2, 1, 1)
            plt.title("Synthetic signal")
            plt.imshow(synthetic_norm[:, : ], aspect="auto")
            plt.subplot(2, 1, 2)
            plt.title("Real signal")
            plt.xlabel("Number of nodes of EGM signals")
            plt.ylabel("Number of time samples")
            plt.imshow(y[0, :, :], aspect="auto")
            plt.savefig(
                experiment_dir + f"synthetic_signal_guided_sampling_{example}_2d_c{c}.png", dpi=300, bbox_inches="tight"
            )
            print(experiment_dir + f"synthetic_signal_guided_sampling_{example}_2d_c{c}.png")
            plt.close()

            #plot 1d

            for node in range(0, 2047, 100):

                plt.figure(figsize=(8,6), tight_layout=True)

                plt.subplot(2, 1, 1)
                plt.title("Synthetic signal")
                plt.plot(synthetic[0:400, node])
                plt.subplot(2, 1, 2)
                plt.title("Real signal")
                plt.plot(y[0,0:400, node ])
                plt.savefig(
                    experiment_dir + f"guided_sampling_1D_{example}_{node}_c{c}.png", dpi=300, bbox_inches="tight"
                )
                print(experiment_dir + f"guided_sampling_1D_{example}_{node}_c{c}.png")
                plt.close()

        #plot histogram z
        plt.figure(figsize=(8,6), tight_layout=True)
        plt.hist(z_mean_train.numpy().flatten(), bins=100)
        plt.title("Distribución de z_mean")
        plt.savefig(
            experiment_dir + f"histogram_guided_c{c}.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        #plot z
        plt.figure(figsize=(8, 6))
        plt.scatter(z_mean_train[:, 0], z_mean_train[:, 1], alpha=0.5)
        plt.title("Distribución del espacio latente (2D)")
        plt.xlabel("z1")
        plt.ylabel("z2")
        plt.grid(True)
        plt.savefig(experiment_dir + f"latent_space_guided_c{c}.png", dpi=300, bbox_inches="tight")


    if reconstruction:

        x_real_one_batch= np.expand_dims(x_real[0, :, :], axis=0)
        labels_real_one_batch=label_real[0]

        if params["algorithm"]=="gen_VAE_2D_v2_cond":
            z, z_mean_train, z_log_var = vae.encode(x_real_one_batch,labels_real_one_batch)
        else:
            try:
                z, z_mean_train, _ = vae.build_encoder_module(x_real_one_batch, vae.input_shape_)
            except:
                z, z_mean_train, z_log_var = vae.encoder(x_real_one_batch, training=False)

        if params["algorithm"]=="gen_VAE_2D_v2_cond":
            synthetic = vae.decode(z, labels_real_one_batch)
        else:
            try:
                synthetic = vae.decode_from_latent(z)
            except:
                synthetic = vae.decoder(z, training=False)

        synthetic=np.array(synthetic)
        synthetic = synthetic.squeeze()
        synthetic_norm = np.zeros(synthetic.shape)
        for i in range(synthetic.shape[1]):
            synthetic_norm[ :, i] = normalize_array(synthetic[ :, i], axis_n=0, high=1.0, low=-1.0)

        plt.figure(figsize=(8,6), tight_layout=True)

        plt.subplot(2, 1, 1)
        plt.title("Synthetic signal")
        plt.imshow(synthetic_norm[0,:, : ], aspect="auto")
        plt.subplot(2, 1, 2)
        plt.title("Real signal")
        plt.imshow(y_train[0, :, :], aspect="auto")
        plt.savefig(
            experiment_dir + "synthetic_signal_reconstruction_2d.png", dpi=300, bbox_inches="tight"
        )
        print(experiment_dir + "synthetic_signal_reconstruction_2d.png")
        plt.close()

        #signal
        plt.figure(figsize=(8,6), tight_layout=True)

        plt.subplot(2, 1, 1)
        plt.title("Synthetic signal")
        plt.plot(synthetic[0:100, 0 ])
        plt.subplot(2, 1, 2)
        plt.title("Real signal")
        plt.plot(y_train[0,0:400, 0 ])
        plt.savefig(
            experiment_dir + "reconstruction_1D.png", dpi=300, bbox_inches="tight"
        )
        print(experiment_dir + "reconstruction_1D.png")
        plt.close()

        plt.figure(figsize=(8,6), tight_layout=True)

        plt.hist(z_mean_train.numpy().flatten(), bins=100)
        plt.title("Distribución de z_mean")
        plt.savefig(
            experiment_dir + "histogram_rec.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        plt.figure(figsize=(8, 6))
        plt.scatter(z_mean_train[:, 0], z_mean_train[:, 1], alpha=0.5)
        plt.title("Distribución del espacio latente (2D)")
        plt.xlabel("z1")
        plt.ylabel("z2")
        plt.grid(True)
        plt.savefig(experiment_dir + "latent_space_rec.png", dpi=300, bbox_inches="tight")

    if interpol:

        x_real_1= np.expand_dims(x_real[0, :, :], axis=0)
        x_real_2= np.expand_dims(x_real[9, :, :], axis=0)
        label_real_1=label_real[0]
        label_real_2=label_real[9]

        if params["algorithm"]=="gen_VAE_2D_v2_cond":
            z1, z_mean_train1, z_log_var1 = vae.encode(x_real_1,label_real_1)
            z2, z_mean_train2, z_log_var2 = vae.encode(x_real_2,label_real_2)

        else:

            try:
                z1, z_mean_train1, _ = vae.build_encoder_module(x_real_1, vae.input_shape_)
                z2, z_mean_train2, _ = vae.build_encoder_module(x_real_2, vae.input_shape_)

            except:
                z1, z_mean_train1, _ = vae.encoder(x_real_1, training=False)
                z2, z_mean_train2, _ = vae.encoder(x_real_2, training=False)
            
        alpha=0.1

        z_interp = (1-alpha) * z1 + alpha * z2

        if params["algorithm"]=="gen_VAE_2D_v2_cond":
            c=2
            c_=np.ones(len(z_interp))*c
            synthetic = vae.decode(z_interp, c_)                  
        
        else:
            try:
                synthetic = vae.decode_from_latent(z_interp)
            except:
                synthetic = vae.decoder(z_interp, training=False)

        classes_train=class_complexity_list_train[:].astype(np.int32)
        idx = np.where(classes_train == c)[0]
        # selecciona esas muestras de y_train
        y = y_train[idx]
        synthetic=np.array(synthetic)
        synthetic = synthetic.squeeze()
        synthetic_norm = np.zeros(synthetic.shape)
        for i in range(synthetic.shape[1]):
            synthetic_norm[ :, i] = normalize_array(synthetic[ :, i], axis_n=0, high=1.0, low=-1.0)

        plt.figure(figsize=(8,6), tight_layout=True)

        plt.subplot(2, 1, 1)
        plt.title("Synthetic signal")
        plt.imshow(synthetic_norm[:, : ], aspect="auto")
        plt.subplot(2, 1, 2)
        plt.title("Real signal")
        plt.imshow(y[0, :, :], aspect="auto")
        plt.savefig(
            experiment_dir + f"synthetic_signal_interp_{c}.png", dpi=300, bbox_inches="tight"
        )
        print(experiment_dir + f"synthetic_signal_interp_{c}.png")
        plt.close()

        #signal
        plt.figure(figsize=(8,6), tight_layout=True)

        plt.subplot(2, 1, 1)
        plt.title("Synthetic signal")
        plt.plot(synthetic[0:400, 0 ])
        plt.subplot(2, 1, 2)
        plt.title("Real signal")
        plt.plot(y[0,0:400, 0 ])
        plt.savefig(
            experiment_dir + f"interp_1D_{c}.png", dpi=300, bbox_inches="tight"
        )
        print(experiment_dir + f"interp_1D_{c}.png")
        plt.close()
    
    if manifold:
        
        if params["algorithm"]=="gen_VAE_2D_v2_cond":
                z, z_mean_train, z_log_var = vae.encode(x_real,label_real)
        else:
            try:
                z, z_mean_train, _ = vae.build_encoder_module(x_real, vae.input_shape_)
            except:
                z, z_mean_train, _ = vae.encoder(x_real, training=False)

        idx = np.random.choice(len(z_mean_train), size=10)
        z_mean_train = np.array(z_mean_train)
        z_base = z_mean_train[idx]
        label_sample=label_real[idx]
        z_sample = z_base + np.random.normal(scale=0.01, size=z_base.shape)



        if params["algorithm"]=="gen_VAE_2D_v2_cond":
            c=2
            c_=np.ones(len(z_sample))*c
            synthetic = vae.decode(z_sample, c_)                  
        
        else:
            try:
                synthetic = vae.decode_from_latent(z_sample)
            except:
                synthetic = vae.decoder(z_sample, training=False)


        for example in range(0,10):

            classes_train=class_complexity_list_train[:].astype(np.int32)
            idx = np.where(classes_train == c)[0]
            # selecciona esas muestras de y_train
            y = y_train[idx]

            one_synthetic = synthetica[example, :, :]

            synthetic=np.array(one_synthetic)
            synthetic = synthetic.squeeze()
            synthetic_norm = np.zeros(synthetic.shape)
            for i in range(synthetic.shape[1]):
                synthetic_norm[ :, i] = normalize_array(synthetic[ :, i], axis_n=0, high=1.0, low=-1.0)
            
            

            plt.figure(figsize=(8,6), tight_layout=True)

            plt.subplot(2, 1, 1)
            plt.title("Synthetic signal")
            plt.imshow(synthetic_norm[:, : ], aspect="auto")
            plt.subplot(2, 1, 2)
            plt.title("Real signal")
            im1=plt.imshow(y[0, :, :], aspect="auto")
            plt.colorbar(im1, orientation='vertical', fraction=0.046, pad=0.04)  # ← Añade barra lateral

            plt.savefig(
                experiment_dir + f"synthetic_signal_guided_sampling_{example}_{c}.png", dpi=300, bbox_inches="tight"
            )
            print(experiment_dir + f"synthetic_signal_guided_sampling_{example}_{c}.png")
            plt.close()

            #signal
            plt.figure(figsize=(8,6), tight_layout=True)

            plt.subplot(2, 1, 1)
            plt.title("Synthetic signal")
            plt.plot(synthetic[0:400, 0 ])
            plt.subplot(2, 1, 2)
            plt.title("Real signal")
            plt.plot(y[0,0:400, 0 ])
            plt.savefig(
                experiment_dir + f"manifold_1D_{example}_{c}.png", dpi=300, bbox_inches="tight"
            )
            print(experiment_dir + f"manifold_1D_{example}_{c}.png")
            plt.close()

    if reconstruction_2D:

            x_sample = y_train[:10]  # Una muestra
            label_sample=label_real[:10]

            if params["algorithm"]=="gen_VAE_2D_v2_cond":
                    z, z_mean_train, z_log_var = vae.encode(x_sample,label_sample)
            else:
                try:
                    z, z_mean_train, _ = vae.build_encoder_module(x_sample, vae.input_shape_)
                except:
                    z, z_mean_train, _ = vae.encoder(x_sample, training=False)

            if params["algorithm"]=="gen_VAE_2D_v2_cond":
                reconstructed = vae.decode(z_mean_train, label_sample)                  
            
            else:
                try:
                    reconstructed = vae.decode_from_latent(z_mean_train)
                except:
                    reconstructed = vae.decoder(z_mean_train, training=False)


            # Prediction
            plt.figure(figsize=(12, 6))

            plt.subplot(1, 2, 1)
            plt.imshow(x_sample[0], aspect='auto', cmap='viridis')
            plt.title("Entrada Original")

            plt.subplot(1, 2, 2)
            plt.imshow(reconstructed[0], aspect='auto', cmap='viridis')
            plt.title("Reconstrucción")

            plt.tight_layout()
            plt.savefig(
                experiment_dir + "prediction.png", dpi=300, bbox_inches="tight"
            )
            plt.close()

            plt.figure(figsize=(8,6), tight_layout=True)

            plt.subplot(2, 1, 1)
            plt.title("Synthetic signal")
            plt.plot(reconstructed[0, 0:400, 0 ])
            plt.subplot(2, 1, 2)
            plt.title("Real signal")
            plt.plot(y_train[0,0:400, 0 ])
            plt.savefig(
                experiment_dir + f"reconstruction_1D_{example}.png", dpi=300, bbox_inches="tight"
            )
            print(experiment_dir + f"reconstruction_1D_{example}.png")
            plt.close()
            

            #Plot multiple inputs samples
            plt.figure(figsize=(12, 6))

            plt.subplot(2, 5, 1)
            plt.imshow(x_sample[0], aspect='auto', cmap='viridis')
            plt.title("Entrada Original")

            plt.subplot(2, 5, 2)
            plt.imshow(x_sample[1], aspect='auto', cmap='viridis')
            plt.title("Entrada Original")

            plt.subplot(2, 5, 3)
            plt.imshow(x_sample[2], aspect='auto', cmap='viridis')
            plt.title("Entrada Original")

            plt.subplot(2, 5, 4)
            plt.imshow(x_sample[3], aspect='auto', cmap='viridis')
            plt.title("Entrada Original")
            
            plt.subplot(2, 5, 5)
            plt.imshow(x_sample[4], aspect='auto', cmap='viridis')
            plt.title("Entrada Original")

            plt.subplot(2, 5, 6)
            plt.imshow(x_sample[5], aspect='auto', cmap='viridis')
            plt.title("Entrada Original")

            plt.tight_layout()
            plt.savefig(
                experiment_dir + "examples_train.png", dpi=300, bbox_inches="tight"
            )
            plt.close()

            

        



