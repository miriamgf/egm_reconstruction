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




import tools_
import tools_.oclusion
from evaluate_function import evaluate_function_multioutput, evaluate_function_multioutput
from numpy import *
from scipy.io import savemat
import h5py
from config import str_to_bool

import tools_
import tools_.tools
 
from models.gen_vae import Gen_VAE
from models.gen_vae_2d import Gen_VAE_2D
from models.gen_vae_2d_skip import Gen_VAE_2D_Skip
from models.gen_vae_2d_v2 import Gen_VAE_2D_v2

from tools_.df_mapping import *
from tools_.tools import *
from tools_.tools_1 import normalize_array
from tools_.signal_gen import SyntheticDataGenerator


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

print("end imports")
# Clear GPU
K.clear_session()
tf.keras.backend.clear_session()
tf.compat.v1.reset_default_graph()


"""
# parse args
print('parsing')
parser = argparse.ArgumentParser(description="Noise params")
parser.add_argument('--SNR_em_noise', type=int, help='EM noise SNR', required=True)
parser.add_argument('--SNR_white_noise', type=int, help='white noise SNR', required=True)
parser.add_argument('--patches_oclussion', type= str, help='Oclussion patches', required=True)
parser.add_argument('--unfold_code', type=int, help='Unfolding order', required=True)
parser.add_argument('--experiment_number', type=int,  help='number of experiment', required=True)


args = parser.parse_args()


SNR_em_noise = args.SNR_em_noise
SNR_white_noise = args.SNR_white_noise
patches_oclussion = args.patches_oclussion
unfold_code =args.unfold_code
experiment_number = args.experiment_number

print(type(patches_oclussion))
#Run script IDE

"""
#PRECONFIG
params = ParseHiperparams().parse_default_hyperparams()
params["split_mode"] = "stratified"
params["oversampling"] = False
params["discard_classes"] =[0,1, 2, 3, 5]
params["classes_to_oversample"]= []
params["latent_dim"]=250
params['early_stopping_patience']=50
params["beta_warmup_epochs"]= 20
params["select_classes"]= [4]
params["filter_EGM"]=False


try:
    print("parsing")
    parser = argparse.ArgumentParser(description="Noise params")
    parser.add_argument("--algorithm", type=str, help="experiment name", required=False)
    parser.add_argument("--optuna", type=str_to_bool, help="True or False", required=False)
    parser.add_argument("--n_nodes", type=int, help="682, 1024", required=False)
    parser.add_argument("--evaluation", type=str_to_bool, help="evaluation", required=False)

    args = parser.parse_args()
    algorithm = args.algorithm
    optuna = args.optuna
    n_nodes = args.n_nodes
    evaluation = args.evaluation

    if evaluation is not None:
        evaluation = args.evaluation
        print("Evaluation mode activated")
    else:
        evaluation = False

    params["algorithm"]=algorithm
    params["n_nodes_regression"]=n_nodes

    if optuna == "True":
        params["optuna_optimization"] = True

except:
    algorithm = params["algorithm"]
    evaluation=True

print('Params to train: ', params)

SNR_em_noise = None
SNR_white_noise = 100
patches_oclussion = "PT"
experiment_number = 0
unfold_code = 1
experiment_name = algorithm

#experiment_name = f"{experiment_name}_baseline_conv2D_annealing_class_2_3_4"
experiment_name="OMAMI_VAE_baseline_conv2D_v2_develop"


params["experiment_name"] = experiment_name
#experiment_name = f"{experiment_name}_c4/"
experiment_name='pruebas_'
root_logdir = "output/logs/"
log_dir = root_logdir + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
data_dir = "/home/profes/miriamgf/tesis/Autoencoders/Data/"
torsos_dir = "../../../../Labeled_torsos/"
figs_dir = "output/figures/"
models_dir = "output/model/"
dict_var_dir = "output/variables/"
dict_results_dir = "output/results/"
experiment_dir = "output/experiments/synthetic_generation/" + experiment_name 


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
    split_mode="random",
    norm_egm=True,
    shuffle_patient= True, 
)()

class_complexity_list_train=class_complexity_list_train[:, 0]
class_complexity_list_test=class_complexity_list_test[:, 0]
class_complexity_list_val=class_complexity_list_val[:, 0]


params["algorithm"] = "gen_VAE_2D_v2"
print("Algorithm selected:", params["algorithm"])

if params["algorithm"] == "gen_VAE_3D":

    y_train=y_train.reshape(y_train.shape[0], y_train.shape[1], 32, 64)
    y_val=y_val.reshape(y_val.shape[0], y_val.shape[1], 32, 64)
    y_test=y_test.reshape(y_test.shape[0], y_test.shape[1], 32, 64)

evaluation=True
if not evaluation:
    params["n_epochs"] = 20

    #fair_train = False

    model, history = TrainModelGen(
        params, y_train, y_test, y_val, y_train, y_test, y_val, models_dir, experiment_dir
    )()

    # Guardar parámetros como JSON en experiment_dir
    params_path = os.path.join(experiment_dir, "params.json")
    with open(params_path, "w") as f:
        json.dump(params, f, indent=4)  

    print(f"Parámetros guardados en {params_path}")



if evaluation:
    vae = Gen_VAE_2D(
        params=params,
        input_shape_=y_test.shape[1:], 
        n_nodes=2048,
        latent_dim=params["latent_dim"],
        tensorboard_logs=experiment_dir + "tb_logs/"
    )

    model_name="/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/experiments/synthetic_generation/OMAMI_VAE_baseline_conv2D_v2_develop_c4/"
    try:
        vae.model.load_weights(model_name + "model_weights.h5")
    except:
        vae = Gen_VAE_2D_v2(params, y_train.shape[1:], n_nodes=2048,
                      tensorboard_logs=experiment_dir + "tb_logs/", latent_dim=params["latent_dim"])
        vae.build(input_shape=(None,) + y_train.shape[1:])   # create variables
        vae.load_weights(experiment_dir + "/model_weights.h5")

    x_real= y_train[:, :, :]
    batch_classes=class_complexity_list_train.astype(np.int32)

    class_labels = {
    4: "Sinusal",
    3: "AF"}

    # Latent exploration 
    try:
        z, z_mean_train, _ = vae.build_encoder_module(x_real, vae.input_shape_)
    except:
        z, z_mean_train, z_log_var = vae.encoder(x_real, training=False)


    pca = PCA(n_components=2)
    z_proj = pca.fit_transform(z_mean_train)  

    dataset_generator=False

    if dataset_generator:
        generator = SyntheticDataGenerator(
            vae_model=vae,                      # modelo VAE 
            latent_dim=params["latent_dim"],    # Dimensión del espacio latente 
            save_dir="/home/pdi/miriamgf/tesis/Autoencoders/Data_generated",
            params=params       
        )

        best_signals = generator.generate_and_select(
            real_signals=x_real,                # Tus señales reales 
            z_mean_train=z_mean_train,          # Los z_mean del set de entrenamiento
            num_generated=100,                 # Número de señales sintéticas a generar 
            num_selected=25                    # Cuántas quedarte al final
        )


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

    # Samplear del espacio latente
    random_sampling = True
    guided_sampling = True
    reconstruction = True
    interpol = True
    manifold = True

    if random_sampling: 

        z = tf.random.normal((1, vae.latent_dim))  # 1 muestras aleatorias
        try:
            synthetic = vae.decode_from_latent(z)
        except:
            synthetic = vae.decoder(z, training=False)

        synthetic=np.array(synthetic)
        synthetic = synthetic.squeeze()
        synthetic_norm = np.zeros(synthetic.shape)
        for i in range(synthetic.shape[1]):
            synthetic_norm[ :, i] = normalize_array(synthetic[ :, i], axis_n=0, high=1.0, low=-1.0)
        
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.title("Synthetic signal")
        plt.imshow(synthetic_norm[:, : ], aspect="auto")
        plt.subplot(2, 1, 2)
        plt.title("Real signal")
        plt.imshow(y_train[0, :, :], aspect="auto")
        plt.savefig(
            experiment_dir + "synthetic_signal_random_sampling.png", dpi=300, bbox_inches="tight"
        )
        print(experiment_dir + "synthetic_signal_random_sampling.png")
        plt.close()

        for node in range(0, 2047, 100):

            plt.figure()
            plt.subplot(2, 1, 1)
            plt.title("Synthetic signal")
            plt.plot(synthetic[0:400, node])
            plt.subplot(2, 1, 2)
            plt.title("Real signal")
            plt.plot(y_train[0,0:400, node ])
            plt.savefig(
                experiment_dir + f"random_sampling_1D_{node}.png", dpi=300, bbox_inches="tight"
            )
            print(experiment_dir + f"random_sampling_1D_{node}.png")
            plt.close()

        plt.figure()
        plt.hist(z.numpy().flatten(), bins=100)
        plt.title("Distribución de z_mean")
        plt.savefig(
            experiment_dir + "histogram_rand.png", dpi=300, bbox_inches="tight"
        )
        plt.close()



    if guided_sampling:

        try:
            z, z_mean_train, _ = vae.build_encoder_module(x_real, vae.input_shape_)
        except:
            z, z_mean_train, z_log_var = vae.encoder(x_real, training=False)

        mu = np.mean(z_mean_train, axis=0)
        sigma = np.std(z_mean_train, axis=0)

        z = np.random.normal(loc=mu, scale=sigma, size=(10, params["latent_dim"]))

        try:
            synthetica = vae.decode_from_latent(z)
        except:
            synthetica = vae.decoder(z, training=False)

        for example in range(0,10):
            one_synthetic = synthetica[example, :, :]

            synthetic=np.array(one_synthetic)
            synthetic = synthetic.squeeze()
            synthetic_norm = np.zeros(synthetic.shape)
            for i in range(synthetic.shape[1]):
                synthetic_norm[ :, i] = normalize_array(synthetic[ :, i], axis_n=0, high=1.0, low=-1.0)

            plt.figure(tight_layout=True)
            plt.subplot(2, 1, 1)
            plt.title("Synthetic signal")
            plt.imshow(synthetic_norm[:, : ], aspect="auto")
            plt.subplot(2, 1, 2)
            plt.title("Real signal")
            plt.xlabel("Number of nodes of EGM signals")
            plt.ylabel("Number of time samples")
            plt.imshow(y_train[0, :, :], aspect="auto")
            plt.savefig(
                experiment_dir + f"synthetic_signal_guided_sampling_{example}.png", dpi=300, bbox_inches="tight"
            )
            print(experiment_dir + f"synthetic_signal_guided_sampling_{example}.png")
            plt.close()

            for node in range(0, 2047, 100):

                plt.figure()
                plt.subplot(2, 1, 1)
                plt.title("Synthetic signal")
                plt.plot(synthetic[0:400, node])
                plt.subplot(2, 1, 2)
                plt.title("Real signal")
                plt.plot(y_train[0,0:400, node ])
                plt.savefig(
                    experiment_dir + f"guided_sampling_1D_{example}_{node}.png", dpi=300, bbox_inches="tight"
                )
                print(experiment_dir + f"guided_sampling_1D_{example}_{node}.png")
                plt.close()

        plt.figure()
        plt.hist(z_mean_train.numpy().flatten(), bins=100)
        plt.title("Distribución de z_mean")
        plt.savefig(
            experiment_dir + "histogram_guided.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        plt.figure(figsize=(8, 6))
        plt.scatter(z_mean_train[:, 0], z_mean_train[:, 1], alpha=0.5)
        plt.title("Distribución del espacio latente (2D)")
        plt.xlabel("z1")
        plt.ylabel("z2")
        plt.grid(True)
        plt.savefig(experiment_dir + "latent_space_guided.png", dpi=300, bbox_inches="tight")

        from sklearn.decomposition import PCA
        import matplotlib.pyplot as plt


        # PCA
        pca = PCA(n_components=2)
        z_proj = pca.fit_transform(z_mean_train)  # (333, 2)

        # Visualización con color por clase
        plt.figure(figsize=(6, 6))
        scatter = plt.scatter(z_proj[:, 0], z_proj[:, 1], c=batch_classes, cmap='viridis', alpha=0.8)
        plt.title("PCA del espacio latente (z_mean)")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.grid(True)

    if reconstruction == False:

        x_real_one_batch= np.expand_dims(x_real[0, :, :], axis=0)

        try:
            z, z_mean_train, _ = vae.build_encoder_module(x_real, vae.input_shape_)
        except:
            z, z_mean_train, _ = vae.encoder(x_real, training=False)
        try:
            synthetic = vae.decode_from_latent(z)
        except:
            synthetic = vae.decoder(z, training=False)
        synthetic=np.array(synthetic)
        synthetic = synthetic.squeeze()
        synthetic_norm = np.zeros(synthetic.shape)
        for i in range(synthetic.shape[1]):
            synthetic_norm[ :, i] = normalize_array(synthetic[ :, i], axis_n=0, high=1.0, low=-1.0)

        plt.figure()
        plt.subplot(2, 1, 1)
        plt.title("Synthetic signal")
        plt.imshow(synthetic_norm[:, : ], aspect="auto")
        plt.subplot(2, 1, 2)
        plt.title("Real signal")
        plt.imshow(y_train[0, :, :], aspect="auto")
        plt.savefig(
            experiment_dir + "synthetic_signal_reconstruction.png", dpi=300, bbox_inches="tight"
        )
        print(experiment_dir + "synthetic_signal_reconstruction.png")
        plt.close()

        #signal
        plt.figure()
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

        plt.figure()
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

    if interpol == False:

        x_real_1= np.expand_dims(x_real[0, :, :], axis=0)
        x_real_2= np.expand_dims(x_real[9, :, :], axis=0)

        try:
            z1, z_mean_train1, _ = vae.build_encoder_module(x_real_1, vae.input_shape_)
            z2, z_mean_train2, _ = vae.build_encoder_module(x_real_2, vae.input_shape_)

        except:
            z1, z_mean_train1, _ = vae.encoder(x_real_1, training=False)
            z2, z_mean_train2, _ = vae.encoder(x_real_2, training=False)
        
        alpha=0.1

        z_interp = (1-alpha) * z1 + alpha * z2


        try:
            synthetic = vae.decode_from_latent(z_interp)
        except:
            synthetic = vae.decoder(z_interp, training=False)

        synthetic=np.array(synthetic)
        synthetic = synthetic.squeeze()
        synthetic_norm = np.zeros(synthetic.shape)
        for i in range(synthetic.shape[1]):
            synthetic_norm[ :, i] = normalize_array(synthetic[ :, i], axis_n=0, high=1.0, low=-1.0)

        plt.figure()
        plt.subplot(2, 1, 1)
        plt.title("Synthetic signal")
        plt.imshow(synthetic_norm[:, : ], aspect="auto")
        plt.subplot(2, 1, 2)
        plt.title("Real signal")
        plt.imshow(y_train[0, :, :], aspect="auto")
        plt.savefig(
            experiment_dir + "synthetic_signal_interp.png", dpi=300, bbox_inches="tight"
        )
        print(experiment_dir + "synthetic_signal_interp.png")
        plt.close()

        #signal
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.title("Synthetic signal")
        plt.plot(synthetic[0:400, 0 ])
        plt.subplot(2, 1, 2)
        plt.title("Real signal")
        plt.plot(y_train[0,0:400, 0 ])
        plt.savefig(
            experiment_dir + "interp_1D.png", dpi=300, bbox_inches="tight"
        )
        print(experiment_dir + "interp_1D.png")
        plt.close()
    
    if manifold:

        try:
            z, z_mean_train, _ = vae.build_encoder_module(x_real, vae.input_shape_)
        except:
            z, z_mean_train, _ = vae.encoder(x_real, training=False)

        idx = np.random.choice(len(z_mean_train), size=10)
        z_mean_train = np.array(z_mean_train)
        z_base = z_mean_train[idx]
        z_sample = z_base + np.random.normal(scale=0.01, size=z_base.shape)

        try:
            synthetica = vae.decode_from_latent(z_sample)
        except:
            synthetica = vae.decoder(z_sample, training=False)
        for example in range(0,10):

            one_synthetic = synthetica[example, :, :]

            synthetic=np.array(one_synthetic)
            synthetic = synthetic.squeeze()
            synthetic_norm = np.zeros(synthetic.shape)
            for i in range(synthetic.shape[1]):
                synthetic_norm[ :, i] = normalize_array(synthetic[ :, i], axis_n=0, high=1.0, low=-1.0)

            plt.figure()
            plt.subplot(2, 1, 1)
            plt.title("Synthetic signal")
            plt.imshow(synthetic_norm[:, : ], aspect="auto")
            plt.subplot(2, 1, 2)
            plt.title("Real signal")
            im1=plt.imshow(y_train[0, :, :], aspect="auto")
            plt.colorbar(im1, orientation='vertical', fraction=0.046, pad=0.04)  # ← Añade barra lateral

            plt.savefig(
                experiment_dir + f"synthetic_signal_guided_sampling_{example}.png", dpi=300, bbox_inches="tight"
            )
            print(experiment_dir + f"synthetic_signal_guided_sampling_{example}.png")
            plt.close()

            #signal
            plt.figure()
            plt.subplot(2, 1, 1)
            plt.title("Synthetic signal")
            plt.plot(synthetic[0:400, 0 ])
            plt.subplot(2, 1, 2)
            plt.title("Real signal")
            plt.plot(y_train[0,0:400, 0 ])
            plt.savefig(
                experiment_dir + f"manifold_1D_{example}.png", dpi=300, bbox_inches="tight"
            )
            print(experiment_dir + f"manifold_1D_{example}.png")
            plt.close()

    if reconstruction:

            x_sample = y_train[:1]  # Una muestra
            _, z_mean, _ = vae.encoder(x_sample, training=False)
            reconstructed = vae.decoder(z_mean, training=False).numpy()

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

            plt.figure()
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

            # Interpolación en el espacio latente
            z_1 = tf.random.normal((1, vae.latent_dim))
            z_2 = tf.random.normal((1, vae.latent_dim))

            alphas = np.linspace(0, 1, 10)
            interpolations = [(1 - alpha) * z_1 + alpha * z_2 for alpha in alphas]
            generated = [vae.decoder(z) for z in interpolations]

            plt.figure()
            # Visualiza
            for i, sample in enumerate(generated):
                plt.imshow(sample[0], aspect='auto')
                plt.title(f"Alpha {alphas[i]:.2f}")
                plt.savefig(
                experiment_dir + f"interpolations_{i}.png", dpi=300, bbox_inches="tight"
            )
                plt.show()

            #Evaluation
            reconstructed = vae.model.predict(y_test)
            mse = tf.reduce_mean(tf.keras.losses.mean_squared_error(y_test, reconstructed)).numpy()
            print("Mean Squared Error (MSE):", mse)

            # Encode test set
            try:
                z, z_mean_train, _ = vae.build_encoder_module(x_real, vae.input_shape_)
            except:
                z, z_mean_train, _ = vae.encoder(x_real, training=False)

            # Clip log var for numerical stability
            logvar = tf.clip_by_value(z_log_var, -10.0, 10.0)

            # Compute KL loss per sample
            kl_loss_per_sample = -0.5 * tf.reduce_sum(1 + logvar - tf.square(z_mean_train) - tf.exp(logvar), axis=1)

            # Mean KL over all samples
            kl_loss_mean = tf.reduce_mean(kl_loss_per_sample).numpy()

            print(f"KL loss (test set): {kl_loss_mean:.6f}")

