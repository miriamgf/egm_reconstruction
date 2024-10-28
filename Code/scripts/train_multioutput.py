import sys
sys.path.append('../Code')
import matplotlib.pyplot as plt
from config import TrainConfig_1
from config import DataConfig
from tensorflow.keras.optimizers import Adam
print('importing tools')
from tools_.preprocessing_network import *
from tools_.tools import *
from tools_.df_mapping import *
import tensorflow as tf
import os
import scipy
import datetime
import time
from evaluate_function import *
from numpy import *
import pickle
from models.multioutput import MultiOutput
import mlflow
import random
import argparse
tf.random.set_seed(42)
import argparse
from tensorflow.keras import backend as K
import mlflow
import tensorflow as tf
import datetime
import time
from tensorflow.keras.models import load_model

print('end imports')
#Clear GPU
K.clear_session()
tf.keras.backend.clear_session()
tf.compat.v1.reset_default_graph()


'''
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

'''
SNR_em_noise=1
SNR_white_noise=20
patches_oclussion='P1'
experiment_number=0
unfold_code=1

experiment_name=datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '_EXP_' + str(experiment_number)#+ '_' + str(SNR_white_noise)+ '_' + str(patches_oclussion)+ '_' + str(unfold_code)


root_logdir = 'output/logs/'
log_dir = root_logdir + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
data_dir = '/home/profes/miriamgf/tesis/Autoencoders/Data_short/'
torsos_dir = '../../../../Labeled_torsos/'
figs_dir = 'output/figures/'
models_dir = 'output/model/'
dict_var_dir = 'output/variables/'
dict_results_dir = 'output/results/'
experiment_dir='output/experiments/experiments_CINC/'+ experiment_name+'/'



if not os.path.exists(experiment_dir):
    os.makedirs(experiment_dir)
    print("Directory for experiment", experiment_dir, 'created')
else:
    experiment_dir=experiment_dir+'_1'
    print("Existing directory, name changed for avoiding rewriting information to:", experiment_dir)

dic_vars = {}
dict_results = {}

# GPU Configuration
physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs:", len(physical_devices))
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)



start = time.time()

all_torsos_names = []
for subdir, dirs, files in os.walk(torsos_dir):
    for file in files:
        if file.endswith('.mat'):
            all_torsos_names.append(file)


all_model_names = []
directory = data_dir
for subdir, dirs, files in os.walk(directory):
    #print(subdir, directory, files)

    if (subdir != directory):
        model_name = subdir.split("/")[-1]
        all_model_names.append(model_name)

all_model_names = sorted(all_model_names)
print(all_model_names)


mlflow.set_tracking_uri(uri="http://10.110.100.78:5000")
mlflow.autolog()

# Load data

if DataConfig.fs == DataConfig.fs_sub:
    DataConfig.fs = DataConfig.fs_sub

Transfer_model = False  # Transfer learning from sinusoids
sinusoids = False

print('Loading dATA')
#Load data
X_1channel, Y, Y_model, egm_tensor, length_list, AF_models, all_model_names, transfer_matrices = load_data(
    directory = data_dir, 
    data_type='1channelTensor',
    n_classes=DataConfig.n_classes,
    subsampling=True,
    fs_sub=DataConfig.fs_sub,
    norm=False,
    SR=True, SNR=DataConfig.SNR,
    n_batch=TrainConfig_1.batch_size_1,
    sinusoid=sinusoids, 
    SNR_em_noise = SNR_em_noise,
    SNR_white_noise = SNR_em_noise, 
    patches_oclussion=patches_oclussion,
    unfold_code=unfold_code, 
    inference = False)

plt.figure()
plt.plot(X_1channel[0:200, 0, 0], label='bsps')
plt.plot(egm_tensor[0:200, 0], label='egm')
plt.legend()
plt.savefig('output/figures/input_output/before_norm.png')

# Normalize BSPS and EGM
X_1channel = normalize_by_models(X_1channel, Y_model)
egm_tensor = normalize_by_models(egm_tensor, Y_model)
X_1channel = np.nan_to_num(X_1channel, nan=0.0)

plt.figure()
plt.plot(X_1channel[0:200, 0, 0], label='bsps')
plt.plot(egm_tensor[0:200, 0], label='egm')
plt.legend()
plt.savefig('output/figures/input_output/norm.png')


new_items = {'Original_X_1channel': X_1channel, 'Y': Y, 'Y_model': Y_model, 'egm_tensor': egm_tensor,
             'AF_models': AF_models, 'all_model_names': all_model_names, 'transfer_matrices': transfer_matrices,
             'X_1channel_norm': X_1channel}
dic_vars.update(new_items)

# Train/Test/Val Split
random_split = True
print('Splitting...')
x_train, x_test, x_val, train_models, test_models, val_models, AF_models_train, AF_models_test, AF_models_val, BSPM_train, BSPM_test, BSPM_val = train_test_val_split_Autoencoder(
    X_1channel,AF_models, Y_model, all_model_names, random_split=True, train_percentage=0.90, test_percentage=0.2, deterministic = True)


print('TRAIN SHAPE:', x_train.shape, 'models:', train_models)
print('TEST SHAPE:', x_test.shape, 'models:', test_models)
print('VAL SHAPE:', x_val.shape, 'models:', val_models)


new_items = {'x_train_raw': x_train, 'x_test_raw': x_test, 'x_val_raw': x_val, 'train_models': train_models,
             'BSPM_train': BSPM_train, 'BSPM_test': BSPM_test, 'BSPM_val': BSPM_val,
             'test_models': test_models, 'AF_models_train': AF_models_train, 'AF_models_test': AF_models_test,
             'AF_models_val': AF_models_val}
dic_vars.update(new_items)


x_train, x_test, x_val = preprocessing_autoencoder_input(x_train, x_test, x_val, TrainConfig_1.batch_size_1)

new_items = {'x_train': x_train, 'x_test': x_test, 'x_val': x_val}
dic_vars.update(new_items)

y_train, y_test, y_val = preprocessing_y(egm_tensor,Y_model, AF_models, train_models,test_models, val_models, TrainConfig_1.batch_size_1, norm =False)

plt.figure()
plt.plot(x_train[0, :, 0, 0, 0], label = 'bsps')
plt.plot(y_train[0, :,0], label = 'egm')
plt.legend()
plt.savefig('output/figures/input_output/preprocessing.png')

# 1. AUTOENCODER
print('Training model...')

optimizer = Adam(learning_rate=TrainConfig_1.learning_rate_1)

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=models_dir + 'regressor' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
                                                 save_weights_only=True,
                                                 verbose=1)
early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)

# Compilar el modelo
#Configure GPU for paralellism 
strategy = tf.distribute.MirroredStrategy()


if TrainConfig_1.parallelism:
    with strategy.scope():
        model = MultiOutput().assemble_full_model(input_shape=x_train.shape[1:], n_nodes=y_train.shape[-1])
        model.compile(optimizer='adam',
                       loss=['mean_squared_error', 'mean_squared_error'], metrics=['mean_absolute_error'], 
                        loss_weights=[1.0, 5.0])

        print(model.summary())
        history = model.fit(x=x_train, y=[x_train, y_train], batch_size=1, epochs=TrainConfig_1.n_epoch_1,
                                    validation_data=(x_val, [x_val, y_val]),
                                    callbacks=[early_stopping_callback, cp_callback])
else:
    if TrainConfig_1.inference_pretrained_model:
        model = load_model('/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/experiments/experiments_CINC/20240827-112359_EXP_0/model_mo.h5')

    else:
        model = MultiOutput().assemble_full_model(input_shape=x_train.shape[1:], n_nodes=y_train.shape[-1])
        model.compile(optimizer='adam', loss=['mean_squared_error', 'mean_squared_error'], loss_weights=[1.0, 5.0], metrics=['mean_absolute_error'])
        print(model.summary())
        history = model.fit(x=x_train, y=[x_train, y_train], batch_size=1, epochs=TrainConfig_1.n_epoch_1,
                                    validation_data=(x_val, [x_val, y_val]),
                                    callbacks=[early_stopping_callback, cp_callback])

        # summarize history for loss
        plt.figure()
        plt.plot(history.history['val_loss'], label = 'Global loss (Validation)')
        plt.plot(history.history['val_Autoencoder_output_loss'], label = 'Autoencoder loss (Validation)' )
        plt.plot(history.history['val_Regressor_output_loss'], label = 'Regressor loss (Validation)')
        plt.plot(history.history['loss'], label = 'Global loss (Train)')
        plt.plot(history.history['Autoencoder_output_loss'], label = 'Autoencoder loss (Train)' )
        plt.plot(history.history['Regressor_output_loss'], label = 'Regressor loss (Train)')
        plt.legend( loc='upper left')
        plt.title('model loss')
        plt.ylabel('MSE')
        plt.xlabel('epoch')
        plt.title('Training and validation curves ')
        plt.savefig(experiment_dir + 'Learning_curves.png')

plt.show()

# Evaluate
#model = load_model('/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/experiments/20240415-172029/model_mo.h5')
pred_test  = model.predict(x_test, batch_size=1) # x_test=[#batches, batch_size, 12, 32, 1]

if TrainConfig_1.use_generator:
    pred_train = model.predict(data_generator(x_train), steps=x_train.shape[0], batch_size=1)
     #TODO: NOT WORKING
    print('Generator')
else:
    try:
        if TrainConfig_1.parallelism:
            with strategy.scope():
                pred_train  = model.predict(x_train, batch_size=1) #(44, 50, 12, 32, 1)
        else:
            pred_train  = model.predict(x_train, batch_size=1) #(44, 50, 12, 32, 1)
    except:
        if TrainConfig_1.parallelism:
            with strategy.scope():

                x_train=x_train[0:50, :, :, :, :]
                y_train=y_train[0:50, :, :]
                pred_train  = model.predict(x_train, batch_size=1) #(44, 50, 12, 32, 1)
        else:
            try:
                x_train=x_train[0:50, :, :, :, :]
                y_train=y_train[0:50, :, :]
                pred_train  = model.predict(x_train, batch_size=1) #(44, 50, 12, 32, 1)
            except:
                x_train=x_train[0:20, :, :, :, :]
                y_train=y_train[0:20, :, :]
                pred_train  = model.predict(x_train, batch_size=1) #(44, 50, 12, 32, 1)
                
results_autoencoder, results_regressor = evaluate_function_multioutput(x_train, y_train, x_test, y_test,
                                                         pred_train, pred_test, model, batch_size=1)

pred_test_autoencoder, pred_test_egm = pred_test[0], pred_test[1]
pred_train_autoencoder, pred_train_egm = pred_train[0], pred_train[1]

print('Results autoencoder:')
print(results_autoencoder)

print('Results regressor:')
print(results_regressor)

new_items = {'pred_test': pred_test, 'pred_train': pred_train, 'conv_autoencoder': model}
dic_vars.update(new_items)


y_test_flat = reshape_tensor(y_test, n_dim_input=y_test.ndim, n_dim_output=2)
reconstruction_flat_test = reshape_tensor(pred_test_egm, n_dim_input=pred_test_egm.ndim, n_dim_output=2)

x_test_flat = reshape_tensor(x_test, n_dim_input=x_test.ndim, n_dim_output=2)
autoencoder_flat_test = reshape_tensor(pred_test_autoencoder, n_dim_input=pred_test_autoencoder.ndim, n_dim_output=2)
estimate_egms_test = reconstruction_flat_test

# normalize
estimate_egm_test_r=estimate_egms_test
estimate_egms_n=reconstruction_flat_test
estimate_egms_n = normalize_by_models(reconstruction_flat_test,BSPM_test)

pred_test_egm_fl=reshape(pred_test_egm, (pred_test_egm.shape[0]*pred_test_egm.shape[1], pred_test_egm.shape[2]))
y_fl=reshape(y_test, (y_test.shape[0]*y_test.shape[1], y_test.shape[2]))
x_fl=reshape(x_test, (x_test.shape[0]*x_test.shape[1], x_test.shape[2]*x_test.shape[3]*x_test.shape[4] ))


#Reconstruction predictions
for i in range(0, 60):   
    interv=random.randrange(0, len(pred_test_egm_fl)-1, 50)
    node=random.randrange(0, estimate_egms_n.shape[-1], 1)
    normalize_ = True 
    rango=500
    #normalize between -1 and 1
    estimate_signal = pred_test_egm_fl[interv: interv+rango, :]
    estimate_egms_norm_represent = normalize_array(estimate_signal, 1, -1)
    plt.figure(figsize=(15, 3))
    plt.subplot(2, 1, 1)
    plt.plot(estimate_egms_norm_represent[: ,node], label='Estimation Test')
    #plt.plot(estimate_egms_norm[interv: interv+200 ,node], label='Estimation Test')
    plt.plot(y_fl[interv:interv+rango, node], label= 'Test', alpha=0.5)
    #plt.plot(latent_vector_test[200:400, 0, 0, 0], label = 'Latent Vector')
    text='Node {} in second {} to {}'.format(node, interv/fs, interv/fs +rango/fs)
    plt.legend()
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.title(text)
    plt.subplot(2, 1, 2)
    x_signal = x_fl[interv: interv+rango, :]
    x_norm_represent = normalize_array(x_signal, 1, -1)
    plt.plot(x_norm_represent[: ,0:5], label='BSP')
    #plt.plot(estimate_egms_norm[interv: interv+200 ,node], label='Estimation Test')
    plt.legend()
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.title('BSPM')
    plt.savefig(experiment_dir +  'EGM_Reconstructions_' + str(i) + '.png')

    plt.show()

time_instant = random.randint(0, TrainConfig_1.batch_size_1)
batch=random.randrange(0, x_test.shape[0]-2, 1)

#Reconstrauction Autoencoders
for i in range(0, 5):   
    batch=batch+1
    plt.figure(tight_layout=True)
    plt.subplot(3, 1, 1)
    plt.imshow(x_test[batch,0,:,:,0])
    plt.title('X test Autoencoder - batch' + str(batch))
    plt.colorbar(label='Colorbar Label')  # Add a colorbar with a label
    plt.subplot(3, 1, 2)
    plt.imshow(pred_test_autoencoder[batch,0,:,:,0])
    plt.title('Test predictions Autoencoder - batch'+ str(batch))
    plt.colorbar(label='Colorbar Label')  # Add a colorbar with a label
    plt.subplot(3, 1, 3)
    difference= x_test[batch,0,:,:,0] - pred_test_autoencoder[batch,0,:,:,0]
    plt.imshow(difference)
    plt.title('Error')
    plt.colorbar(label='Colorbar Label')  # Add a colorbar with a label
    plt.savefig(experiment_dir +  'Autoencoder_reconstructions.png')
    plt.show()

# 2D EGM plots
plt.figure(layout='tight', figsize=(15, 10))
plt.subplot(1, 3, 1)
plt.imshow(normalize_array(y_fl[0:1000, :].T, 1, 0, 0), cmap='Greys')
plt.title('y test (egm)')
plt.xlabel('time')
plt.ylabel('nodes')
plt.colorbar(orientation="horizontal", pad=0.2)
plt.subplot(1, 3, 2)
plt.imshow(normalize_array(pred_test_egm_fl[0:1000, :].T, 1, 0, 0), cmap='Greys')
plt.title('Reconstruction')
plt.xlabel('time')
plt.ylabel('nodes')
plt.colorbar(orientation="horizontal", pad=0.2)
plt.subplot(1, 3, 3)
dif = normalize_array(y_fl[0:1000, :].T, 1, 0, 0) - normalize_array(pred_test_egm_fl[0:1000, :].T, 1, 0, 0)
dif = abs(dif)
plt.imshow(dif, cmap='Greys')
plt.title('y_test - reconstruction')
plt.xlabel('time')
plt.ylabel('nodes')
plt.colorbar(orientation="horizontal", pad=0.2)
plt.savefig(experiment_dir +  '2D_EGM_predictions.png')
plt.show()

# PSD
nperseg_value = 2 * DataConfig.fs_sub
fig = plt.figure(layout='tight', figsize=(10, 6))
plt.subplot(3, 1, 3)
fs=DataConfig.fs_sub
# EGM reconstructionjjj
for height in range(0, estimate_egms_n.shape[1], 5):
    f, Pxx_den = scipy.signal.welch(estimate_egms_n[:, height], fs, nperseg=nperseg_value,
                                    noverlap=nperseg_value // 2, scaling='density', detrend='linear')
    plt.plot(f, Pxx_den, linewidth=0.5)
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD [V**2/Hz]')
    # plt.ylim([0,0.2])
    titlee = "PSD Welch of EGM signal estimation. node {}".format(height)
    plt.title('EGM signals reconstruction')

plt.subplot(3, 1, 2)
input=reshape(x_test, (x_test.shape[0]*x_test.shape[1], x_test.shape[2]*x_test.shape[3]*x_test.shape[4]))
for height in range(0, input.shape[1], 5):
    f, Pxx_den = scipy.signal.welch(input[:, height], fs, nperseg=nperseg_value,
                                    noverlap=nperseg_value // 2, scaling='density', detrend='linear')
    plt.plot(f, Pxx_den, linewidth=0.5)
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD [V**2/Hz]')
    # plt.ylim([0,0.2])
    titlee = "PSD of BSP (Input). node {}".format(height)
    plt.title('BSP (Input) ')

plt.subplot(3, 1, 1)
for height in range(0, y_fl.shape[1], 5):
    f, Pxx_den = scipy.signal.welch(y_fl[:, height], fs, nperseg=nperseg_value,
                                    noverlap=nperseg_value // 2, scaling='density', detrend='linear')
    plt.plot(f, Pxx_den, linewidth=0.5)
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD [V**2/Hz]')
    # plt.ylim([0,0.2])
    titlee = "PSD of original egm (Input). node {}".format(height)
    plt.title('EGM (Label) ')

fig.suptitle("Welch Periodogram (window size=200 samples)", fontsize=15)
plt.savefig(experiment_dir +  'PSD.png')


# DF mapping: Calculate DF Maps and Phase maps from reconstruction and labels --> Plot 3D in Matlab
if DataConfig.DF_Mapping:
    print('Computing DF Mapping...')
    DF_mapping(y_test, pred_test_egm, BSPM_test, AF_models_test, experiment_dir,norm=True)

# Calculate metrics DTW, RMSE and Correlation BY AF MODELS: Meand and std
# *This metrics are calculated appart because thay are not computed in evaluate_function, (...)
# (...) as they cannot be included in the tensorflow metric callback

#TODO: Arreglar DTW
dtw_array, dtw_array_random = [0,0]#DTW_by_AFModels(AF_models_test, estimate_egms_n, y_test_subsample)
rmse_array = RMSE_by_AFModels(AF_models_test, estimate_egms_n, y_test_flat)
correlation_array, test_models_corr  = correlation_by_AFModels(AF_models_test, estimate_egms_n, y_test_flat)

# Mean and STD of Spearman Correlation, DTW and RMSE
corr_mean = np.mean(correlation_array, axis=1)
corr_std = np.std(correlation_array, axis=1)

dtw_mean = 0 #np.mean(dtw_array, axis=1)
dtw_std = 0 #np.std(dtw_array, axis=1)
dtw_mean_random = 0 #np.mean(dtw_array_random, axis=1)
dtw_std_random = 0 # np.std(dtw_array_random, axis=1)
rmse_mean = np.mean(rmse_array, axis=1)
rmse_std = np.std(rmse_array, axis=1)


new_items = {'corr_mean': corr_mean, 'corr_std': corr_std, 'rmse_mean': rmse_mean, 'rmse_std': rmse_std, 'dtw_mean': dtw_mean,
             'dtw_std': dtw_std, 'dtw_mean_random': dtw_mean_random, 'dtw_std_random': dtw_std_random}
dic_vars.update(new_items)

results = pd.DataFrame(columns=['MSE AE', 'DTW AE', 'MSE Reconstruction', 'TWD Reconstruction'])

# Interpolation for mapping in 3D
estimate_egms_reshaped = reshape(estimate_egms_n, (estimate_egms_n.shape[0], estimate_egms_n.shape[1], 1, 1))
interpol = interpolate_reconstruction(estimate_egms_reshaped)
Test_estimation = reshape(interpol, (interpol.shape[0], interpol.shape[1]))

label_represent = y_test_flat[:, :]
estimate_labels_reshaped = reshape(label_represent, (label_represent.shape[0], label_represent.shape[1], 1, 1))
interpol_label = interpolate_reconstruction(estimate_labels_reshaped)
Label = reshape(interpol_label, (interpol_label.shape[0], interpol_label.shape[1]))

print('Saving variables...')

new_correlation_array = interpolate_fun(correlation_array, len(correlation_array), y_train.shape[2])
new_rmse_array = interpolate_fun(rmse_array, len(rmse_array), y_train.shape[2])

# %%
# Save the model names in train, test and val
test_model_name= [all_model_names[index] for index in AF_models_test]
val_model_name= [all_model_names[index] for index in AF_models_val]
train_model_name= [all_model_names[index] for index in AF_models_train]

mdic = {"reconstruction": Test_estimation, "label": Label}

dic_by_models = array_to_dic_by_models(mdic, test_models, AF_models_test, all_model_names)

variables = {"RMSEmean": rmse_mean, "RMSEstd": rmse_std, 'Corrmean': corr_mean, "corrstd": corr_std,"test_corr_models": test_models_corr,
             'dtwmean': dtw_mean, 'dtwstd': dtw_std,
             'corrbynodes': new_correlation_array, 'rmsenodes': new_rmse_array, 'test_model_name': np.unique(test_model_name),
             "train_model_name": np.unique(train_model_name),
             "val_model_name": np.unique(val_model_name)}

savemat(dict_results_dir + "/reconstruction"+experiment_name+".mat", mdic)
savemat(experiment_dir + "/matlab_3d_map_" + experiment_name + ".mat", mdic)
savemat(experiment_dir + "/reconstructions_by_model_" + experiment_name + ".mat", dic_by_models)


savemat(dict_var_dir + "/variables.mat", variables)

# Write dictionary string representation to text file
file_path = experiment_dir + 'metrics.txt'

with open(file_path, "w") as f:
    for key, value in variables.items():
        f.write(f"{key}: {value}\n")

# Save models
if sinusoids:
    model.save(models_dir + 'sinusoid_pretrained/model_reconstruction.h5'+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    model.save(models_dir +'sinusoid_pretrained/model_autoencoder.h5'+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
else:
    model.save(models_dir +'output/model/model_multioutput.h5')
    model.save(experiment_dir +'model_mo.h5')



model.save('output/model/model_multioutput.h5'+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
# Crear un archivo para guardar el summary

file_name=experiment_dir+ 'model_summary.txt'

with open(file_name, 'w') as f:
    # Redirigir la salida estándar al archivo
    model.summary(print_fn=lambda x: f.write(x + '\n'))

# Save results to csv and export
results_Autoencoder = pd.DataFrame.from_dict(results_autoencoder, orient='index', columns=['Autoencoder'])
results_Reconstruction = pd.DataFrame.from_dict(results_regressor, orient='index', columns=['Reconstruction'])
global_results = pd.concat([results_Autoencoder, results_Reconstruction], axis=1)
global_results.to_csv(dict_var_dir + '/Results_MO')
global_results.to_csv(experiment_dir + '/Results_MO.csv')


global_results.round(3)

# Save dictionaries into pickle and .mat

with open(dict_var_dir + 'variables_MO.pkl', 'wb') as fp:
    pickle.dump(dic_vars, fp)
with open(dict_results_dir + 'dict_results_reconstruction_MO.pkl', 'wb') as fp:
    pickle.dump(results_regressor, fp)
with open(dict_results_dir + 'dict_results_autoencoder_MO.pkl', 'wb') as fp:
    pickle.dump(results_autoencoder, fp)

#savemat(dict_var_dir + "dic_vars.mat", dic_vars) #TODO: cannot be saved to .mat because now is saving a keras model
savemat(dict_results_dir + "dict_results_autoencoder.mat", results_autoencoder)
savemat(dict_results_dir + "dict_results_reconstruction.mat",results_regressor)



     
# %%
end = time.time()
hyperparams = {'lr': TrainConfig_1.learning_rate_1, 'fs': DataConfig.fs_sub, 'epochs': TrainConfig_1.n_epoch_1, 'batch_size': TrainConfig_1.batch_size_1,
                'execution time': (end - start) / 60}

# Specify the file path
file_path = experiment_dir + 'hyperparams.txt'

# Write dictionary string representation to text file
with open(file_path, "w") as f:
    for key, value in hyperparams.items():
        f.write(f"{key}: {value}\n")

print((end - start) / 60, 'Mins of execution')
print("------------- Finished -------------->", experiment_name)













