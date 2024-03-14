from models.autoencoder_model import autoencoder
from models.reconstruction_model import reconstruction
from config import TrainConfig_1
from config import TrainConfig_2
from config import DataConfig
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.optimizers import Adam
from tools_ import freq_phase_analysis as freq_pha
from tools_ import plots
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

tf.random.set_seed(42)

root_logdir = '../output/logs/'
log_dir = root_logdir + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
data_dir = '../../../../Data/'
torsos_dir = '../../../../Labeled_torsos/'
figs_dir = '../output/figures/'
models_dir = '../output/model/'
dict_var_dir = '../output/variables/'
dict_results_dir = '../output/results/'

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
    print(subdir, directory, files)

    if (subdir != directory):
        model_name = subdir.split("/")[-1]
        all_model_names.append(model_name)

all_model_names = sorted(all_model_names)
print(len(all_model_names))

# Load data

if DataConfig.fs == DataConfig.fs_sub:
    DataConfig.fs = DataConfig.fs_sub

Transfer_model = False  # Transfer learning from sinusoids
sinusoids = False


#Load data
X_1channel, Y, Y_model, egm_tensor, length_list, AF_models, all_model_names, transfer_matrices = load_data(
    data_type='1channelTensor', n_classes=DataConfig.n_classes, subsampling=True, fs_sub=DataConfig.fs_sub, norm=False, SR=True, SNR=DataConfig.SNR,
    n_batch=TrainConfig_1.batch_size_1, sinusoid=sinusoids)

# Normalize BSPS
X_1channel = normalize_by_models(X_1channel, Y_model)

new_items = {'Original_X_1channel': X_1channel, 'Y': Y, 'Y_model': Y_model, 'egm_tensor': egm_tensor,
             'AF_models': AF_models, 'all_model_names': all_model_names, 'transfer_matrices': transfer_matrices,
             'X_1channel_norm': X_1channel}
dic_vars.update(new_items)

# Train/Test/Val Split
random_split = True
print('Splitting...')
x_train, x_test, x_val, train_models, test_models, val_models, AF_models_train, AF_models_test, AF_models_val, BSPM_train, BSPM_test, BSPM_val = train_test_val_split_Autoencoder(
    X_1channel,AF_models, Y_model, all_model_names, random_split=True, train_percentage=0.6, test_percentage=0.2)

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

# 1. AUTOENCODER
print('Training Autoencoder...')
encoder, decoder = autoencoder(x_train)
optimizer = Adam(learning_rate=TrainConfig_1.learning_rate_1)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=models_dir + 'regressor' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
                                                 save_weights_only=True,
                                                 verbose=1)
early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

conv_autoencoder = Model(inputs=encoder.input, outputs=decoder(encoder.outputs))
conv_autoencoder.compile(optimizer=optimizer, loss=losses.mean_squared_error,
                         metrics=[losses.mean_squared_error, losses.mean_absolute_error])
history = conv_autoencoder.fit(x_train, x_train, batch_size=TrainConfig_1.batch_size_1, epochs=TrainConfig_1.n_epoch_1,
                               validation_data=(x_val, x_val),
                               callbacks=[early_stopping_callback, cp_callback])

# summarize history for loss
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.title('Training and validation curves ')
plt.savefig(figs_dir + str(DataConfig.fs_sub) + 'Learning_curves_AE.png')
plt.show()

# Evaluate
pred_test = conv_autoencoder.predict(x_test, batch_size=TrainConfig_1.batch_size_1) # x_test=[# batches, batch_size, 12, 32, 1]
pred_train = conv_autoencoder.predict(x_train, batch_size=TrainConfig_1.batch_size_1) #(44, 50, 12, 32, 1)

dict_results_autoencoder = evaluate_function(x_train, x_train, x_test, x_test,pred_train, pred_test,
                                        conv_autoencoder, batch_size=TrainConfig_1.batch_size_1)
print('Results autoencoder:')
print(dict_results_autoencoder)

new_items = {'pred_test': pred_test, 'pred_train': pred_train, 'conv_autoencoder': conv_autoencoder}
dic_vars.update(new_items)



#!!!!!!!! PLOT[2]!!!!!!!!!!!!!!!

# Input --> Latent space

latent_vector_train = encoder.predict(x_train, batch_size=TrainConfig_1.batch_size_1)
latent_vector_test = encoder.predict(x_test, batch_size=TrainConfig_1.batch_size_1)
latent_vector_val = encoder.predict(x_val, batch_size=TrainConfig_1.batch_size_1)

y_train, y_test, y_val, x_train_ls, x_test_ls, x_val_ls, n_nodes = preprocessing_regression_input(latent_vector_train, latent_vector_test, latent_vector_val,
                                                                                         train_models, test_models, val_models,
                                                                                         Y_model, egm_tensor, AF_models, TrainConfig_2.batch_size_2)

new_items = {'x_train_ls': x_train, 'x_test_ls': x_test, 'x_val_ls': x_val, 'y_train_ls':y_train, 'y_test_ls':y_test, 'y_val_ls':y_val,
             'latent_vector_train':latent_vector_train,  'latent_vector_test': latent_vector_test, 'latent_vector_val':latent_vector_val }
dic_vars.update(new_items)

print('Training Recontruction...')
regressor = reconstruction(x_train_ls, y_train)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=models_dir + 'autoencoder' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
                                                 save_weights_only=True,
                                                 verbose=1)
optimizer = Adam(lr=TrainConfig_2.learning_rate_2)
regressor.compile(optimizer=optimizer, loss=losses.mean_squared_error,
                  metrics=[losses.mean_squared_error, losses.mean_absolute_error,
                           tf.keras.metrics.RootMeanSquaredError()])
history = regressor.fit(x_train_ls, y_train, batch_size=TrainConfig_2.batch_size_2, epochs=TrainConfig_2.n_epoch_2, validation_data=(x_val_ls, y_val),
                        callbacks=[early_stopping_callback, cp_callback])

# %%
# summarize history for loss
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(figs_dir + str(DataConfig.fs_sub) + '/Reconstruction_loss.png')
plt.show()


pred_test_egm = regressor.predict(x_test_ls, batch_size=TrainConfig_2.batch_size_2)
pred_train_egm = regressor.predict(x_train_ls, batch_size=TrainConfig_2.batch_size_2)

dict_results_reconstruction = evaluate_function(x_train_ls, y_train, x_test_ls, y_test, pred_train_egm, pred_test_egm,
                                           regressor, batch_size=TrainConfig_2.batch_size_2)

print('dict_results_reconstruction: ')
print(dict_results_reconstruction)

new_items = {'pred_test_egm': pred_test_egm, 'pred_train_egm': pred_train_egm, 'regressor': regressor}
dic_vars.update(new_items)


y_test_flat = reshape_tensor(y_test, n_dim_input=y_test.ndim, n_dim_output=2)
reconstruction_flat_test = reshape_tensor(pred_test_egm, n_dim_input=pred_test_egm.ndim, n_dim_output=2)

y_test_subsample = y_test_flat
estimate_egms_test = reconstruction_flat_test

#TODO: Arreglar normalize_by_models para regression

estimate_egm_test_r=estimate_egms_test
#estimate_egms_n = normalize_by_models(reconstruction_flat_test,BSPM_test)
# Normalize (por muestras)
norm = True
if norm:
    estimate_egms_n = []
    for model in np.unique(BSPM_test):
        # 1. Normalize Reconstruction
        arr_to_norm_estimate = estimate_egm_test_r[
            np.where((BSPM_test == model))]  # select window of signal belonging to model i
        estimate_egms_norm = normalize_array(arr_to_norm_estimate, 1, -1, 0)  # 0: por muestas
        estimate_egms_n.extend(estimate_egms_norm)  # Add to new norm array
    estimate_egms_n = np.array(estimate_egms_n)

# DF mapping: Calculate DF Maps and Phase maps from reconstruction and labels --> Plot 3D in Matlab
if DataConfig.DF_Mapping:
    print('Computing DF Mapping...')
    DF_mapping(y_test, pred_test_egm, BSPM_test, AF_models_test, norm=True)

# Calculate metrics DTW, RMSE and Correlation BY AF MODELS: Meand and std
# *This metrics are calculated appart because thay are not computed in evaluate_function, (...)
# (...) as they cannot be included in the tensorflow metric callback

#TODO: Arreglar DTW
dtw_array, dtw_array_random = [0,0]#DTW_by_AFModels(AF_models_test, estimate_egms_n, y_test_subsample)
rmse_array = RMSE_by_AFModels(AF_models_test, estimate_egms_n, y_test_subsample)
correlation_array = correlation_by_AFModels(AF_models_test, estimate_egms_n, y_test_subsample)

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
interpol = tf.keras.layers.UpSampling2D(size=(4, 1), interpolation='bilinear')(estimate_egms_reshaped)
Test_estimation = reshape(interpol, (interpol.shape[0], interpol.shape[1]))

label_represent = y_test_subsample[:, :]
estimate_labels_reshaped = reshape(label_represent, (label_represent.shape[0], label_represent.shape[1], 1, 1))
interpol_label = tf.keras.layers.UpSampling2D(size=(4, 1), interpolation='bilinear')(estimate_labels_reshaped)
Label = reshape(interpol_label, (interpol_label.shape[0], interpol_label.shape[1]))

print('Saving variables...')

new_correlation_array = interpolate_fun(correlation_array, len(correlation_array), 2048)
new_rmse_array = interpolate_fun(rmse_array, len(rmse_array), 2048)

# %%
# Save the model names in train, test and val
test_model_name= [all_model_names[index] for index in AF_models_test]
val_model_name= [all_model_names[index] for index in AF_models_val]
train_model_name= [all_model_names[index] for index in AF_models_train]

mdic = {"reconstruction": Test_estimation, "label": Label}
variables = {"RMSEmean": rmse_mean, "RMSEstd": rmse_std, 'Corrmean': corr_mean, "corrstd": corr_std,
             'dtwmean': dtw_mean, 'dtwstd': dtw_std,
             'corrbynodes': new_correlation_array, 'rmsenodes': new_rmse_array, 'test_model_name': test_model_name,
             "train_model_name": train_model_name,
             "val_model_name": val_model_name}

savemat(dict_results_dir + "/reconstruction2304.mat", mdic)
savemat(dict_var_dir + "/variables23_04.mat", variables)

# Save models
if sinusoids:
    regressor.save(models_dir + 'sinusoid_pretrained/model_reconstruction.h5'+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    conv_autoencoder.save(models_dir +'sinusoid_pretrained/model_autoencoder.h5'+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
else:
    regressor.save(models_dir +'output/model/model_reconstruction.h5')
    conv_autoencoder.save(models_dir +'output/model/model_autoencoder.h5')

regressor.save('output/model/model_reconstruction.h5'+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
conv_autoencoder.save('output/model/model_autoencoder.h5'+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

# Save results to csv and export
results_Autoencoder = pd.DataFrame.from_dict(dict_results_autoencoder, orient='index', columns=['Autoencoder'])
results_Reconstruction = pd.DataFrame.from_dict(dict_results_reconstruction, orient='index', columns=['Reconstruction'])
global_results = pd.concat([results_Autoencoder, results_Reconstruction], axis=1)
global_results.to_csv(dict_var_dir + '/Results2304')
global_results.round(3)

# Save dictionaries into pickle and .mat

with open(dict_var_dir + 'variables.pkl', 'wb') as fp:
    pickle.dump(dic_vars, fp)
with open(dict_results_dir + 'dict_results_reconstruction.pkl', 'wb') as fp:
    pickle.dump(dict_results_reconstruction, fp)
with open(dict_results_dir + 'dict_results_autoencoder.pkl', 'wb') as fp:
    pickle.dump(dict_results_autoencoder, fp)

#savemat(dict_var_dir + "dic_vars.mat", dic_vars) #TODO: cannot be saved to .mat because now is saving a keras model
savemat(dict_results_dir + "dict_results_autoencoder.mat", dict_results_autoencoder)
savemat(dict_results_dir + "dict_results_reconstruction.mat", dict_results_reconstruction)



# %%
end = time.time()
print((end - start) / 60, 'Mins of execution')

models_dir = '../output/model/'
dict_var_dir = '../output/variables/'
dict_results_dir = '../output/results/'