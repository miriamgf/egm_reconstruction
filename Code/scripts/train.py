import sys
import os
import datetime
from ..models import autoencoder_model

from fastdtw import fastdtw
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from tensorflow.keras.optimizers import Adam
from ..tools_ import freq_phase_analysis as freq_pha
from ..Plots import *
import tensorflow as tf
from ..tools_ import tools as tools
from tools import *


root_logdir = '../Logs/'
data_dir = '../../../Data/'
figs_dir = 'Figs/'
models_dir = '../Models/'

# Tensorboard logs name generator
def get_run_logdir():
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)

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

transfer_matrices = []
for torso in all_torsos_names:
    BSP_pos = scipy.io.loadmat(torsos_dir + torso).get('torso')

all_model_names = []
directory = data_dir
for subdir, dirs, files in os.walk(directory):
    print(subdir, directory, files)

    if (subdir != directory):
        model_name = subdir.split("/")[-1]
        all_model_names.append(model_name)

all_model_names = sorted(all_model_names)
print(all_model_names)
print(len(all_model_names), 'Models')

# Load data
n_classes = 3  # 1: Rotor/no rotor ; 2: RA/LA/No rotor (2 classes) ; 3: 7 regions (3 classes) + no rotor (8 classes)
n_batch = 50  # Batch size
fs_sub = 50
fs = fs_sub
Transfer_model = False  # Transfer learning from sinusoids
sinusoids = False
n_epoch = 2
# Folders ECGI Summit
# Folders ECGI Summit
try:
    os.mkdir('output/figures/'+ str(fs_sub))
except:
    pass

try:
    os.mkdir('output/figures/'+ str(fs_sub))
except:
    pass

try:
    os.mkdir('output/figures/'+ str(fs_sub))
except:
    pass


X_1channel, Y, Y_model, egm_tensor, length_list, AF_models, all_model_names, transfer_matrices = load_data(
    data_type='1channelTensor', n_classes=n_classes, subsampling=True, fs_sub=fs_sub, norm=False, SR=True, SNR=20,
    n_batch=n_batch, sinusoid=sinusoids)  # , n_batch=n_batch)

# Normalize BSPS
X_1channel = normalize_by_models(X_1channel, Y_model)

# Train/Test/Val Split
random_split = True
print('Splitting...')
x_train, x_test, x_val, train_models, test_models, val_models, AF_models_train, AF_models_test, AF_models_val, BSPM_train, BSPM_test, BSPM_val = train_test_val_split_Autoencoder(
    X_1channel,
    AF_models, Y_model, all_model_names, random_split=True, train_percentage=0.6, test_percentage=0.2)

print('TRAIN SHAPE:', x_train.shape, 'models:', train_models)
print('TEST SHAPE:', x_test.shape, 'models:', test_models)
print('VAL SHAPE:', x_val.shape, 'models:', val_models)

# Save the model names in train, test and val
test_model_name = [all_model_names[index] for index in AF_models_test]
val_model_name = [all_model_names[index] for index in AF_models_val]
train_model_name = [all_model_names[index] for index in AF_models_train]

'''
# PLOT:  Batch generation
x_train_batch = reshape(x_train, (int(len(x_train) / n_batch), n_batch, x_train.shape[1], x_train.shape[2], 1))
x_test_batch = reshape(x_test, (int(len(x_test) / n_batch), n_batch, x_test.shape[1], x_test.shape[2], 1))
x_val_batch = reshape(x_val, (int(len(x_val) / n_batch), n_batch, x_val.shape[1], x_val.shape[2], 1))

# Reshape AF_models to compute corresponding AF model to each sample 
x_train_batch_AF = reshape(AF_models_train, (int(len(AF_models_train) / n_batch), n_batch))
x_test_batch_AF = reshape(AF_models_test, (int(len(AF_models_test) / n_batch), n_batch))
x_val_batch_AF = reshape(AF_models_val, (int(len(AF_models_val) / n_batch), n_batch))

# Plot random batches
a = randint(0, x_train_batch.shape[0] - 1)  # Random batch


plt.figure(layout='tight', figsize=(10, 10))
for i in range(0, n_batch):
    plt.subplot(10, 10, i+1 )
    plt.imshow(x_train_batch[a,i, :, :, 0])
    plt.title(str(i) + '- AF model' + str(x_train_batch_AF[a, i]))
plt.suptitle('Train Batch'+str(a))
plt.savefig('output/figures/'+ str(fs_sub) + '/Batch1.png')
plt.show()


a=randint(0, x_test_batch.shape[0]-1) #Random batch
plt.figure(layout='tight', figsize=(10, 10))
for i in range(0, n_batch):
    plt.subplot(10, 10, i+1 )
    plt.imshow(x_test_batch[a,i, :, :, 0])
    plt.title('AF model' + str(x_test_batch_AF[a, i]))
plt.suptitle('Test Batch'+str(a))
plt.savefig('output/figures/'+ str(fs_sub) + '/Batch2.png')
plt.show()
'''
plotting_before_AE(X_1channel, x_train, x_test, x_val, Y_model, AF_models)

# Reshape and batch_generation to fit Conv (Add 1 dimension)
print(x_train.shape, x_test.shape, x_val.shape)
x_train = reshape(x_train, (int(len(x_train) / n_batch), n_batch, x_train.shape[1], x_train.shape[2], 1))
x_test = reshape(x_test, (int(len(x_test) / n_batch), n_batch, x_test.shape[1], x_test.shape[2], 1))
x_val = reshape(x_val, (int(len(x_val) / n_batch), n_batch, x_val.shape[1], x_val.shape[2], 1))
print(x_train.shape, x_test.shape, x_val.shape)

# 1. AUTOENCODER
print('Training Autoencoder...')


encoder, decoder = autoencoder(x_train)

optimizer = Adam(lr=0.00001)
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

conv_autoencoder = Model(inputs=encoder.input, outputs=decoder(encoder.outputs))
conv_autoencoder.compile(optimizer='adam', loss=losses.mean_squared_error,
                         metrics=[losses.mean_squared_error, losses.mean_absolute_error])
history = conv_autoencoder.fit(x_train, x_train, batch_size=1, epochs=n_epoch, validation_data=(x_val, x_val),
                               callbacks=[callback, tensorboard_callback])

decoded_imgs = conv_autoencoder.predict(x_test, batch_size=1)
decoded_imgs_train = conv_autoencoder.predict(x_train, batch_size=1)

# Reshape to remove the 1 at the end
print(decoded_imgs.shape, x_test.shape)

decoded_imgs = reshape(decoded_imgs,
                       (len(decoded_imgs), decoded_imgs.shape[1], decoded_imgs.shape[2], decoded_imgs.shape[3]))
decoded_imgs_train = reshape(decoded_imgs_train, (
len(decoded_imgs_train), decoded_imgs_train.shape[1], decoded_imgs_train.shape[2], decoded_imgs_train.shape[3]))

# Evaluate
results_test = conv_autoencoder.evaluate(x_test, batch_size=50)
results_train = conv_autoencoder.evaluate(x_train, batch_size=1)

# Save
mse_autoencoder_test, mae_autoencoder_test = results_test[0], results_test[2]
mse_autoencoder_train, mae_autoencoder_train = results_train[0], results_train[2]

print(results_test)

# Reshape for dtw
decoded_flat = reshape(decoded_imgs,
                       (decoded_imgs.shape[0] * decoded_imgs.shape[1], decoded_imgs.shape[2] * decoded_imgs.shape[3]))
decoded_flat_train = reshape(decoded_imgs_train, (
decoded_imgs_train.shape[0] * decoded_imgs_train.shape[1], decoded_imgs_train.shape[2] * decoded_imgs_train.shape[3]))
x_test_flat = reshape(x_test, (x_test.shape[0] * x_test.shape[1], x_test.shape[2] * x_test.shape[3]))
x_train_flat = reshape(x_train, (x_train.shape[0] * x_train.shape[1], x_train.shape[2] * x_train.shape[3]))

# MAE and MSE
#mse_autoencoder_test = mean_squared_error(x_test_flat, decoded_flat)
#mse_autoencoder_train = mean_squared_error(x_train_flat, decoded_flat_train)


# DTW !

dtw_test, path = fastdtw(decoded_flat, x_test_flat)
dtw_train, path = fastdtw(decoded_flat_train, x_train_flat)

x_test = reshape(x_test, (len(x_test), x_test.shape[1], x_test.shape[2], x_test.shape[3]))
x_train = reshape(x_train, (len(x_train), x_train.shape[1], x_train.shape[2], x_train.shape[3]))

# %%
results_autoencoder = {'mse test': mse_autoencoder_test, 'mse train': mse_autoencoder_train,
                       'mae test': mae_autoencoder_test, 'mae train': mae_autoencoder_train,
                       'dtw_test': dtw_test, 'dtw_train': dtw_train}

print('results autoencoder: ', results_autoencoder)

# summarize history for loss
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.title('Training and validation curves ')
plt.savefig('output/figures/' + str(fs_sub) + 'Learning_curves_AE.png')
plt.show()

#
time_instant = random.randint(0, 49)
batch = random.randint(0, x_test.shape[0])

print(batch, time_instant)

plt.figure(figsize=(5, 5), layout='tight')
plt.subplot(3, 1, 1)
# image1=define_image(time_instant,x_test )
image1 = x_test[batch, time_instant, ::]
min_val, max_val = np.amin(image1), np.amax(image1)
plt.imshow(image1, vmin=min_val, vmax=max_val)  # , cmap='jet')
plt.colorbar()
plt.title('Original')
plt.subplot(3, 1, 2)
# image2=define_image(time_instant,decoded_imgs)
image2 = decoded_imgs[batch, time_instant, ::]
plt.imshow(image2, vmin=min_val, vmax=max_val)  # , cmap='jet')
plt.colorbar()
plt.title('Reconstructed')
plt.subplot(3, 1, 3)
plt.imshow(image2 - image1, vmin=min_val, vmax=max_val)
plt.title('Error (Reconstructed-Original)')
plt.colorbar()
plt.savefig('output/figures/' + str(fs_sub) + '/AE_reconstruction.png')
plt.show()

# PLOTS 50 samples --> 1 second
batch = random.randint(0, x_test.shape[0])
'''
# Random signal visualizer: in each execution plots the first second of the BSPS in random nodes
plt.figure(figsize=(10, 3))
a = randint(0, decoded_imgs.shape[2] - 1)
b = randint(0, decoded_imgs.shape[3] - 1)
# s=randint(0, len(decoded_imgs)-fs-1 )

title = "50 samples from BSP from nodes {} x {} in the model {}".format(a, b, x_test_batch_AF[batch, 0])
plt.plot(x_test[batch, :, a, b], label=('Input'))
plt.plot(decoded_imgs[batch, :, a, b], label=('Reconstruction'))
plt.fill_between(np.arange(n_batch), decoded_imgs[batch, :, a, b], x_test[batch, :, a, b], color='lightgreen')
plt.ylabel('Amplitude')
plt.xlabel('Samples')
plt.legend()
plt.title(title)
plt.savefig('output/figures/' + str(fs_sub) + '/AE_Recontruction2.png')

plt.show()
'''
# %%
latent_vector = encoder.predict(x_test, batch_size=1)

# create function to center data
center_function = lambda x: x - x.mean(axis=0)

# Center latent space
latent_vector = center_function(latent_vector)


# 3.3 Plot latent space frames in random time instant


time_instant = randint(0, 49)
batch = random.randint(0, x_test.shape[0])

plt.figure(figsize=(6, 6), layout='tight')
title = "Filters from Latent Space in time instant {} ".format(time_instant)
for i in range(0, 12):
    plt.subplot(6, 2, i + 1)
    # plt.plot(6, 2, i)
    plt.imshow(latent_vector[batch, time_instant, :, :, i])  # , cmap='jet')
    plt.colorbar()
plt.suptitle(title)
plt.savefig('output/figures/' + str(fs_sub) + '/Latentspace1.png')

plt.show()


plt.figure()
plt.plot(x_test[batch, :, 1, 1], label='Input')  # , cmap='jet')

for i in range(0, 12):
    # plt.plot(6, 2, i)
    plt.plot(latent_vector[batch, :, 1, 1, i], alpha=0.5, label=i)  # , cmap='jet')
    plt.title('Feature vector in time (50 samples) of 12 filters')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.savefig('output/figures/' + str(fs_sub) + '/Latentspace2.png')

plt.show()


# Flatten
x_train_fl = reshape(x_train, (x_train.shape[0] * x_train.shape[1], x_train.shape[2], x_train.shape[3]))
x_test_fl = reshape(x_test, (x_test.shape[0] * x_test.shape[1], x_test.shape[2], x_test.shape[3]))
x_val_fl = reshape(x_val, (x_val.shape[0] * x_val.shape[1], x_val.shape[2], x_val.shape[3]))
decoded_imgs_fl = reshape(decoded_imgs,
                          (decoded_imgs.shape[0] * decoded_imgs.shape[1], decoded_imgs.shape[2], decoded_imgs.shape[3]))
#latent_vector = reshape(latent_vector, (
#latent_vector.shape[0] * latent_vector.shape[1], latent_vector.shape[2], latent_vector.shape[3],
#latent_vector.shape[4]))

# Plot of reconstruction (orange) over the context of 14 seconds of random input signal

# PLOT INPUT VS OUTPUT
window_length_test = 500  # samples
window_length_subsegment = 200
t_vector = np.array(np.linspace(0, window_length_test / fs, window_length_test))
t_vector_reshaped = t_vector.reshape(t_vector.shape[0], 1, 1)

# Random big window to show x_test
randomonset_test = random.randrange(len(x_test_fl) - window_length_test)
random_window_test = [randomonset_test, randomonset_test + window_length_test]

x_test_cut = x_test_fl[random_window_test[0]:random_window_test[1], 1, 1]
decoded_imgs_cut = decoded_imgs_fl[random_window_test[0]:random_window_test[1], 1, 1]

# smaller window to show subsegment inside x_test
randomonset = random.randrange(len(x_test_cut) - window_length_subsegment)
random_window_subsegment = [randomonset, randomonset + window_length_subsegment]
print(decoded_imgs_cut.size, 'total length', random_window_subsegment, 'subwindow size')
copy_decoded_imgs_cut = decoded_imgs_cut
decoded_imgs_cut[:random_window_subsegment[0]] = None
decoded_imgs_cut[random_window_subsegment[1]:] = None

plt.figure(figsize=(10, 2))
plt.plot(t_vector, x_test_cut, alpha=0.5, label=('Test'))
plt.plot(t_vector, decoded_imgs_cut, label=('Reconstruction'))
plt.xlabel('Time(s)')
plt.ylabel('mV')
plt.title('14 seconds of Test BSPS. Over it, 2 seconds of BSPS reconstruction')
plt.savefig('output/figures/' + str(fs_sub) + '/BSPS_AE.png')
plt.show()

# PSD Welch of input, output and Latent Space

plot_Welch_periodogram(x_test_fl, latent_vector, decoded_imgs_fl, fs=fs)
plt.savefig('output/figures/' + str(fs_sub) + '/Periodogram_AE.png')

# Input --> Latent space
# %%
latent_vector_train = encoder.predict(x_train, batch_size=1)
latent_vector_test = encoder.predict(x_test, batch_size=1)
latent_vector_val = encoder.predict(x_val, batch_size=1)

latent_space_n, egm_tensor_n = preprocess_latent_space(latent_vector_train, latent_vector_test, latent_vector_val,
                                                       Y_model, egm_tensor, dimension=5)

# Split egm_tensor
if random_split:
    x_train = latent_space_n[np.in1d(AF_models, train_models)]
    x_test = latent_space_n[np.in1d(AF_models, test_models)]
    x_val = latent_space_n[np.in1d(AF_models, val_models)]
else:
    x_train = latent_space_n[np.where((Y_model >= 1) & (Y_model <= 200))]
    x_test = latent_space_n[np.where((Y_model > 180) & (Y_model <= 244))]
    x_val = latent_space_n[np.where((Y_model > 244) & (Y_model <= 286))]

# Split EGM (Label)
if random_split:
    y_train = egm_tensor_n[np.in1d(AF_models, train_models)]
    y_test = egm_tensor_n[np.in1d(AF_models, test_models)]
    y_val = egm_tensor_n[np.in1d(AF_models, val_models)]

else:

    y_train = egm_tensor_n[np.where((Y_model >= 1) & (Y_model <= 200))]
    y_test = egm_tensor_n[np.where((Y_model > 180) & (Y_model <= 244))]
    y_val = egm_tensor_n[np.where((Y_model > 244) & (Y_model <= 286))]

# %% Subsample EGM nodes

y_train_subsample = y_train[:, 0:2048:4]
y_test_subsample = y_test[:, 0:2048:4]
y_val_subsample = y_val[:, 0:2048:4]

n_nodes = y_train_subsample.shape[1]

# Batch generation
x_train_ls = reshape(x_train,
                     (int(len(x_train) / n_batch), n_batch, x_train.shape[1], x_train.shape[2], x_train.shape[3]))
x_test_ls = reshape(x_test, (int(len(x_test) / n_batch), n_batch, x_test.shape[1], x_test.shape[2], x_test.shape[3]))
x_val_ls = reshape(x_val, (int(len(x_val) / n_batch), n_batch, x_val.shape[1], x_val.shape[2], x_val.shape[3]))

y_train = reshape(y_train_subsample, (int(len(y_train_subsample) / n_batch), n_batch, y_train_subsample.shape[1]))
y_test = reshape(y_test_subsample, (int(len(y_test_subsample) / n_batch), n_batch, y_test_subsample.shape[1]))
y_val = reshape(y_val_subsample, (int(len(y_val_subsample) / n_batch), n_batch, y_val_subsample.shape[1]))

print('Training Recontruction...')
estimator = reconstruction_model(x_train_ls, y_train)

optimizer = Adam(learning_rate=0.00001)
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
estimator.compile(optimizer=optimizer, loss=losses.mean_squared_error,
                  metrics=[losses.mean_squared_error, losses.mean_absolute_error,
                           tf.keras.metrics.RootMeanSquaredError()])
history = estimator.fit(x_train_ls, y_train, batch_size=1, epochs=n_epoch, validation_data=(x_val_ls, y_val),
                        callbacks=[callback])

# %%
# summarize history for loss
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('output/figures/' + str(fs_sub) + '/Reconstruction_loss.png')
plt.show()

estimate_egms_test = estimator.predict(x_test_ls, batch_size=1)
estimate_egms_train = estimator.predict(x_train_ls, batch_size=1)

results_test = estimator.evaluate(x_test_ls, batch_size=50)
results_train = estimator.evaluate(x_train_ls, batch_size=50)

# Flatten y
y_test_flat = reshape(y_test, (y_test.shape[0] * y_test.shape[1], y_test.shape[2]))
y_train_flat = reshape(y_train, (y_train.shape[0] * y_train.shape[1], y_train.shape[2]))

# Flatten x for MSE nad MAE
reconstruction_flat_test = reshape(estimate_egms_test, (
estimate_egms_test.shape[0] * estimate_egms_test.shape[1], estimate_egms_test.shape[2]))

reconstruction_flat_train = reshape(estimate_egms_train, (
estimate_egms_train.shape[0] * estimate_egms_train.shape[1], estimate_egms_train.shape[2]))

# Save
mse_reconstruction_test, mae_reconstruction_test, rmse_reconstruction_test = results_test[0], results_test[2], \
results_test[3]
mse_reconstruction_train, mae_reconstruction_train, rmse_reconstruction_train = results_train[0], results_train[2], \
results_test[3]

# MAE and MSE
mae_reconstruction_test = mean_absolute_error(y_test_flat, reconstruction_flat_test)
mae_reconstruction_train = mean_absolute_error(y_train_flat, reconstruction_flat_train)

mse_reconstruction_test = mean_squared_error(y_test_flat, reconstruction_flat_test)
mse_reconstruction_train = mean_squared_error(y_train_flat, reconstruction_flat_train)

# DTW
dtw_test, path = fastdtw(reconstruction_flat_test, y_test_flat)
dtw_train, path = fastdtw(reconstruction_flat_train, y_train_flat)

results_reconstruction = {'mse test': mse_reconstruction_test, 'mse train': mse_reconstruction_train,
                          'mae test': mae_reconstruction_test, 'mae train': mae_reconstruction_train,
                          'rmse test': rmse_reconstruction_test, 'rmse train': rmse_reconstruction_train,
                          'dtw_test': dtw_test, 'dtw_train': dtw_train}

y_test_subsample = y_test_flat
estimate_egms_test = reconstruction_flat_test

estimate_egms_n = normalize_by_models(reconstruction_flat_test,BSPM_test)



norm = True
# Normalize (por muestras)
if norm:
    estimate_egms_n = []

    for model in np.unique(BSPM_test):
        print(model)
        print(model)
        print(model)

        # 1. Normalize Reconstruction
        arr_to_norm_estimate = reconstruction_flat_test[
            np.where((BSPM_test == model))]  # select window of signal belonging to model i
        estimate_egms_norm = normalize_array(arr_to_norm_estimate, 1, -1, 0)  # 0: por muestas
        estimate_egms_n.extend(estimate_egms_norm)  # Add to new norm array
    estimate_egms_n = np.array(estimate_egms_n)

# Plots

for i in range(0, 20):
    interv = random.randrange(0, len(estimate_egms_test) - 1, 50)
    # interv= 6600+i
    node = random.randrange(0, n_nodes, 1)
    # node= 69
    normalize_ = True
    rango = 4 * fs
    # normalize between -1 and 1
    estimate_signal = estimate_egms_n[interv: interv + rango, :]
    # estimate_egms_norm_represent = normalize_array(estimate_signal, 1, -1, 0)
    estimate_egms_norm_represent = estimate_signal

    plt.figure(figsize=(15, 3))

    plt.plot(estimate_egms_norm_represent[:, node], label='Estimation Test')
    # plt.plot(estimate_egms_norm[interv: interv+200 ,node], label='Estimation Test')
    plt.plot(y_test_subsample[interv:interv + rango, node], label='Test', alpha=0.5)
    # plt.plot(latent_vector_test[200:400, 0, 0, 0], label = 'Latent Vector')
    text = 'Node {} in second {} to {}'.format(node, interv / fs, interv / fs + rango / fs)
    plt.legend()
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.title(text)
    plt.savefig('output/figures/' + str(fs_sub) + '/Reconstruction' + str(i))
    plt.show()

plt.figure(layout='tight', figsize=(15, 10))
plt.subplot(1, 3, 1)
plt.imshow(normalize_array(y_test_subsample[0:1000, :].T, 1, 0, 0), cmap='Greys')
plt.title('y test (egm)')
plt.xlabel('time')
plt.ylabel('nodes')
plt.colorbar(orientation="horizontal", pad=0.2)
plt.subplot(1, 3, 2)
plt.imshow(normalize_array(estimate_egms_n[0:1000, :].T, 1, 0, 0), cmap='Greys')
plt.title('Reconstruction')
plt.xlabel('time')
plt.ylabel('nodes')
plt.colorbar(orientation="horizontal", pad=0.2)
plt.subplot(1, 3, 3)
dif = normalize_array(y_test_subsample[0:1000, :].T, 1, 0, 0) - normalize_array(estimate_egms_n[0:1000, :].T, 1, 0, 0)
dif = abs(dif)
plt.imshow(dif, cmap='Greys')
plt.title('y_test - reconstruction')
plt.xlabel('time')
plt.ylabel('nodes')
plt.colorbar(orientation="horizontal", pad=0.2)
plt.savefig('output/figures/' + str(fs_sub) + '/2D_Rec.png')
plt.show()

# Correlation and RMSE

# correlation_pearson_time=corr_pearson_cols(estimate_egms_n.T, y_test_subsample.T)
# correlation_pearson_nodes=corr_pearson_cols(estimate_egms_n, y_test_subsample)
# correlation_spearman_time=corr_spearman_cols(estimate_egms_n.T, y_test_subsample.T)
# correlation_spearman_nodes=corr_spearman_cols(estimate_egms_n, y_test_subsample)


# DF Mapping
DF_Mapping = False

if DF_Mapping:
    unique_test_models = np.unique(AF_models_test)
    for model in unique_test_models:
        # model= AF_models_test[1]
        estimation_array = estimate_egms_n[
            np.where((AF_models_test == model))]  # select window of signal belonging to model i
        y_array = y_test_subsample[np.where((AF_models_test == model))]  # select window of signal belonging to model i

        DF_rec, sig_k_rec, phase_rec = freq_pha.kuklik_DF_phase(estimation_array.T, fs)
        DF_label, sig_k_label, phase_label = freq_pha.kuklik_DF_phase(y_array.T, fs)

        # Interpolate DF Mapping to 2048 nodes

        sig_k_rec_i = interpolate_fun(sig_k_rec.T, len(sig_k_rec.T), 2048, sig=False)
        sig_k_label_i = interpolate_fun(sig_k_label.T, len(sig_k_label.T), 2048, sig=False)
        phase_rec_i = interpolate_fun(sig_k_rec.T, len(phase_rec.T), 2048, sig=False)
        phase_label_i = interpolate_fun(sig_k_label.T, len(phase_label.T), 2048, sig=False)

        print(sig_k_rec_i.shape, 'interpo df')

        # %%
        dic_DF = {"DF_rec": DF_rec, "sig_k_rec": sig_k_rec_i, "phase_rec": phase_rec_i,
                  "DF_label": DF_label, "sig_k_label": sig_k_label_i, "phase_label": phase_label_i}
        savemat("saved_var/" + str(fs_sub) + "/DF_Mapping_variables_" + str(model) + ".mat", dic_DF)

'''
plt.figure(figsize=(15, 5), layout='tight')
plt.subplot(3, 1, 1)
plt.plot(correlation_spearman_time, label='spearman')
plt.plot(correlation_pearson_time, alpha=0.5, label='pearson')
plt.legend()
plt.title('Time')
plt.subplot(2, 1, 2)
plt.plot(correlation_spearman_nodes, label='spearman')
plt.plot(correlation_pearson_nodes, alpha=0.5, label='pearson')
plt.title('Nodes')
plt.savefig('output/figures/'+ str(fs_sub) + '/Correlation.png')
plt.show()

'''
# %%
correlation_list = []

unique_AF_test_models = np.unique(AF_models_test)

for model in unique_AF_test_models:
    # 1. Normalize Reconstruction
    estimation_array = estimate_egms_n[
        np.where((AF_models_test == model))]  # select window of signal belonging to model i
    y_array = y_test_subsample[np.where((AF_models_test == model))]  # select window of signal belonging to model i
    correlation_pearson_nodes = corr_spearman_cols(estimation_array, y_array)
    correlation_list.extend([correlation_pearson_nodes])
correlation_array = np.array(correlation_list)

plt.figure(figsize=(15, 7), layout='tight')

for i in range(0, len(correlation_list) - 1):
    plt.subplot(len(correlation_list) - 1, 1, i + 1)
    plt.plot(correlation_list[i], label=all_model_names[unique_AF_test_models[i]])
    plt.title(all_model_names[unique_AF_test_models[i]])
    plt.xlabel('Nodes')
plt.ylabel('Coefficient of correlation')
plt.suptitle('Spearson correlation')
plt.savefig('output/figures/' + str(fs_sub) + '/Correlation_2.png')
plt.show()

# %%

dtw_array, dtw_array_random = DTW_by_AFModels(AF_models_test, estimate_egms_n, y_test_subsample)
rmse_array = RMSE_by_AFModels(AF_models_test, estimate_egms_n, y_test_subsample)
correlation_array = correlation_by_AFModels(AF_models_test, estimate_egms_n, y_test_subsample)

# %%
# Mean and STD of Spearman Correlation, DTW and RMSE
corr_mean = np.mean(correlation_array, axis=1)
corr_std = np.std(correlation_array, axis=1)
dtw_mean = np.mean(dtw_array, axis=1)
dtw_std = np.std(dtw_array, axis=1)
dtw_mean_random = np.mean(dtw_array_random, axis=1)
dtw_std_random = np.std(dtw_array_random, axis=1)
rmse_mean = np.mean(rmse_array, axis=1)
rmse_std = np.std(rmse_array, axis=1)

# Labels for plotting
# Labels for plotting
labels = []
for i in range(0, len(corr_mean)):
    labels.extend([all_model_names[unique_AF_test_models[i] - 1]])

x_pos = np.arange(len(labels))
x = np.linspace(0, len(labels), len(labels))

# Corr
plt.figure(figsize=(7, 2))
# fig, ax = plt.subplots()
plt.errorbar(x, corr_mean,
             yerr=corr_std,
             fmt='--o', label='Test')

# ax.set_xticks(x_pos)
# plt.xticks(rotation=90)
# ax.set_xticklabels(labels)
plt.tight_layout()
plt.xlabel('AF Model')
plt.title('Spearman Correlation in test models')
plt.savefig('output/figures/' + str(fs_sub) + '/Corr3.png')

plt.show()

# DTW
plt.figure(figsize=(7, 2))
# fig, ax = plt.subplots()
plt.errorbar(x, dtw_mean,
             yerr=dtw_std,
             fmt='--o', label='Test')
plt.errorbar(x, dtw_mean_random,
             yerr=dtw_std_random,
             fmt='--o', label='Random')
# plt.xticks(rotation=90)
# ax.set_xticklabels(labels)
plt.tight_layout()
plt.xlabel('AF Model')
plt.title('DTW in test models')
plt.legend()
plt.savefig('output/figures/' + str(fs_sub) + '/DTW.png')

plt.show()

# RMSE
plt.figure(figsize=(7, 2))
# fig, ax = plt.subplots()
plt.errorbar(x, rmse_mean,
             yerr=rmse_std,
             fmt='--o')
# plt.xticks(rotation=90)
# ax.set_xticklabels(labels)
plt.tight_layout()
plt.xlabel('AF Model')
plt.title('RMSE in test models')
plt.savefig('output/figures/' + str(fs_sub) + '/RMSE.png')
plt.show()

# plot_Welch_reconstruction(latent_vector_test, estimate_egms_n, fs, n_nodes,y_test_subsample, nperseg_value=100)

LS_signal = x_test
output_signal = estimate_egms_n
nperseg_value = 2 * fs

fig = plt.figure(layout='tight', figsize=(10, 6))

plt.subplot(3, 1, 1)

# reconstruction
for height in range(0, 511, 5):
    f, Pxx_den = scipy.signal.welch(output_signal[:, height], fs, nperseg=nperseg_value,
                                    noverlap=nperseg_value // 2, scaling='density', detrend='linear')
    plt.plot(f, Pxx_den, linewidth=0.5)
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD [V**2/Hz]')
    # plt.ylim([0,0.2])

    titlee = "PSD Welch of EGM signal estimation. node {}".format(height)
    plt.title('EGM signals reconstruction')

# Latent Space
plt.subplot(3, 1, 2)
for height in range(0, 3):
    for width in range(0, 4):
        for filters in range(0, 12, 2):
            f, Pxx_den = scipy.signal.welch(LS_signal[:, height, width, filters], fs, nperseg=nperseg_value,
                                            noverlap=nperseg_value // 2, scaling='density', detrend='linear')
            plt.plot(f, Pxx_den, linewidth=0.5)
            plt.xlabel('frequency [Hz]')
            plt.ylabel('PSD [V**2/Hz]')
            titlee = "PSD Welch of Latent Space signal. nodes ( - )".format(height, width)
            plt.title('latent Space')
            # plt.ylim([0,0.2])

fig.suptitle("Welch Periodogram (window size=200 samples)", fontsize=15)

# Input
plt.subplot(3, 1, 3)

for height in range(0, 511, 5):
    f, Pxx_den = scipy.signal.welch(y_test_subsample[:, height], fs, nperseg=nperseg_value,
                                    noverlap=nperseg_value // 2, scaling='density', detrend='linear')
    plt.plot(f, Pxx_den, linewidth=0.5)
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD [V**2/Hz]')
    titlee = "PSD Welch of EGM signal estimation. node {}".format(height)
    plt.title('EGM signals Real')
    # plt.ylim([0,0.2])

plt.savefig('output/figures/' + str(fs_sub) + '/Periodogram_rec.png')
plt.show()


# %%
results = pd.DataFrame(columns=['MSE AE', 'DTW AE', 'MSE Reconstruction', 'TWD Reconstruction'])

# %%
estimate_egms_reshaped = reshape(estimate_egms_n, (estimate_egms_n.shape[0], estimate_egms_n.shape[1], 1, 1))
interpol = tf.keras.layers.UpSampling2D(size=(4, 1), interpolation='bilinear')(estimate_egms_reshaped)
Test_estimation = reshape(interpol, (interpol.shape[0], interpol.shape[1]))

# %%
label_represent = y_test_subsample[:, :]
estimate_labels_reshaped = reshape(label_represent, (label_represent.shape[0], label_represent.shape[1], 1, 1))
interpol_label = tf.keras.layers.UpSampling2D(size=(4, 1), interpolation='bilinear')(estimate_labels_reshaped)
Label = reshape(interpol_label, (interpol_label.shape[0], interpol_label.shape[1]))

print('Saving variables...')

new_correlation_array = interpolate_fun(correlation_array, len(correlation_array), 2048)
new_rmse_array = interpolate_fun(rmse_array, len(rmse_array), 2048)

# %%
mdic = {"reconstruction": Test_estimation, "label": Label}
variables = {"RMSEmean": rmse_mean, "RMSEstd": rmse_std, 'Corrmean': corr_mean, "corrstd": corr_std,
             'dtwmean': dtw_mean, 'dtwstd': dtw_std,
             'corrbynodes': new_correlation_array, 'rmsenodes': new_rmse_array, 'test_model_name': test_model_name,
             "train_model_name": train_model_name,
             "val_model_name": val_model_name}

savemat("output/variables/" + str(fs_sub) + "/reconstruction2304.mat", mdic)
savemat("output/variables/" + str(fs_sub) + "/variables23_04.mat", variables)

# Save models
if sinusoids:
    estimator.save('sin_pretrained/model_reconstruction.h5')
    conv_autoencoder.save('sin_pretrained/model_autoencoder.h5')
else:
    estimator.save('output/model/model_reconstruction.h5')
    conv_autoencoder.save('output/model/model_autoencoder.h5')

estimator.save('output/model/model_reconstruction.h5')
conv_autoencoder.save('output/model/model_autoencoder.h5')

# Save results to csv and export
results_Autoencoder = pd.DataFrame.from_dict(results_autoencoder, orient='index', columns=['Autoencoder'])
results_Reconstruction = pd.DataFrame.from_dict(results_reconstruction, orient='index', columns=['Reconstruction'])
global_results = pd.concat([results_Autoencoder, results_Reconstruction], axis=1)
global_results.to_csv('Results_csv/' + str(fs_sub) + '/Results2304')

global_results.round(3)

# %%
end = time.time()
print((end - start) / 60, 'Mins of execution')
