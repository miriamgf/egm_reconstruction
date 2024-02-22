# %%
root_logdir = '../Logs/'
data_dir = '../Data2/'
figs_dir = 'Figs/'
models_dir = '../Models/'

# %%
# Tensorboard logs name generator
def get_run_logdir():
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)

# %%
import os
import time
import tensorflow as tf
import keras
from keras import models, layers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, f1_score
from itertools import product
from tools_ import *
from Architectures import *
from tensorflow.keras import datasets, layers, models, losses, Model
import pickle
from generators import *
import sys 
from numpy import reshape
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import cv2
import glob
from PIL import Image
import matplotlib.image
import time
import random
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.io import savemat
from Plots import *
from tensorflow.keras.optimizers import Adam
import datetime
import freq_phase_analysis as freq_pha
from scipy.interpolate import interp1d
import signal as sg
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error





# GPU Configuration
physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs:", len(physical_devices))
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
    
start = time.time()


all_torsos_names=[] 
for subdir, dirs, files in os.walk(torsos_dir):
    for file in files:
        if file.endswith('.mat'):
            all_torsos_names.append(file)
    
transfer_matrices=[]
for torso in all_torsos_names:
    BSP_pos = scipy.io.loadmat(torsos_dir + torso).get('torso')



all_model_names = []
directory = data_dir
for subdir, dirs, files in os.walk(directory):
    print(subdir, directory, files)

    if (subdir != directory):
        model_name = subdir.split("/")[-1]
        all_model_names.append(model_name)
       
all_model_names=sorted(all_model_names)
print (all_model_names)
print(len(all_model_names), 'Models')

# Load data
n_classes = 3 #1: Rotor/no rotor ; 2: RA/LA/No rotor (2 classes) ; 3: 7 regions (3 classes) + no rotor (8 classes)
n_batch = 50 # Batch size
fs_sub = 50 
fs = fs_sub  
Transfer_model = False #Transfer learning from sinusoids 
sinusoids= False
n_epoch = 3
# Folders ECGI Summit
try:
    os.mkdir('Figures/'+ str(fs_sub))
except:
    pass

try:
    os.mkdir('saved_var/'+ str(fs_sub))
except:
    pass

try:
    os.mkdir('Results_csv/'+ str(fs_sub))
except:
    pass

X_1channel,Y,Y_model, egm_tensor, length_list, AF_models, all_model_names, transfer_matrices = load_data (data_type='1channelTensor', n_classes=n_classes, subsampling= True, fs_sub=fs_sub, norm=False, SR=True, SNR=20, n_batch = n_batch, sinusoid = sinusoids) #, n_batch=n_batch)


# Normalize BSPS and EGMs by models (each Y model is normalized )
X_1channel = normalize_by_models(X_1channel,Y_model)
egm_tensor_n = normalize_by_models(egm_tensor,Y_model)

# Train/Test/Val Split BSPS
random_split = True
print('Splitting...')
x_train, x_test, x_val, train_models, test_models, val_models, AF_models_train, AF_models_test, AF_models_val, BSPM_train, BSPM_test, BSPM_val = train_test_val_split_Autoencoder(X_1channel,
 AF_models, Y_model, all_model_names, random_split =True,train_percentage= 0.6, test_percentage=0.2 )

#Split EGMs accordingly

if random_split: 
    y_train=egm_tensor_n[np.in1d(AF_models, train_models)]
    y_test=egm_tensor_n[np.in1d(AF_models, test_models)]
    y_val=egm_tensor_n[np.in1d(AF_models, val_models)]

else:

    y_train=egm_tensor_n[np.where((Y_model>=1) & (Y_model<=200))]
    y_test=egm_tensor_n[np.where((Y_model>180) & (Y_model<=244))]
    y_val=egm_tensor_n[np.where((Y_model>244) & (Y_model<=286))]


print('TRAIN SHAPE:',  x_train.shape, 'models:',train_models  )
print('TEST SHAPE:', x_test.shape,  'models:',test_models  )
print('VAL SHAPE:', x_val.shape ,  'models:',val_models  )

# Save the model names in train, test and val
test_model_name= [all_model_names[index] for index in AF_models_test]
val_model_name= [all_model_names[index] for index in AF_models_val]
train_model_name= [all_model_names[index] for index in AF_models_train]

# Subsample y
y_train = y_train[:, 0:2048:4]
y_test = y_test[:, 0:2048:4]
y_val = y_val[:, 0:2048:4]

#Batch generation
x_train_batch= reshape(x_train, (int(len(x_train)/n_batch),n_batch,  x_train.shape[1],  x_train.shape[2],1))
x_test_batch = reshape(x_test, (int(len(x_test)/n_batch),n_batch,   x_test.shape[1],  x_test.shape[2],1))
x_val_batch= reshape(x_val, (int(len(x_val)/n_batch), n_batch,  x_val.shape[1],  x_val.shape[2],1))

y_train_batch= reshape(y_train, (int(len(y_train)/n_batch),n_batch,  y_train.shape[1]))
y_test_batch = reshape(y_test, (int(len(y_test)/n_batch),n_batch,   y_test.shape[1]))
y_val_batch= reshape(y_val, (int(len(y_val)/n_batch), n_batch,  y_val.shape[1]))

# Reshape AF_models to compute corresponding AF model to each sample 
x_train_batch_AF= reshape(AF_models_train, (int(len(AF_models_train)/n_batch),n_batch)) 
x_test_batch_AF= reshape(AF_models_test, (int(len(AF_models_test)/n_batch),n_batch))
x_val_batch_AF= reshape(AF_models_val, (int(len(AF_models_val)/n_batch),n_batch))

plotting_before_AE(X_1channel, x_train, x_test, x_val, Y_model, AF_models )

#Reshape to fit ConvLSTM2D (Add 1 dimension at the beggining)
print('x shape before',x_train.shape, x_test.shape, x_val.shape) 
print('y shape before',y_train.shape, y_test.shape, y_val.shape) 

x_train= reshape(x_train, (int(len(x_train)/n_batch),n_batch,  x_train.shape[1],  x_train.shape[2],1))
x_test = reshape(x_test, (int(len(x_test)/n_batch),n_batch,   x_test.shape[1],  x_test.shape[2],1))
x_val= reshape(x_val, (int(len(x_val)/n_batch), n_batch,  x_val.shape[1],  x_val.shape[2],1))

y_train= reshape(y_train, (int(len(y_train)/n_batch),n_batch,  y_train.shape[1],1))
y_test = reshape(y_test, (int(len(y_test)/n_batch),n_batch,   y_test.shape[1],1))
y_val= reshape(y_val, (int(len(y_val)/n_batch), n_batch,  y_val.shape[1],1))

print('x shape after',x_train.shape, x_test.shape, x_val.shape) 
print('y shape after',y_train.shape, y_test.shape, y_val.shape) 

    
# 1. AUTOENCODER
print('Training Autoencoder...')
input_shape= x_train.shape[1:]
#reconstruction = reconstruction_BSPM_EGM(x_train, y_train)
model = MultiOutput().assemble_full_model(input_shape)


optimizer = Adam(lr=0.00001)
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

# Train
#DirectNet = Model(inputs=reconstruction.input, outputs = reconstruction.outputs)
model.compile(optimizer = optimizer, 
              loss={
                  'autoencoder_output': losses.mean_squared_error, 
                  'reconstruction_output': losses.mean_squared_error},
             
              metrics={
                  'autoencoder_output': [losses.mean_squared_error, losses.mean_absolute_error], 
                  'reconstruction_output': [losses.mean_squared_error, losses.mean_absolute_error]})
model.summary()


print('Fitting')
history = model.fit(x_train, [x_train, y_train], batch_size=1, epochs=n_epoch, validation_data=(x_val, y_val), callbacks= [callback, tensorboard_callback])
print('Predict')
decoded_BSPS, decoded_EGM  = model.predict(x_test, batch_size=50)

#Reshape to remove the 1 at the end
print(decoded_BSPS.shape, x_test.shape)
decoded_BSPS = reshape(decoded_BSPS, (len(decoded_BSPS), decoded_BSPS.shape[1],  decoded_BSPS.shape[2]))
decoded_EGM = reshape(decoded_EGM, (len(decoded_EGM), decoded_EGM.shape[1],  decoded_EGM.shape[2]))

decoded_imgs_train = reshape(decoded_imgs_train, (len(decoded_imgs_train), decoded_imgs_train.shape[1], decoded_imgs_train.shape[2]))

#Evaluate
results_test = model.evaluate(y_test, batch_size=50)
results_train = model.evaluate(y_train, batch_size=1)

#Save
mse_autoencoder_test, mae_autoencoder_test= results_test[0], results_test[2]
mse_autoencoder_train, mae_autoencoder_train= results_train[0], results_train[2]
print(results_test)

#Reshape for 
decoded_flat = reshape(decoded_imgs, (decoded_imgs.shape[0]*decoded_imgs.shape[1], decoded_imgs.shape[2]*decoded_imgs.shape[3]))
decoded_flat_train= reshape(decoded_imgs_train, (decoded_imgs_train.shape[0]*decoded_imgs_train.shape[1], decoded_imgs_train.shape[2]*decoded_imgs_train.shape[3]))  
x_test_flat = reshape(x_test, (x_test.shape[0]*x_test.shape[1], x_test.shape[2]*x_test.shape[3]))
x_train_flat = reshape(x_train, (x_train.shape[0]*x_train.shape[1], x_train.shape[2]*x_train.shape[3]))

#MAE and MSE
mae_autoencoder_test = mean_absolute_error(x_test_flat, decoded_flat)
mae_autoencoder_train = mean_absolute_error(x_train_flat, decoded_flat_train)

mse_autoencoder_test = mean_squared_error(x_test_flat, decoded_flat)
mse_autoencoder_train = mean_squared_error(x_train_flat, decoded_flat_train)


mse_autoencoder_test, mae_autoencoder_test

# %%
results_autoencoder= {'mse test': mse_autoencoder_test, 'mse train': mse_autoencoder_train,
'mae test': mae_autoencoder_test, 'mae train': mae_autoencoder_train,
 'dtw_test': dtw_test, 'dtw_train': dtw_train }
results_autoencoder

# summarize history for loss
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.title('Training and validation curves ')
plt.savefig('Figures/'+ str(fs_sub) + 'Learning_curves_AE.png')
plt.show()


# 
time_instant= random.randint(0, 49)
batch= random.randint(0, x_test.shape[0])

print(batch, time_instant)


plt.figure(figsize=(5,5), layout='tight')
plt.subplot(3,1,1)
#image1=define_image(time_instant,x_test )
image1=x_test[batch, time_instant, : :]
min_val, max_val = np.amin(image1), np.amax(image1)
plt.imshow(image1, vmin=min_val, vmax=max_val)#, cmap='jet')
plt.colorbar()
plt.title('Original')
plt.subplot(3,1,2)
#image2=define_image(time_instant,decoded_imgs)
image2=decoded_imgs[batch, time_instant, : :]
plt.imshow(image2, vmin=min_val, vmax=max_val) #, cmap='jet')
plt.colorbar()
plt.title('Reconstructed')
plt.subplot(3,1,3)
plt.imshow(image2-image1, vmin=min_val, vmax=max_val)
plt.title('Error (Reconstructed-Original)')
plt.colorbar()
plt.savefig('Figures/'+ str(fs_sub) + '/AE_reconstruction.png')
plt.show()


#PLOTS 50 samples --> 1 second
batch= random.randint(0, x_test.shape[0])

#Random signal visualizer: in each execution plots the first second of the BSPS in random nodes 
plt.figure(figsize=(10, 3))
a=randint(0, decoded_imgs.shape[2]-1)
b=randint(0, decoded_imgs.shape[3]-1)
#s=randint(0, len(decoded_imgs)-fs-1 )

title="50 samples from BSP from nodes {} x {} in the model {}".format(a,b,x_test_batch_AF[batch, 0] )
plt.plot(x_test[batch, :, a , b], label=('Input'))
plt.plot(decoded_imgs[batch, :, a , b], label=('Reconstruction'))
plt.fill_between(np.arange(n_batch), decoded_imgs[batch, :, a , b], x_test[batch, :, a , b], color='lightgreen')
plt.ylabel('Amplitude')
plt.xlabel('Samples')
plt.legend()
plt.title(title)
plt.savefig('Figures/'+ str(fs_sub) + '/AE_Recontruction2.png')
plt.show()

# %%
latent_vector = encoder.predict(x_test,batch_size=1)

#create function to center data
center_function = lambda x: x - x.mean(axis=0)

#Center latent space
latent_vector = center_function(latent_vector)


# %% [markdown]
# # 3.3 Plot latent space frames in random time instant 

# %%
time_instant= randint(0, 49 )
batch= random.randint(0, x_test.shape[0])

plt.figure(figsize=(6,6), layout='tight')
title="Filters from Latent Space in time instant {} ".format(time_instant)
for i in range (0,12):
    plt.subplot(6, 2, i+1)
    #plt.plot(6, 2, i) 
    plt.imshow(latent_vector[batch,time_instant, :, :,i])#, cmap='jet')
    plt.colorbar()
plt.suptitle(title)
plt.savefig('Figures/'+ str(fs_sub) + '/Latentspace1.png')

plt.show()

# %%
plt.figure()
plt.plot(x_test[batch, :, 1, 1], label ='Input')#, cmap='jet')

for i in range (0,12):
    #plt.plot(6, 2, i) 
    plt.plot(latent_vector[batch, :, 1, 1,i], alpha=0.5,  label=i)#, cmap='jet')
    plt.title('Feature vector in time (50 samples) of 12 filters')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.savefig('Figures/'+ str(fs_sub) + '/Latentspace2.png')

plt.show()

# %%
#Flatten 
x_train_fl=reshape(x_train, (x_train.shape[0]*x_train.shape[1], x_train.shape[2], x_train.shape[3] ))
x_test_fl=reshape(x_test, (x_test.shape[0]*x_test.shape[1], x_test.shape[2], x_test.shape[3] ))
x_val_fl=reshape(x_val, (x_val.shape[0]*x_val.shape[1], x_val.shape[2], x_val.shape[3] ))
decoded_imgs_fl=reshape(decoded_imgs, (decoded_imgs.shape[0]*decoded_imgs.shape[1], decoded_imgs.shape[2], decoded_imgs.shape[3] ))
latent_vector=reshape(latent_vector, (latent_vector.shape[0]*latent_vector.shape[1], latent_vector.shape[2], latent_vector.shape[3], latent_vector.shape[4] ))


# Plot of reconstruction (orange) over the context of 14 seconds of random input signal 

#PLOT INPUT VS OUTPUT
window_length_test=500 #samples
window_length_subsegment=200
t_vector= np.array(np.linspace(0, window_length_test/fs, window_length_test))
t_vector_reshaped=t_vector.reshape(t_vector.shape[0], 1, 1)

#Random big window to show x_test
randomonset_test=random.randrange(len(x_test_fl)-window_length_test)
random_window_test= [randomonset_test, randomonset_test + window_length_test]

x_test_cut=x_test_fl[random_window_test[0]:random_window_test[1],1, 1 ]
decoded_imgs_cut=decoded_imgs_fl[random_window_test[0]:random_window_test[1],1, 1 ]

#smaller window to show subsegment inside x_test
randomonset=random.randrange(len(x_test_cut)-window_length_subsegment)
random_window_subsegment= [randomonset, randomonset + window_length_subsegment]
print(decoded_imgs_cut.size, 'total length', random_window_subsegment, 'subwindow size')
copy_decoded_imgs_cut=decoded_imgs_cut
decoded_imgs_cut[:random_window_subsegment[0]]=None
decoded_imgs_cut[random_window_subsegment[1]:]=None

plt.figure(figsize=(10,2))
plt.plot(t_vector, x_test_cut, alpha=0.5, label=('Test'))
plt.plot(t_vector, decoded_imgs_cut, label=('Reconstruction') )
plt.xlabel('Time(s)')
plt.ylabel('mV')
plt.title('14 seconds of Test BSPS. Over it, 2 seconds of BSPS reconstruction')
plt.savefig('Figures/'+ str(fs_sub) + '/BSPS_AE.png')
plt.show()

# PSD Welch of input, output and Latent Space

plot_Welch_periodogram(x_test_fl, latent_vector, decoded_imgs_fl, fs=fs)
plt.savefig('Figures/'+ str(fs_sub) + '/Periodogram_AE.png')


#Input --> Latent space 

latent_vector_train = encoder.predict(x_train, batch_size=1)
latent_vector_test = encoder.predict(x_test, batch_size=1)
latent_vector_val  = encoder.predict(x_val, batch_size=1)

latent_space_n, egm_tensor_n = preprocess_latent_space(latent_vector_train, latent_vector_test, latent_vector_val, Y_model, egm_tensor, dimension = 5)


