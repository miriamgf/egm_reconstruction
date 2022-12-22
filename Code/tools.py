# -*- coding: utf-8 -*-
#"""
#Created on Thu Jan 28 11:26:06 2021

#@author: Miguel Ángel
#"""

import sys,os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from scipy.io import loadmat
import numpy as np
from random import randint
import matplotlib.pyplot as plt
from itertools import product
from numpy.random import default_rng
import scipy.io
import h5py
from scipy import signal as sigproc
from scipy.interpolate import interp1d
from math import floor
from scipy import signal
from add_white_noise import *


#%% Path Models
#%% Path Models
current = os.path.dirname(os.path.realpath(__file__))
directory = str(os.path.dirname(current))+'/Data/'
torsos_dir = str(os.path.dirname(current))+'/Labeled_torsos/' 
fs=500

#%%

#%% Add noise to signals
def add_noise(X,SNR=20, fs=50):
    
    X_noisy, _ = addwhitenoise(X, SNR=SNR, fs=fs)
    
    #Normalizar
    #mm = np.mean(X_noisy, axis=0)
    #ss = np.std(X_noisy, axis=0)
    #X_noisy_normalized = (X_noisy - mm[np.newaxis,:]) / ss[np.newaxis,:]
    #X_noisy_normalized=  X_noisy 
    return X_noisy


def ECG_filtering(signal, fs, f_low=3, f_high=30):
    """
    Frequency filtering of ECG-EGM.
    SR model: low-pass filtering, 4th-order Butterworth filter.
    FA models: bandpass filtering, 4th-order Butterworth filter.
    
    Parameters:
        signal (array): signal to process
        fs (int): sampling rate
        f_low (int-float): low cut-off frecuency (default=3Hz)
        f_high (int-float): high cut-off frecuency (default=30Hz)
        model (string): FA model to assess (default: SR)
    Returns:
        proc_ECG_EGM (array): filtered ECG-EGM
    """
    
    # Remove DC component
    sig_temp=remove_mean(signal)
    
    #Bandpass filtering
    b, a = sigproc.butter(4, [f_low/round((fs/2)), f_high/round((fs/2))], btype='bandpass')
    proc_ECG_EGM=np.zeros(sig_temp.shape)

    for index in range(0, sig_temp.shape[0]):
        proc_ECG_EGM[index,:]=sigproc.filtfilt(b,a,sig_temp[index,:])

    return proc_ECG_EGM

def load_data(data_type, n_classes = 2,  SR = True, subsampling=True, fs_sub=50, SNR=20, norm=False):
    """
    Returns y values depends of the classification model
    
    Parameters:
        data_type: '3channelTensor' -> tx6x4x3 tensor (used for CNN-based classification)
                   '1channelTensor' -> tx6x16 tensor (used for CNN-based classification)

                   'Flat' ->  tx64 matrix (used for MLP-based classification) or tx10
    Returns:
        X -> input data.
        Y -> labels.
        Y_model -> model associated for each time instant.
    """

    #% Check models in directory
    all_model_names = []
    
    for subdir, dirs, files in os.walk(directory):
        if (subdir != directory):
            model_name = subdir.split("/")[-1]
            if 'Sinusal' in model_name and SR == False:
                continue
            else:
                all_model_names.append(model_name)
                
    all_model_names=sorted(all_model_names)
    print (all_model_names)
    
    
    #% Load models
    X = []
    Y = []
    Y_model = []
    n_model = 1
    egm_tensor=[]
    
    # Load corrected transfer matrices
    transfer_matrices=load_transfer()
    
    for model_name in all_model_names: 
        
        #%% 1)  Compute EGM of the model
        egms = load_egms(model_name)
        
        
        # 1.1) Discard models <1500 
        if len(egms[1])<1500:
            continue
        
        # 1.2)  EGMs filtering.
        x= ECG_filtering(egms, fs)   
        
        #1.3 Normalize models
        if norm == True:
            high = 1
            low = -1

            mins = np.min(x, axis=0)
            maxs = np.max(x, axis=0)
            rng = maxs - mins

            bsps_64_n = high - (((high - low) * (maxs - x)) / rng)
                
        # 2) Compute the Forward problem with each of the transfer matrices
        for matrix in transfer_matrices:
            y = forward_problem(x,matrix[0])
            bsps_64 = (y[matrix[1].ravel(),:])
            bsps_64_or=bsps_64
            
            # 3) Add NOISE and Filter 
            if SNR != None:
               bsps_64=add_noise(np.array(bsps_64),SNR=SNR, fs=fs)
  
            # 5) Filter AFTER adding noise
            bsps_64= ECG_filtering(bsps_64, fs)
            
            
            # RESAMPLING signal to fs= fs_sub
            if subsampling: 
                bsps_64 = signal.resample_poly(bsps_64,fs_sub,500, axis=1)
                x_sub = signal.resample_poly(x,fs_sub,500, axis=1)

            y_labels = get_labels(n_classes, model_name)
            y_labels_list=y_labels.tolist()
           
            # RESAMPLING labels to fs= fs_sub
            if subsampling: 
                y_labels = sample(y_labels,len(bsps_64[1]))
         
            y_model = np.full(len(y_labels), n_model)
            n_model += 1
            
            Y.extend(y_labels)
            Y_model.extend(y_model)
            egm_tensor.extend(x_sub.T)
            
            if data_type == '3channelTensor':
                tensors_model = get_tensor_model(bsps_64, tensor_type='3channel')
                X.extend(tensors_model)
            elif data_type == '1channelTensor':
                tensors_model = get_tensor_model(bsps_64, tensor_type='1channel')
                X.extend(tensors_model)
            else:
                X.extend(bsps_64.T)
                
               
    return np.array(X),np.array(Y),np.array(Y_model),np.array(egm_tensor)

def sample(input,count):
     ss=float(len(input))/count
     return [ input[int(floor(i*ss))] for i in range(count) ]

def load_egms(model_name):
    """
    Load electrograms and select 2500 time instants
 
    Parameters:
        model (str): Model to load
 
    Returns:
        x: Electrograms for the selected model
    """
    
    try:
        EG = np.transpose(np.array((h5py.File(directory + model_name + '/EGMs.mat','r')).get('x')))
    except:
        EG = scipy.io.loadmat(directory + model_name + '/EGMs.mat').get('x')
    
    return EG

def remove_mean(signal):
    """
    Remove mean from signal
 
    Parameters:
        signal (array): signal to process
 
    Returns:
        signotmean: signal with its mean removed
    """
    signotmean=np.zeros(signal.shape)
    for index in range(0, signal.shape[0]):
        signotmean[index,:]=sigproc.detrend(signal[index,:],type='constant')
    return signotmean



def forward_problem(EGMs,MTransfer):
    """
    Calculate ECGI forward problem from atrial EGMs
 
    Parameters:
        EGMs (array): atrial electrograms.
        MTransfer (matrix): transfer matrix.
    Returns:
        ECG (array): ECG reconstruction (torso).
    """
    ECG=np.matmul(MTransfer,EGMs)
    return ECG

def load_transfer(ten_leads=False):
    """
    Load the transfer matrix for atria and torso models
 
    Returns:
        MTransfer: Transfer matrix for atria and torso models
    """
    
    all_torsos_names=[] 
    for subdir, dirs, files in os.walk(torsos_dir):
        for file in files:
            if file.endswith('.mat'):
                all_torsos_names.append(file)
    
    transfer_matrices=[]
    for torso in all_torsos_names:
        MTransfer = scipy.io.loadmat(torsos_dir + torso).get('TransferMatrix')
        
        if ten_leads==True:
            BSP_pos = scipy.io.loadmat(torsos_dir + torso).get('torso')['leads'][0,0]
        else:
            BSP_pos = scipy.io.loadmat(torsos_dir + torso).get('torso')['bspmcoord'][0,0]
  
        # Transform transfer matrix to account for WCT correction. A matrix is the result of
        # referencing MTransfer to a promediated MTransfer. THe objective is to obtain an ECG
        # referenced to the WCT, following the next expression:
        # ECG_CTW = MTransfer * EGM - M_ctw * MTransfer * EGM = 
        # = (MTransfer - M_ctw * MTransfer) * EGM = MTransfer_ctw * EGM
        M_wct=(1/(MTransfer.shape)[0])*np.ones(((MTransfer.shape)[0],(MTransfer.shape)[0]))
        A=MTransfer-np.matmul(M_wct,MTransfer)
        
        transfer_matrices.append((A,BSP_pos-1))

    return transfer_matrices    

def get_bsps_192(model_name,ten_leads=False):
  """
    Reduce the 659 BSPs to 192, 3 BSPSs for each node.

    Parameters: 
      model_name: name of the model

    Return:
      bsps_192: array with 192 BSPs

  """
  torso_electrodes = loadmat(directory + "/torso_electrodes.mat")
  torso_electrodes = torso_electrodes["torso_electrodes"][0]

  bsps = loadmat(directory + model_name + '/BSPs.mat') 
  bsps = bsps['y']
  
  bsps_192 = []
  for i in torso_electrodes:
      bsps_192.append(bsps[i,:])
    
  return np.array(bsps_192)

def get_tensor_model(bsps_64,tensor_type='3channel'):
    """
    Get X (tensor) from one model

    Parameters:
      bsps_64: 64 x n_time matrix bsps for 1 model
      y_model: array of y labels from 1 model

    Return:
      all_tensors: array of all tnesor from 1 model
    """
    all_tensors = np.array([]) 
    
    for t in range(0, bsps_64.shape[1]):
        if tensor_type == '3channel':
            tensor_model = get_subtensor_2(bsps_64[:,t],tensor_type)
        else:
            tensor_model = get_subtensor_2(bsps_64[:,t],tensor_type)
        if all_tensors.size == 0:
            all_tensors = tensor_model
        else:
            all_tensors = np.concatenate((all_tensors, tensor_model), axis=0)

    return all_tensors

def get_bsps_64(bsps_192,seed='Y'):
    """
    Reduce 192 BSPs to 64, selecting 1 random BSPs of the 3 posibilities for each node
    
    Parameters: 
    bsps_192: 192 x n_times matrix for 1 model
    
      Returns:
        bsps_64: 64 x n_time matrix for 1 model
    """
    
    if seed == 'Y':
    	rng = default_rng(0)
    else:
    	rng = default_rng()
    	
    bsps_64 = []
    pos = 0
    num_row = int(bsps_192.shape[0]/3)

    for i in range(0, num_row):
      rand = rng.integers(low=0,high=2,endpoint=True)
      bsps = bsps_192[pos:pos+3][rand]
      bsps_64.append(bsps)
      pos += 3
    
    return np.array(bsps_64)

def replace_null_labels(labels):
    closest_i=labels[(labels!=0).argmax(axis=0)] #in position 0, as there is no left position it takes the first right label instead
    for index in range(0,len(labels)):
        if labels[index]>0:
                closest_i=labels[index]
        else:
            labels[index]=closest_i
    return labels

def get_labels(class_type, model_name):
    """
    Get y labels for each classification type: 2, 3 or 7.

    Parameters:
      class_type:  2 -> binary classification, exist(1) or does not exist(0) driver.
                   3 -> multi-classification: does not exist(0), LA(1) and RA(2)
                   7 -> multi-classification: does not exist(0), 1-7 for each region
                   Any other value of class_type raise a ValueError.

      model_name: name of the model.

      Return:
        labels: y labels fro the model
    """

    regions = loadmat(directory + "/regions.mat")
    regions = regions['regions'][0]
    
    path_driver_position = directory + model_name + '/driver_position.mat'
    driver_position = loadmat(path_driver_position)
    driver_position = driver_position['driver_position']

    nodes_aux = driver_position[:,1]
    driver_position = driver_position[:,0]
    
    #Obtener el nodo correspondiente a una etiqueta
    nodes = []
    for i in nodes_aux:
        if i.size == 0:
            nodes.append(9999)
        else:
            nodes.append(i)
    nodes = np.array(nodes,dtype=int)
    
    # Obtener solo las etiquetas.
    labels = np.array(driver_position,dtype=int)
    
    # Hay o no hay rotor
    if class_type == 2:
        return labels 
    
    # No hay, Izq o dcha
    if class_type == 3:
        for subdir, dirs, files in os.walk(directory):
            for subdir2, dirs2, files2 in os.walk(subdir):
                if subdir != directory:
                    # Si es RA y hay driver, y=2
                    if model_name.startswith("RA"):
                        labels[labels == 1] = 2
                    if model_name == 'TwoRotors_181219':
                        labels[labels == 1] = 2
        return labels
    
    # No hay o 7 regiones
    if class_type == 6:
        for index, item in enumerate(labels):
            if item != 0:
              
                labels[index] = regions[nodes[index]-1]
        # replace 0 with previous label
        labels=replace_null_labels(labels)
                
        return labels

def get_subtensor_2(bsps_64_t,tensor_type='3_channel'):
    """
    Get (6 x 4 x 3) tensor for 1 instance of time.
    
    Parameters: 
    bsps_64_t: 1 instance of time of bsps_64
    
    Return:
    subtensor: 6 x 4 x 3 matrix 
    """
    
    patches = get_patches_name(bsps_64_t)
    
    if tensor_type=='3channel':
        torso = np.array([[patches['A6'], patches['A12'], patches['B12'], patches['B6']],
                      [patches['A5'], patches['A11'], patches['B11'], patches['B5']],
                      [patches['A4'], patches['A10'], patches['B10'], patches['B4']],
                      [patches['A3'], patches['A9'], patches['B9'], patches['B3']],
                      [patches['A2'], patches['A8'], patches['B8'], patches['B2']],
                      [patches['A1'], patches['A7'], patches['B7'], patches['B1']]])
        
        back = np.array([[patches['D12'], patches['D6'], patches['C12'], patches['C6']],
                    [patches['D11'], patches['D5'], patches['C11'], patches['C5']],
                    [patches['D10'], patches['D4'], patches['C10'], patches['C4']],
                    [patches['D9'], patches['D3'], patches['C9'], patches['C3']],
                    [patches['D8'], patches['D2'], patches['C8'], patches['C2']],
                    [patches['D7'], patches['D1'], patches['C7'], patches['C1']]])
        
        side = np.array([[0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [patches['R8'], patches['R4'], patches['L4'], patches['L8']],
                    [patches['R7'], patches['R3'], patches['L3'], patches['L7']],
                    [patches['R6'], patches['R2'], patches['L2'], patches['L6']],
                    [patches['R5'], patches['R1'], patches['L1'], patches['L5']]])
         
        subtensor = np.stack((torso, side, back), axis=-1)
        subtensor = subtensor.reshape(1,6,4,3)
    
    else:
        interp_lat_R4=np.mean((patches['A6'],patches['A5'],patches['R4']))
        interp_lat_R8=np.mean((patches['D12'],patches['D11'],patches['R8']))
        interp_lat_L8=np.mean((patches['C6'],patches['C5'],patches['L8']))
        interp_lat_L4=np.mean((patches['B6'],patches['B5'],patches['L4']))

        interp_lat_R1=np.mean((patches['A1'],patches['A2'],patches['R1']))
        interp_lat_R5=np.mean((patches['D7'],patches['D8'],patches['R5']))
        interp_lat_L5=np.mean((patches['C1'],patches['C2'],patches['L5']))
        interp_lat_L1=np.mean((patches['B1'],patches['B2'],patches['L1']))

        subtensor = np.array([[[patches['B6'],patches['B12'],patches['A12'],patches['A6'],interp_lat_R4, interp_lat_R8,patches['D12'],patches['D6'],patches['C12'],patches['C6'],interp_lat_L8,interp_lat_L4,patches['B6'],patches['B12'],patches['A12'],patches['A6']],
                               [patches['B5'],patches['B11'],patches['A11'],patches['A5'],patches['R4'],patches['R8'],patches['D11'],patches['D5'],patches['C11'],patches['C5'],patches['L8'],patches['L4'],patches['B5'],patches['B11'],patches['A11'],patches['A5']],
                               [patches['B4'],patches['B10'],patches['A10'],patches['A4'],patches['R3'],patches['R7'],patches['D10'],patches['D4'],patches['C10'],patches['C4'],patches['L7'],patches['L3'],patches['B4'],patches['B10'],patches['A10'],patches['A4']],
                               [patches['B3'],patches['B9'],patches['A9'],patches['A3'],patches['R2'],patches['R6'],patches['D9'],patches['D3'],patches['C9'],patches['C3'],patches['L6'],patches['L2'],patches['B3'],patches['B9'],patches['A9'],patches['A3']],
                               [patches['B2'],patches['B8'],patches['A8'],patches['A2'],patches['R1'],patches['R5'],patches['D8'],patches['D2'],patches['C8'],patches['C2'],patches['L5'],patches['L1'],patches['B2'],patches['B8'],patches['A8'],patches['A2']],
                               [patches['B1'],patches['B7'],patches['A7'],patches['A1'],interp_lat_R1,interp_lat_R5,patches['D7'],patches['D1'],patches['C7'],patches['C1'],interp_lat_L5,interp_lat_L1,patches['B1'],patches['B7'],patches['A7'],patches['A1']]
                               ]])
    
        subtensor = subtensor.reshape(1,6,16)
    
    return subtensor


def get_patches_name(bsps_64):
    """
    Get names of patches in bsps_64

    Parameters:
        bsps_64: 
    Return:
        patches: dictionary whit patche name as key and bsps as value.
    """
    patches = {}

    index = 1
    for i in range(0,12):
      patches["A{0}".format(index)] = bsps_64[i]
      index +=1
      
    index = 1
    for i in range(12,24):
      patches["B{0}".format(index)] = bsps_64[i]
      index +=1
      
    index = 1
    for i in range(24,36):
      patches["C{0}".format(index)] = bsps_64[i]
      index +=1 
      
    index = 1
    for i in range(36,48):
      patches["D{0}".format(index)] = bsps_64[i]
      index +=1 
      
    index = 1
    for i in range(48,56):
      patches["L{0}".format(index)] = bsps_64[i]
      index +=1 
      
    index = 1
    for i in range(56,64):
      patches["R{0}".format(index)] = bsps_64[i]
      index +=1
      
    return patches
    
def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
      pass
        #print('Confusion matrix, without normalization')

    #print(cm)

    thresh = cm.max() / 2.
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                  horizontalalignment="center",
                  color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')