# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 18:18:11 2021
@author: Miguel Ángel
"""

import tensorflow as tf
import keras
import numpy as np
from sklearn.model_selection import train_test_split
from add_white_noise import *
from tools import *
from numpy.random import default_rng
from keras.preprocessing.image import ImageDataGenerator
from scipy import stats
from keras.utils import np_utils
import random
from sklearn.utils import shuffle
import tensorflow as tf
import keras
from keras import models, layers
import tensorflow_addons as tfa
from sklearn.model_selection import train_test_split,KFold
from tensorflow.keras.layers import Concatenate, Dense, Input, LSTM, Embedding, Dropout, Activation, GRU, Flatten
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import os
import time
import pickle




directory = '../Data/'

#%% Train-test split (by time and by model)
def train_test_val_split(Y_model,split_type='Time',test_size=0.2,validation_size=0.2, seed=None):
    """
    Return the time indexes associated with the training, test and validation sets, using time independence or model independence.
    Parameters
    ----------
    Y_model : TYPE
        DESCRIPTION.
    split_type : TYPE, optional
        DESCRIPTION. The default is 'Time'.
    test_size : TYPE, optional
        DESCRIPTION. The default is 0.2.
    validation_size : TYPE, optional
        DESCRIPTION. The default is 0.2.
    seed : TYPE, optional
        DESCRIPTION. The default is None.
    Returns
    -------
    t_train : TYPE
        DESCRIPTION.
    t_val : TYPE
        DESCRIPTION.
    t_test : TYPE
        DESCRIPTION.
    """
    t_indexes=np.arange(0,len(Y_model))
    if split_type=='Splitted':
        # Check the number of models
        n_models=np.unique(Y_model)
    
        # Create train-val-test lists
        train_list=[]
        val_list=[]
        test_list=[]
        
        # Split time instants in train-val-test
        start_value=0
        for model in n_models:
            labels=Y_model[np.where(Y_model==model)]
        
            # Check length of signals
            len_sig=len(labels)
            len_test = int(np.floor(test_size * len_sig))
            len_train_temp = len_sig-len_test
            len_val = int(np.floor(validation_size * len_train_temp))
            len_train=len_train_temp-len_val

            train_vector=np.arange(start_value,start_value+len_train)
            val_vector=np.arange(start_value+len_train,start_value+len_train+len_val)
            test_vector=np.arange(start_value+len_train+len_val,start_value+len_train+len_val+len_test)
            
            start_value+=len_sig
            
            train_list.extend(train_vector)
            val_list.extend(val_vector)
            test_list.extend(test_vector)
              
        t_train = np.sort(np.array(train_list).ravel())
        np.random.shuffle(t_train)
        t_val=np.sort(np.array(val_list).ravel())
        t_test=np.sort(np.array(test_list).ravel())
        
    elif split_type=='Time':
        t_train_val, t_test=train_test_split(t_indexes, test_size=test_size, random_state=seed)
        t_train, t_val=train_test_split(t_train_val, test_size=validation_size, random_state=seed)
        
    else:
        if seed != None:
            rng = default_rng(seed)
        else:
            rng = default_rng()
            
        # Check the number of models.
        y_model_unique=np.unique(Y_model)
        
        # Select the models that will go to test
        n_models_test=int(np.ceil(len(y_model_unique)*(test_size)))
        models_to_test=set((sorted(rng.choice(y_model_unique,size=n_models_test,
                                                      replace=False))))
        
        # Obtain the models that will go to training/val and test sets
        y_model_unique=set(y_model_unique.tolist())
        models_to_train_val=y_model_unique-models_to_test
        models_to_train_val=np.array(list(models_to_train_val))
        models_to_test=np.array(list(models_to_test))
        
        # Get the time instants associated with training/val and test sets.
        t_test=[]
        for n_model in models_to_test:
            t_values=t_indexes[np.where(Y_model==n_model)]
            t_test.extend(t_values)
            
        t_train_val = set(t_indexes.tolist())-set(t_test)
        
        t_test=np.array(t_test)
        t_train_val = np.array(list(t_train_val))
        
        # Obtain the indexes of the training and validation sets
        t_train, t_val=train_test_split(t_train_val, test_size=validation_size, random_state=seed)
        
    return t_train,t_val,t_test


#%% Add noise to signals
def add_noise(X,SNR=20, fs=50):
    
    X_noisy, _, _ = addwhitenoise(X, SNR=SNR, fs=fs)
    #Normalizar
    mm = np.mean(X_noisy, axis=1)
    ss = np.std(X_noisy, axis=1)
    X_noisy_normalized = (X_noisy - mm[:, np.newaxis]) / ss[:, np.newaxis]
        
    return X_noisy_normalized

#%% Interpolate tensor
def preprocess_input(x_tensor,tensor_type):
    if tensor_type=='3channel':
        x_reshaped = tf.cast(np.reshape(x_tensor,(1,x_tensor.shape[0],x_tensor.shape[1],x_tensor.shape[2])),tf.float32)
        x_interpolated = tf.keras.layers.UpSampling2D(size=(25, 38), interpolation='bilinear')(x_reshaped) 
    else:
        x_reshaped = tf.cast(np.reshape(x_tensor,(1,x_tensor.shape[0],x_tensor.shape[1],1)),tf.float32)
        x_interpolated = tf.keras.layers.UpSampling2D(size=(13, 12), interpolation='bilinear')(x_reshaped) 
    
    return x_interpolated.numpy()[0,:,:,:]

#%% Generator for RNN
def generator_batches_RNN(X,Y,Y_model, test_percentage=0.2, val_percentage=0.2, input_size=50, number_classes = 8, fs=500, val=True, shuffle_batches_train = True):
    
    train_list=[]
    val_list=[]
    test_list=[]
    n_models=np.unique(Y_model)

    # Store signals
    for model in n_models:
        signals=X[np.where(Y_model==model)]
        labels=Y[np.where(Y_model==model)]
        
        # Check length of signals
        len_sig=signals.shape[0]
        len_test = int(np.floor(test_percentage * len_sig))
        if len_test<input_size: len_test=input_size
        
        len_val = int(np.floor(val_percentage * len_sig))
        if len_val<input_size: len_val=input_size
        
        len_train = len_sig-len_val-len_test
        if len_train<input_size:
             raise ValueError("The training size is less than the input size")
        
        
        # Split signals in training, val and test
        train_sigs=signals[0:len_train,:].reshape((1,len_train,64))
        val_sigs=signals[len_train:len_train+len_val,:].reshape((1,len_val,64))
        test_sigs=signals[len_train+len_val:len_train+len_val+len_test,:].reshape((1,len_test,64))
        
        # Same with labels
        train_labels=labels[0:len_train]
        val_labels=labels[len_train:len_train+len_val]
        test_labels=labels[len_train+len_val:len_train+len_val+len_test]
        
        # Check the number of full signals chunks (depends on the input size)
        train_chunks=len_train//input_size
        val_chunks=len_val//input_size
        test_chunks=len_test//input_size
        
        # Truncate signals depending on the input size
        train_sigs=np.split(train_sigs[:,0:input_size*train_chunks,:],train_chunks,axis=1)
        val_sigs=np.split(val_sigs[:,0:input_size*val_chunks,:],val_chunks,axis=1)
        test_sigs=np.split(test_sigs[:,0:input_size*test_chunks,:],test_chunks,axis=1)
        
        # Same with labels (for each chunk, I pick the mode)
        train_labels=stats.mode(np.split(train_labels[0:input_size*train_chunks],train_chunks),axis=1)[0].ravel().tolist()
        val_labels=stats.mode(np.split(val_labels[0:input_size*val_chunks],val_chunks),axis=1)[0].ravel().tolist()
        test_labels=stats.mode(np.split(test_labels[0:input_size*test_chunks],test_chunks),axis=1)[0].ravel().tolist()
        
        # Associate signals with labels
        train_data=list(zip(train_sigs,keras.utils.np_utils.to_categorical(train_labels, num_classes=number_classes).reshape(train_chunks, 1,number_classes)))
        val_data=list(zip(val_sigs,keras.utils.np_utils.to_categorical(val_labels, num_classes=number_classes).reshape(val_chunks,1,number_classes)))
        test_data=list(zip(test_sigs,keras.utils.np_utils.to_categorical(test_labels, num_classes=number_classes).reshape(test_chunks,1,number_classes)))
        
        # Store data in its corresponding list
        train_list.extend(train_data)
        val_list.extend(val_data)
        test_list.extend(test_data)
        
    if val==True:                    
        if shuffle_batches_train == True:
            train_list = shuffle(train_list)
        
        n_batches_train=len(train_list)
        n_batches_val=len(val_list)
        n_batches_test=len(test_list)

        train_gen=aux_generator_RNN(train_list)
        val_gen=aux_generator_RNN(val_list)  
        test_gen=aux_generator_RNN(test_list)
        
        return n_batches_train,n_batches_val,n_batches_test,train_gen,val_gen,test_gen
    else:
        train_list = train_list + val_list
        
        if shuffle_batches_train == True:
            train_list = shuffle(train_list)
    
        n_batches_train=len(train_list)
        n_batches_test=len(test_list)
        
        train_gen=aux_generator_RNN(train_list)
        test_gen=aux_generator_RNN(test_list)
        
        return n_batches_train,n_batches_test,train_gen,test_gen



    
def model_CV(input_size, n_labels, n_batches_train, n_batches_test, train_gen,
             test_gen, acc_per_fold,loss_per_fold, acc_per_fold_train, loss_per_fold_train, conf_matrix_test, conf_matrix_train,classif_report_list,k):
            
            attention=False
            # Clear previous keras models
            root_logdir = '../Logs/'
            data_dir = '../Data'
            figs_dir = 'Figs/'
            models_dir = '/home/mgutierrez/Desktop/RNN/RNN_AF_Driver_Prediction/Code/Models/'
            # Tensorboard logs name generator
            def get_run_logdir():
                run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
                return os.path.join(root_logdir, run_id)
            
            if attention:
                
                class Attention(tf.keras.Model):
                    def __init__(self, units):
                        super(Attention, self).__init__()
                        self.W1 = tf.keras.layers.Dense(units) # input x weights
                        self.W2 = tf.keras.layers.Dense(units) # hidden states h weights
                        self.V = tf.keras.layers.Dense(1) # V
    
                    def call(self, features, hidden):
                        # hidden shape == (batch_size, hidden size)
                        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
                        # we are doing this to perform addition to calculate the score
                        hidden_with_time_axis = tf.expand_dims(hidden, 1)
    
                        # score shape == (batch_size, max_length, 1)
                        # we get 1 at the last axis because we are applying score to self.V
                        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
                        score = tf.nn.tanh(
                            self.W1(features) + self.W2(hidden_with_time_axis)) ## w[x, h]
                        # attention_weights shape == (batch_size, max_length, 1)
                        attention_weights = tf.nn.softmax(self.V(score), axis=1) ## v tanh(w[x,h])
    
                        # context_vector shape after sum == (batch_size, hidden_size)
                        context_vector = attention_weights * features ## attention_weights * x, right now the context_vector shape [batzh_size, max_length, hidden_size]
                        context_vector = tf.reduce_sum(context_vector, axis=1)
                        return context_vector, attention_weights
    
        
                sequence_input = Input(shape=(50,64))
                model = models.Sequential()
                reshape=layers.Reshape((2,25,64), input_shape=(input_size,64))(sequence_input)
                conv1=layers.TimeDistributed(layers.Conv1D(filters=32, kernel_size=5, strides=1 ))(reshape)
                maxpool1=layers.TimeDistributed(layers.MaxPooling1D(pool_size=3))(conv1)
                conv2=layers.TimeDistributed(layers.Conv1D(filters=12, kernel_size=3))(maxpool1)
                maxpool2=layers.TimeDistributed(layers.MaxPooling1D(pool_size=5))(conv2)
                flattenlayer=layers.TimeDistributed(layers.Flatten())(maxpool2)
                dropout = Dropout(0.8)(flattenlayer)
                #lstm = layers.Bidirectional(layers.LSTM(30, return_sequences = True), name="bi_lstm_0")(dropout)
    
                (lstm, forward_h, forward_c) = layers.LSTM(20,return_sequences=True,return_state=True)(dropout)
                state_h = forward_h
                state_c = forward_c
                context_vector, attention_weights = Attention(20)(lstm, state_h) # `lstm` the input features; `state_h` the hidden states from LSTM
                #dense1 = Dense(20, activation="relu")(context_vector)
                dropout = Dropout(0.8)(context_vector)
                output = Dense(n_labels, activation="softmax")(dropout)
                model = keras.Model(inputs=sequence_input, outputs=output)
                
                
                
            else:
                model = models.Sequential()
                model.add(tf.keras.layers.Reshape((2,25,64), input_shape=(input_size,64)))
                model.add(layers.TimeDistributed(layers.Conv1D(filters=64, kernel_size=5, strides=1 ),input_shape=(None,25,64)))
                model.add(layers.TimeDistributed(layers.MaxPooling1D(pool_size=3)))
                model.add(layers.TimeDistributed(layers.Conv1D(filters=32, kernel_size=3)))
                model.add(layers.TimeDistributed(layers.MaxPooling1D(pool_size=5)))
                model.add(layers.TimeDistributed(layers.Flatten()))
                model.add(layers.Dropout(0.5))
                model.add(layers.LSTM(30))
                model.add(layers.Dropout(0.7))
                model.add(layers.Dense(n_labels, activation= 'softmax' ))
                
                
            model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['acc',tfa.metrics.CohenKappa(num_classes=n_labels)])

            # Check summary of model    
            model.summary()

            print('------------------------------------------------------------------------')
            print(f'Training for fold {k}:')
            
            # Callbacks
            # ModelCheckPoint Callback
            checkpoint_filepath = models_dir+'RNN_model_check'
            model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_filepath,
                save_weights_only=True,
                monitor='loss',
                mode='auto',
                save_best_only=True)
            
            # Reduce Learning Rate on Plateau
            reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor = 'loss', min_delta=0.01, patience=3,factor=0.2)
            
            # Early stopping
            early_stop = keras.callbacks.EarlyStopping(patience=4, monitor = 'loss', min_delta=0.01)
            
            # Tensorboard
            run_logdir = get_run_logdir()
            tboard = keras.callbacks.TensorBoard(run_logdir)
            
            # Callbacks list
            callbacks_list = [model_checkpoint_callback,reduce_lr,early_stop,tboard]
            # Fit data to model
            history=model.fit(train_gen,epochs=100,steps_per_epoch=n_batches_train, 
                      callbacks=[early_stop, reduce_lr], verbose=1)
            
            #Save Model
                                                     
            models_dir='/home/mgutierrez/Desktop/RNN/RNN_AF_Driver_Prediction/Code/saved_model/my_model_fold_{0}'.format(k)

            with open(models_dir+'RNN_model_history.pkl', 'wb') as file_pi:
                    pickle.dump(history.history, file_pi)

            # Load CheckPoint Weights
            #model.load_weights(checkpoint_filepath)

            # Save final model 
            #*{Bug: no me deja guardar el modelo con attention, not implemented error}
            if not attention:
                model.save(models_dir+'RNN_model_history.h5', overwrite=True)
            
            #model.load_weights('checkpoint_filepath')
            scores_train=model.evaluate(train_gen,steps = n_batches_train)
            #model.load_weights('checkpoint_filepath').expect_partial()
            scores_test = model.evaluate(test_gen,steps = n_batches_test, verbose=2)
            
            print(scores_test)
            print('***************************************************************************')
            print(f'Score for fold Train {k}: {model.metrics_names[0]} of {scores_train[0]}; {model.metrics_names[1]} of {scores_train[1]*100}%')
            print(f'Score for fold Test {k}: {model.metrics_names[0]} of {scores_test[0]}; {model.metrics_names[1]} of {scores_test[1]*100}%')
            print('***************************************************************************')
            
            
            acc_per_fold_train.append(scores_train[1] * 100)
            loss_per_fold_train.append(scores_train[0] * 100)
            acc_per_fold.append(scores_test[1] * 100)
            loss_per_fold.append(scores_test[0]*100)
            
            # Confusion matrix
            # Predict on train set
            Y_orig_train = []
            Y_pred_train = []
            for i in range(0,n_batches_train):
                # Obtain batch
                batch = next(train_gen)
                # Get batch data
                x = batch[0]
                # Store original labels and predictions
                Y_orig_train.extend(list(batch[1]))
                Y_pred_train.extend(model.predict(x))

            # Predict on test set
            Y_orig_test = []
            Y_pred_test = []
            for i in range(0,n_batches_test):
                # Obtain batch
                batch = next(test_gen)
                # Get batch data
                x = batch[0]
                # Store original labels and predictions
                Y_orig_test.extend(list(batch[1]))
                Y_pred_test.extend(model.predict(x))
                
            cnf_matrix = confusion_matrix((np.argmax(Y_orig_train, axis=1)), np.argmax(Y_pred_train,axis=1))
            conf_matrix_train.extend([cnf_matrix])
            cnf_matrix = confusion_matrix((np.argmax(Y_orig_test, axis=1)), np.argmax(Y_pred_test,axis=1))
            conf_matrix_test.extend([cnf_matrix])
            
            y_true=np.argmax(Y_orig_test, axis=1)
            y_pred=np.argmax(Y_pred_test, axis=1)   
            classif_report=[classification_report(y_true, y_pred, output_dict=True)]
            classif_report_list.extend([classif_report])
            
            return acc_per_fold, loss_per_fold, acc_per_fold_train, loss_per_fold_train, conf_matrix_test, conf_matrix_train, classif_report_list
        
def generator_batches_RNN_CV(X,Y,Y_model,Mode, sequential,n_folds, shuffle_batches_train, shuffle_all_batches, input_size,n_labels = 8,fs=50, val=False):
    
    n_models=np.unique(Y_model)
    acc_per_fold,loss_per_fold = [],[]
    acc_per_fold_train,loss_per_fold_train = [],[]
    conf_matrix_test, conf_matrix_train = [], []
    classif_report_list=[]
    
    if Mode==1: # By SETS BSPS (Full BSPS signals)

        k = 1

        size_folds=round(len(n_models)/n_folds)
        all_indexes=list(range(1,130))
        folds_list=list(np.split(np.array(range(0,130)), n_folds))


        for K in range(0,n_folds):

            train_list=[]
            val_list=[]
            test_list=[]

            Test_index=folds_list[K]
            Train_index=list(set(all_indexes)-set(folds_list[K]))

             # Split signals in train and test
            train_sigs=X[np.where(np.in1d(Y_model, Train_index))]
            train_sigs=train_sigs.reshape(1,len(train_sigs), 64)
            test_sigs=X[np.where(np.in1d(Y_model, Test_index))]
            test_sigs=test_sigs.reshape(1,len(test_sigs), 64)

            # Same with labels
            train_labels=Y[np.where(np.in1d(Y_model, Train_index))]
            test_labels=Y[np.where(np.in1d(Y_model,   Test_index))]

            # Check the number of full signals chunks (depends on the input size)
            train_chunks=train_sigs.shape[1]//input_size
            test_chunks=test_sigs.shape[1]//input_size

            # Truncate signals depending on the input size
            train_sigs=np.split(train_sigs[:,0:input_size*train_chunks,:],train_chunks,axis=1)
            test_sigs=np.split(test_sigs[:,0:input_size*test_chunks,:],test_chunks,axis=1)

            # Same with labels (for each chunk, I pick the mode)
            train_labels=stats.mode(np.split(train_labels[0:input_size*train_chunks],train_chunks),axis=1)[0].ravel().tolist()
            test_labels=stats.mode(np.split(test_labels[0:input_size*test_chunks],test_chunks),axis=1)[0].ravel().tolist()


            print('Training:',Train_index)
            print('Training labels:',np.unique(train_labels))
            print('Testing:',Test_index)
            print('Test labels:',np.unique(test_labels))
            print('****************************************************+')
            
            # Associate signals with labels
            train_data=list(zip(train_sigs,keras.utils.np_utils.to_categorical(train_labels, num_classes=n_labels).reshape(train_chunks, 1,n_labels)))
            test_data=list(zip(test_sigs,keras.utils.np_utils.to_categorical(test_labels, num_classes=n_labels).reshape(test_chunks,1,n_labels)))

            # Store data in its corresponding list
            train_list.extend(train_data)
            #val_list.extend(val_data)
            test_list.extend(test_data)

            if shuffle_batches_train == True:
                train_list = shuffle(train_list)

            n_batches_train=len(train_list)
            n_batches_test=len(test_list)

            train_gen=aux_generator_RNN(train_list)
            test_gen=aux_generator_RNN(test_list)
           
            acc_per_fold, loss_per_fold, acc_per_fold_train, loss_per_fold_train, conf_matrix_test, conf_matrix_train, classif_report_list= model_CV(input_size, n_labels, n_batches_train, n_batches_test, train_gen,
             test_gen, acc_per_fold,loss_per_fold, acc_per_fold_train, loss_per_fold_train, conf_matrix_test, conf_matrix_train,classif_report_list,k)


            # Increase fold number
            k += 1
    


    elif Mode==2:
        
        # By models (each modelof BSP: train, test)

        # 1) Params
        n_folds=4
        input_size=50
        n_models=np.unique(Y_model)


        #4) Divide train/val/test

        k = 1

        for K in range(0,n_folds):
            train_list=[]
            val_list=[]
            test_list=[]

            for model in n_models:
                signals=X[np.where(Y_model==model)]
                labels=Y[np.where(Y_model==model)]
                len_fold= round(len(signals)/n_folds)
                len_fold=50

                if sequential:
                    test_index=list(range(0,K*len_fold+(len_fold)))
                    train_index=list(range(K*len_fold +len_fold ,K*len_fold+(len_fold)+len_fold))
                else:
                    test_index=list(range(K*len_fold,K*len_fold+(len_fold)))
                    train_index=list(set(list(range(1,len(signals)))) - set(test_index))
                    

                # Split signals in training, val and test
                train_sigs=signals[train_index,:].reshape((1,len(train_index),64))
                test_sigs=signals[test_index,:].reshape((1,len(test_index),64))

                # Same with labels
                train_labels=labels[train_index]
                test_labels=labels[test_index]

                # Check the number of full signals chunks (depends on the input size)
                train_chunks=len(train_index)//input_size
                test_chunks=len(test_index)//input_size

                # Truncate signals depending on the input size
                train_sigs=np.split(train_sigs[:,0:input_size*train_chunks,:],train_chunks,axis=1)
                test_sigs=np.split(test_sigs[:,0:input_size*test_chunks,:],test_chunks,axis=1)

                # Same with labels (for each chunk, I pick the mode)
                train_labels=stats.mode(np.split(train_labels[0:input_size*train_chunks],train_chunks),axis=1)[0].ravel().tolist()
                test_labels=stats.mode(np.split(test_labels[0:input_size*test_chunks],test_chunks),axis=1)[0].ravel().tolist()

                # Associate signals with labels
                train_data=list(zip(train_sigs,keras.utils.np_utils.to_categorical(train_labels, num_classes=n_labels).reshape(train_chunks, 1,n_labels)))
                test_data=list(zip(test_sigs,keras.utils.np_utils.to_categorical(test_labels, num_classes=n_labels).reshape(test_chunks,1,n_labels)))

                # Store data in its corresponding list
                train_list.extend(train_data)
                test_list.extend(test_data)


            if shuffle_batches_train == True:
                train_list = shuffle(train_list)

            n_batches_train=len(train_list)
            n_batches_test=len(test_list)

            train_gen=aux_generator_RNN(train_list)
            test_gen=aux_generator_RNN(test_list)

            train_gen=aux_generator_RNN(train_list)
            val_gen=aux_generator_RNN(val_list)  
            test_gen=aux_generator_RNN(test_list)

            acc_per_fold, loss_per_fold, acc_per_fold_train, loss_per_fold_train, conf_matrix_test, conf_matrix_train, classif_report_list= model_CV (input_size, n_labels, n_batches_train, n_batches_test, train_gen,
             test_gen, acc_per_fold,loss_per_fold, acc_per_fold_train,loss_per_fold_train, conf_matrix_test, conf_matrix_train,classif_report_list,k)

            # Increase fold number
            k += 1

    elif Mode==3:

        #Sequential cross val

        # 1) Params
        n_folds=5
        input_size=50
        shuffle_batches_train = True
        chunks= X.shape[0]//input_size

        #1) Reshape into batches of size 50 and truncate
        # Truncate signals depending on the input size
        truncated_sigs=np.array(np.array_split(X,chunks,axis=0))
        truncated_labels= np.array(np.array_split(Y,chunks,axis=0))
        truncated_models= np.array(np.array_split(Y_model,chunks,axis=0))


        #2) Shuffle signals, labels and models synchronously
        if shuffle_batches_train == True:
            temp = list(zip(truncated_sigs, truncated_labels, truncated_models ))
            random.shuffle(temp)
            tuple_sigs, tuple_labels, tuple_models = zip(*temp)
            # a and b come out as tuples, and so must be converted to lists.
            truncated_sigs, truncated_labels, truncated_models = np.array(tuple_sigs), np.array(tuple_labels), np.array(tuple_models)

        #3) Mode of each batch in labels
        truncated_labels=stats.mode(list(truncated_labels) ,axis=1)[0].ravel()

        #4) Divide train/val/test

        kfold = KFold(n_splits=n_folds, shuffle=False)
        k = 1
        acc_per_fold,loss_per_fold = [],[]
        acc_per_fold_train,loss_per_fold_train = [],[]

        for _,test_i in kfold.split(truncated_sigs,truncated_labels): 

            if test_i[0]==0: #first iteration: test 0-len_test, hence no training --> invalid
                continue
            else:
                train_i=np.array(range(0, test_i[0]-1)) #Always train at the beggining

            truncated_sigs=truncated_sigs.reshape(truncated_sigs.shape[0], 1, input_size, X.shape[1]) #Add dimension 1
            X_train = truncated_sigs[train_i]
            X_test = truncated_sigs[test_i]
            y_train, y_test = truncated_labels[train_i], truncated_labels[test_i]


            train_set=list(zip(X_train,
                               tf.keras.utils.to_categorical(y_train, num_classes=n_labels).reshape(len(train_i),1,n_labels)))

            test_set=list(zip(X_test,
                              tf.keras.utils.to_categorical(y_test, num_classes=n_labels).reshape(len(test_i),1,n_labels)))


            #Create generator
            train_gen=aux_generator_RNN(train_set)
            test_gen=aux_generator_RNN(test_set) 
            
            n_batches_train=len(train_set)
            n_batches_test=len(test_set)

            acc_per_fold, loss_per_fold, acc_per_fold_train, loss_per_fold_train, conf_matrix_test, conf_matrix_train, classif_report_list= model_CV(input_size, n_labels, n_batches_train, n_batches_test, train_gen,
             test_gen, acc_per_fold,loss_per_fold, acc_per_fold_train,loss_per_fold_train, conf_matrix_test, conf_matrix_train,classif_report_list,k)

            

            # Increase fold number
            k += 1

    elif Mode==4:
        #By batches 

        # 1) Params


        chunks= X.shape[0]//input_size

        #1) Reshape into batches of size 50 and truncate
        # Truncate signals depending on the input size
        truncated_sigs=np.array(np.array_split(X,chunks,axis=0))
        truncated_labels= np.array(np.array_split(Y,chunks,axis=0))
        truncated_models= np.array(np.array_split(Y_model,chunks,axis=0))


        #2) Shuffle signals, labels and models synchronously
        if shuffle_all_batches== True:
            temp = list(zip(truncated_sigs, truncated_labels, truncated_models ))
            random.shuffle(temp)
            tuple_sigs, tuple_labels, tuple_models = zip(*temp)
            # a and b come out as tuples, and so must be converted to lists.
            truncated_sigs, truncated_labels, truncated_models = np.array(tuple_sigs), np.array(tuple_labels), np.array(tuple_models)

        #3) Mode of each batch in labels
        truncated_labels=stats.mode(list(truncated_labels) ,axis=1)[0].ravel()

        #4) Divide train/val/test

        kfold = KFold(n_splits=n_folds, shuffle=False)
        k = 1
        acc_per_fold,loss_per_fold = [],[]
        acc_per_fold_train,loss_per_fold_train = [],[]

        for train_i,test_i in kfold.split(truncated_sigs,truncated_labels): 

            truncated_sigs=truncated_sigs.reshape(truncated_sigs.shape[0], 1,50, 64) #Add dimension 1
            X_train = truncated_sigs[train_i]
            X_test = truncated_sigs[test_i]
            y_train, y_test = truncated_labels[train_i], truncated_labels[test_i]


            train_set=list(zip(X_train,
                               tf.keras.utils.to_categorical(y_train, num_classes=n_labels).reshape(len(train_i),1,n_labels)))

            test_set=list(zip(X_test,
                              tf.keras.utils.to_categorical(y_test, num_classes=n_labels).reshape(len(test_i),1,n_labels)))

            n_batches_train=len(train_set)
            n_batches_test=len(test_set)
            #Create generator
            train_gen=aux_generator_RNN(train_set)
            test_gen=aux_generator_RNN(test_set) 


            acc_per_fold, loss_per_fold, acc_per_fold_train, loss_per_fold_train, conf_matrix_test, conf_matrix_train,classif_report_list= model_CV(input_size, n_labels, n_batches_train, n_batches_test, train_gen,
             test_gen, acc_per_fold,loss_per_fold, acc_per_fold_train,loss_per_fold_train, conf_matrix_test, conf_matrix_train,classif_report_list,k)
          

            # Increase fold number
            k += 1

    elif Mode==5:

    # By AF Models

        # 1) Params

        k = 1
        all_indexes=list(range(1,130))
        acc_per_fold,loss_per_fold = [],[]
        acc_per_fold_train,loss_per_fold_train = [],[]
        list_BSPS_models=list(range(1,131))
        x=10 #number of BSPS models in each fold
        aux_list= lambda list_BSPS_models, x: [list_BSPS_models[i:i+x] for i in range(0, len(list_BSPS_models), x)]
        fold_list_AF=aux_list(list_BSPS_models, x)
        if shuffle_all_batches:
            fold_list_AF=shuffle(fold_list_AF)

        x=3 #number of AF models in each fold
        aux_list= lambda fold_list_AF, x: [fold_list_AF[i:i+x] for i in range(0, len(fold_list_AF), x)]
        fold_list_models=(aux_list(fold_list_AF, x))
        folds_list=[]
        for i in fold_list_models:
            folds_list.append(np.ravel(i))

        for K in range(0,n_folds):

            train_list=[]
            val_list=[]
            test_list=[]

             # Split signals in train and test
            test_sigs=X[np.where(np.in1d(Y_model, folds_list[K]))]
            test_sigs=test_sigs.reshape(1,len(test_sigs), 64)
            train_sigs=X[np.where(np.in1d(Y_model, list(set(all_indexes)-set(folds_list[K]))))]
            train_sigs=train_sigs.reshape(1,len(train_sigs), 64)
            
            # Same with labels
            test_labels=Y[np.where(np.in1d(Y_model, folds_list[K]))]
            train_labels=Y[np.where(np.in1d(Y_model,   list(set(all_indexes)-set(folds_list[K]))))]
            print('Test',folds_list[K])
            print('Train',list(set(all_indexes)-set(folds_list[K])))
            print('Test labels',np.unique(test_labels))
            print('Train labels',np.unique(train_labels))

            # Check the number of full signals chunks (depends on the input size)
            train_chunks=train_sigs.shape[1]//input_size
            test_chunks=test_sigs.shape[1]//input_size

            # Truncate signals depending on the input size
            train_sigs=np.split(train_sigs[:,0:input_size*train_chunks,:],train_chunks,axis=1)
            test_sigs=np.split(test_sigs[:,0:input_size*test_chunks,:],test_chunks,axis=1)

            # Same with labels (for each chunk, I pick the mode)
            train_labels=stats.mode(np.split(train_labels[0:input_size*train_chunks],train_chunks),axis=1)[0].ravel().tolist()
            test_labels=stats.mode(np.split(test_labels[0:input_size*test_chunks],test_chunks),axis=1)[0].ravel().tolist()

            # Associate signals with labels
            train_data=list(zip(train_sigs,keras.utils.np_utils.to_categorical(train_labels, num_classes=n_labels).reshape(train_chunks, 1,n_labels)))
            test_data=list(zip(test_sigs,keras.utils.np_utils.to_categorical(test_labels, num_classes=n_labels).reshape(test_chunks,1,n_labels)))

            # Store data in its corresponding list
            train_list.extend(train_data)
            #val_list.extend(val_data)
            test_list.extend(test_data)

            if shuffle_batches_train == True:
                train_list = shuffle(train_list)

            n_batches_train=len(train_list)
            n_batches_test=len(test_list)

            train_gen=aux_generator_RNN(train_list)
            test_gen=aux_generator_RNN(test_list)


            acc_per_fold, loss_per_fold, acc_per_fold_train, loss_per_fold_train, conf_matrix_test, conf_matrix_train, classif_report_list= model_CV (input_size, n_labels, n_batches_train, n_batches_test, train_gen,
             test_gen, acc_per_fold,loss_per_fold, acc_per_fold_train,loss_per_fold_train, conf_matrix_test, conf_matrix_train,classif_report_list,k)

            

            # Increase fold number
            k += 1
            
    elif Mode==6: #AF MODELS LOO

    # By AF Models

        # 1) Params

        k = 1
        all_indexes=list(range(1,130))
       
        list_BSPS_models=list(range(1,131))
        x=10 #number of BSPS models in each fold
        aux_list= lambda list_BSPS_models, x: [list_BSPS_models[i:i+x] for i in range(0, len(list_BSPS_models), x)]
        fold_list_AF=aux_list(list_BSPS_models, x)
        if shuffle_all_batches:
            fold_list_AF=shuffle(fold_list_AF)

        x=1 #number of AF models in each fold
        aux_list= lambda fold_list_AF, x: [fold_list_AF[i:i+x] for i in range(0, len(fold_list_AF), x)]
        fold_list_models=(aux_list(fold_list_AF, x))
        folds_list=[]
        for i in fold_list_models:
            folds_list.append(np.ravel(i))
        print(folds_list)
        for K in range(0,len(folds_list)):

            train_list=[]
            val_list=[]
            test_list=[]

             # Split signals in train and test
            test_sigs=X[np.where(np.in1d(Y_model, folds_list[K]))]
            test_sigs=test_sigs.reshape(1,len(test_sigs), 64)
            train_sigs=X[np.where(np.in1d(Y_model, list(set(all_indexes)-set(folds_list[K]))))]
            train_sigs=train_sigs.reshape(1,len(train_sigs), 64)
            print('Test',folds_list[K])
            print('Train',list(set(all_indexes)-set(folds_list[K])))

            # Same with labels
            test_labels=Y[np.where(np.in1d(Y_model, folds_list[K]))]
            train_labels=Y[np.where(np.in1d(Y_model,   list(set(all_indexes)-set(folds_list[K]))))]

            # Check the number of full signals chunks (depends on the input size)
            train_chunks=train_sigs.shape[1]//input_size
            test_chunks=test_sigs.shape[1]//input_size

            # Truncate signals depending on the input size
            train_sigs=np.split(train_sigs[:,0:input_size*train_chunks,:],train_chunks,axis=1)
            test_sigs=np.split(test_sigs[:,0:input_size*test_chunks,:],test_chunks,axis=1)

            # Same with labels (for each chunk, I pick the mode)
            train_labels=stats.mode(np.split(train_labels[0:input_size*train_chunks],train_chunks),axis=1)[0].ravel().tolist()
            test_labels=stats.mode(np.split(test_labels[0:input_size*test_chunks],test_chunks),axis=1)[0].ravel().tolist()

            # Associate signals with labels
            train_data=list(zip(train_sigs,keras.utils.np_utils.to_categorical(train_labels, num_classes=n_labels).reshape(train_chunks, 1,n_labels)))
            test_data=list(zip(test_sigs,keras.utils.np_utils.to_categorical(test_labels, num_classes=n_labels).reshape(test_chunks,1,n_labels)))

            # Store data in its corresponding list
            train_list.extend(train_data)
            #val_list.extend(val_data)
            test_list.extend(test_data)

            if shuffle_batches_train == True:
                train_list = shuffle(train_list)

            n_batches_train=len(train_list)
            n_batches_test=len(test_list)

            train_gen=aux_generator_RNN(train_list)
            test_gen=aux_generator_RNN(test_list)


            acc_per_fold, loss_per_fold, acc_per_fold_train, loss_per_fold_train, conf_matrix_test, conf_matrix_train, classif_report_list= model_CV (input_size, n_labels, n_batches_train, n_batches_test, train_gen,
             test_gen, acc_per_fold,loss_per_fold, acc_per_fold_train,loss_per_fold_train, conf_matrix_test, conf_matrix_train,classif_report_list,k)

            

            # Increase fold number
            k += 1
        
    return acc_per_fold, loss_per_fold, acc_per_fold_train, loss_per_fold_train, conf_matrix_test, conf_matrix_train, classif_report_list
        
        
        
        
        
def aux_generator_RNN(list_data):
    while True:
        for batch in list_data:
            yield(batch)
        
#%% Generator for MLP and CNN
def generator_batches(X,Y,t_indexes, batch_size = 64, data_type='Tensor',SNR=20, imgAugm=False, tensor_type='3channel'):  
    #while True:
    if SNR != None:
        if type(SNR).__name__=='int':
            X_noisy_full=add_noise(X,SNR)
            X_data_batches=X_noisy_full[t_indexes,:]
            Y_data_batches=Y[t_indexes]
        elif type(SNR).__name__=='list':
            X_data_batches=[]
            Y_data_batches=[]
            for rate in SNR:
                X_noisy_full=add_noise(X,rate)
                X_data_batches.append(X_noisy_full[t_indexes,:])
                Y_data_batches.append(Y[t_indexes])                 
            X_data_batches=np.vstack(X_data_batches)
            Y_data_batches=np.concatenate(Y_data_batches)
    else:
        X_data_batches=X[t_indexes,:]
        Y_data_batches=Y[t_indexes]
    
       
    indexes_split=np.arange(0,X_data_batches.shape[0],batch_size)
    if indexes_split[-1]<(X_data_batches.shape[0]-1): 
        indexes_split=np.append(indexes_split,X_data_batches.shape[0]-1)    
    n_batches=len(indexes_split)-1
    
    batch_generator=set_generator(X_data_batches,Y_data_batches,indexes_split,data_type,imgAugm,tensor_type,batch_size)
    
    return n_batches, batch_generator

def set_generator(X_data_batches,Y_data_batches,indexes_split,data_type,imgAugm,tensor_type,batch_size):
    datagen = ImageDataGenerator(zoom_range=0.25,rotation_range=15,width_shift_range=0.15,height_shift_range=0.15)
    while True:
        if data_type!='Flat' and imgAugm==True:
            x_batch=[]
            y_batch=[]
            
            chosen_instant=np.random.randint(0,len(Y_data_batches),size=batch_size)
            x_img=get_tensor_model(X_data_batches[chosen_instant,:].T,tensor_type)
            y_label=keras.utils.to_categorical(Y_data_batches[chosen_instant],num_classes=8)
            for j in range(0,len(x_img)):
                if tensor_type=='1channel':
                    x_batch.append(preprocess_input(x_img[j,:,:],tensor_type))
                else:
                    x_batch.append(preprocess_input(x_img[j,:,:,:],tensor_type))
            DA_iter=datagen.flow(np.array(x_batch),y_label,batch_size)
            yield(next(DA_iter))
        else:      
            for i in range(0,len(indexes_split)-1):
                y_batch_prev=Y_data_batches[indexes_split[i]:indexes_split[i+1]]
                y_batch=keras.utils.to_categorical(y_batch_prev,num_classes=8)
                x_batch=X_data_batches[indexes_split[i]:indexes_split[i+1],:]
                if data_type=='Flat':
                    yield(x_batch, y_batch)
                else:
                    x_batch_prev=get_tensor_model(x_batch.T,tensor_type)
                    x_batch=[]
                    for j in range(0,len(x_batch_prev)):
                        if tensor_type=='1channel':
                            x_batch.append(preprocess_input(x_batch_prev[j,:,:],tensor_type))
                        elif tensor_type=='1channel_repeated':
                            input_temp=preprocess_input(x_batch_prev[j,:,:],tensor_type)
                            rep_temp=np.repeat(input_temp[:,:,:],3,2)
                            x_batch.append(rep_temp)
                        else:
                            x_batch.append(preprocess_input(x_batch_prev[j,:,:,:],tensor_type))
                    x_batch=np.array(x_batch)
                    yield(x_batch, y_batch)