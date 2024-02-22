import tensorflow as tf
from keras import models, layers
from tensorflow.keras import layers, models

def reconstruction_model(x_train_ls, y_train):
    '''
    Reconstruction model with Conv3D
    '''

    initializer = tf.keras.initializers.HeNormal()
    estimator =models.Sequential()
    estimator.add(layers.Conv3D(64, (2,2,2), strides=1, padding='same', activation='leaky_relu',
                                input_shape=(x_train_ls.shape[1:]), kernel_initializer=initializer))

    estimator.add(layers.UpSampling3D((1,2, 2)))
    #estimator.add(BatchNormalization())
    estimator.add(layers.Conv3D(32, (3,3,3), strides=1, padding='same', activation='leaky_relu',kernel_regularizer =tf.keras.regularizers.l2 (l=0.01)))
    estimator.add(layers.UpSampling3D((1,2 ,2)))
    #estimator.add(BatchNormalization())
    estimator.add(layers.Conv3D(4, (3,3, 3), strides=1, padding='same', activation='leaky_relu', kernel_regularizer =tf.keras.regularizers.l2( l=0.01)))
    #estimator.add(BatchNormalization())
    estimator.add(layers.TimeDistributed(layers.Flatten()))
    estimator.add(layers.LSTM(20, return_sequences=True))
    estimator.add(layers.Dropout(0.5))
    #estimator.add(layers.LSTM(20, return_sequences=True))
    #estimator.add(layers.Dropout(0.5))
    estimator.add(layers.Dense(y_train.shape[2], activation='linear'))
    estimator.summary()
    return estimator
