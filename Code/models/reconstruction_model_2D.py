import tensorflow as tf
from keras import models, layers
from keras.layers import BatchNormalization
from tensorflow.keras import layers, models


def reconstruction_model_2D(x_train_ls, y_train):
    '''
    Reconstruction model with Conv2D
    '''
    initializer = tf.keras.initializers.HeNormal()

    estimator =models.Sequential()
    estimator.add(layers.Conv2D(12, (2,2), strides=1, padding='same', activation='leaky_relu',
                                input_shape=x_train_ls.shape[1:], kernel_initializer=initializer))
    estimator.add(layers.UpSampling2D((2, 2)))
    #estimator.add(BatchNormalization())
    estimator.add(layers.Conv2D(12, (2,2), strides=1, padding='same', activation='leaky_relu',kernel_regularizer =tf.keras.regularizers.l1( l=0.1)))
    estimator.add(layers.UpSampling2D((2 ,2)))
    estimator.add(layers.Conv2D(3,(2,2), strides=1, padding='same', activation='leaky_relu', kernel_regularizer =tf.keras.regularizers.l1(l=0.01)))
    estimator.add(BatchNormalization())

    estimator.add(layers.Flatten())
    estimator.add(layers.Reshape((50, -1)))

    estimator.add(layers.TimeDistributed(layers.Flatten()))
    estimator.add(layers.LSTM(15, return_sequences=True))

    estimator.add(layers.Dense(540, activation='leaky_relu'))

    estimator.add(layers.Dropout(0.6))
    #estimator.add(layers.Dense(y_train_subsample.shape[1], activation='linear'))

#estimator.summary()