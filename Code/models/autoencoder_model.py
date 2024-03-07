from keras import models, layers
import time
import tensorflow as tf
import keras
from keras.models import load_model
from keras.layers import BatchNormalization
from tensorflow.keras import datasets, layers, models, losses, Model


def autoencoder(x_train):
    '''
    Autoencoder model with Conv3D
    input_shape=(depth, height, width, channels)
    '''

    initializer = tf.keras.initializers.HeNormal()
    encoder = models.Sequential()
    encoder.add(layers.Conv3D(64, (2, 2, 2), strides=1, padding='same', activation='leaky_relu',
                              input_shape=(None, *x_train.shape[2:]), kernel_initializer=initializer,
                              kernel_regularizer=tf.keras.regularizers.l2(l=0.01)))
    encoder.add(layers.Conv3D(64, (2, 2, 2), strides=1, padding='same', activation='leaky_relu'))
    encoder.add(layers.Conv3D(32, (2, 2, 2), strides=1, padding='same', activation='leaky_relu'))
    encoder.add(layers.MaxPooling3D((1, 2, 2)))
    encoder.add(layers.Conv3D(12, (2, 2, 2), strides=1, padding='same', activation='leaky_relu',
                              kernel_regularizer=tf.keras.regularizers.l2(l=0.001)))
    encoder.add(layers.MaxPooling3D((1, 2, 2)))
    encoder.add(layers.Conv3D(12, (2, 2, 2), strides=1, padding='same',
                              activation='linear'))  # , kernel_regularizer =tf.keras.regularizers.l2( l=0.01)))
    encoder.add(layers.MaxPooling3D((1, 1, 2)))
    encoder.summary()

    decoder = models.Sequential()
    decoder.add(layers.Conv3D(12, (2, 2, 2), strides=1, padding='same', activation='leaky_relu',
                              input_shape=encoder.output.shape[1:]))
    decoder.add(layers.UpSampling3D((1, 1, 2)))
    decoder.add(layers.Conv3D(32, (2, 2, 2), strides=1, padding='same',
                              activation='leaky_relu'))  # kernel_regularizer =tf.keras.regularizers.l2( l=0.01)))
    decoder.add(layers.UpSampling3D((1, 2, 2)))
    decoder.add(layers.Conv3D(32, (2, 2, 2), strides=1, padding='same',
                              activation='leaky_relu'))  # , kernel_regularizer =tf.keras.regularizers.l2( l=0.01)))
    decoder.add(layers.UpSampling3D((1, 2, 2)))
    decoder.add(layers.Conv3D(1, (2, 2, 2), strides=1, padding='same', activation='linear',
                              kernel_regularizer=tf.keras.regularizers.l2(l=0.001)))
    decoder.summary()

    return encoder, decoder
