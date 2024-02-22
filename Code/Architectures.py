
from keras import models, layers
import time
import tensorflow as tf
import keras
from keras.models import load_model
from keras.layers import BatchNormalization
from tensorflow.keras import datasets, layers, models, losses, Model


def autoencoder_model(x_train):
    '''
    Autoencoder model with Conv3D
    '''
    
    initializer = tf.keras.initializers.HeNormal()
    encoder = models.Sequential()
    encoder.add(layers.Conv3D(64, (2, 2, 2), strides=1, padding='same', activation='leaky_relu',
    input_shape=(None, *x_train.shape[2:]),kernel_initializer=initializer, kernel_regularizer =tf.keras.regularizers.l2( l=0.01)))
    encoder.add(layers.Conv3D(64, (2, 2, 2), strides=1, padding='same', activation='leaky_relu'))
    encoder.add(layers.Conv3D(32, (2, 2, 2), strides=1, padding='same', activation='leaky_relu'))
    encoder.add(layers.MaxPooling3D((1, 2, 2)))
    encoder.add(layers.Conv3D(12, (2, 2, 2), strides=1, padding='same', activation='leaky_relu',kernel_regularizer =tf.keras.regularizers.l2( l=0.001)))
    encoder.add(layers.MaxPooling3D((1, 2, 2)))
    encoder.add(layers.Conv3D(12, (2, 2, 2), strides=1, padding='same', activation='linear'))#, kernel_regularizer =tf.keras.regularizers.l2( l=0.01)))
    encoder.add(layers.MaxPooling3D((1, 1, 2)))
    encoder.summary()

    decoder = models.Sequential()
    decoder.add(layers.Conv3D(12, (2,2, 2), strides=1, padding='same', activation='leaky_relu', input_shape=encoder.output.shape[1:]))
    decoder.add(layers.UpSampling3D((1, 1, 2)))
    decoder.add(layers.Conv3D(32, (2,2, 2), strides=1, padding='same', activation='leaky_relu'))# kernel_regularizer =tf.keras.regularizers.l2( l=0.01)))
    decoder.add(layers.UpSampling3D((1, 2, 2)))
    decoder.add(layers.Conv3D(32, (2,2, 2), strides=1, padding='same', activation='leaky_relu'))#, kernel_regularizer =tf.keras.regularizers.l2( l=0.01)))
    decoder.add(layers.UpSampling3D((1, 2,2)))
    decoder.add(layers.Conv3D(1, (2,2, 2), strides=1, padding='same', activation='linear', kernel_regularizer =tf.keras.regularizers.l2( l=0.001)))
    decoder.summary()

    return encoder, decoder


def autoencoder_model_2D(x_train):
    '''
    Autoencoder model with Conv2D
    '''
    
    initializer = tf.keras.initializers.HeNormal()
    encoder = models.Sequential()
    encoder.add(layers.Conv2D(64, (2, 2), strides=1, padding='same', activation='leaky_relu',
    input_shape=x_train.shape[1:],kernel_regularizer =tf.keras.regularizers.l2( l=0.001), kernel_initializer=initializer))
    encoder.add(layers.Conv2D(32, (2, 2), strides=1, padding='same', activation='leaky_relu'))
    encoder.add(layers.Conv2D(32, (2, 2), strides=1, padding='same', activation='leaky_relu'))
    encoder.add(layers.MaxPooling2D((2, 2)))
    encoder.add(layers.Conv2D(12, (2, 2), strides=1, padding='same', activation='leaky_relu'))
    encoder.add(layers.MaxPooling2D((2, 2)))
    encoder.add(layers.Conv2D(12, (2, 2), strides=1, padding='same', activation='linear',kernel_regularizer =tf.keras.regularizers.l2( l=0.001)))
    encoder.add(layers.MaxPooling2D((1, 2)))
    encoder.summary()


    decoder = models.Sequential()
    decoder.add(layers.Conv2D(12, (2,2), strides=1, padding='same', activation='leaky_relu', input_shape=encoder.output.shape[1:]))
    decoder.add(layers.UpSampling2D((1, 2)))
    decoder.add(layers.Dropout(0.2))
    decoder.add(layers.Conv2D(12, (2,2), strides=1, padding='same', activation='leaky_relu'))
    decoder.add(layers.UpSampling2D((2, 2)))
    decoder.add(layers.Dropout(0.2))
    decoder.add(layers.Conv2D(32, (2,2), strides=1, padding='same', activation='leaky_relu'))
    decoder.add(layers.UpSampling2D((2, 2)))
    decoder.add(layers.Dropout(0.2))
    decoder.add(layers.Conv2D(1, (2,2), strides=1, padding='same', activation='linear',kernel_regularizer =tf.keras.regularizers.l2( l=0.01)))
    decoder.summary()

    return encoder, decoder





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

def reconstruction_model_2D(x_train_ls, y_train):
    '''
    Reconstruction model with Conv2D
    '''
    initializer = tf.keras.initializers.HeNormal()

    estimator =models.Sequential()
    estimator.add(layers.Conv2D(12, (2,2), strides=1, padding='same', activation='leaky_relu',
                                input_shape=latent_vector_train.shape[1:], kernel_initializer=initializer))
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



def transfer_learning_AE (n_layers):
    '''Load pretrained model, freeze the last layers (decoder)'''
    pretrained_model= load_model('saved_var/sinusoid_pretr/conv_autoencoder.h5')

    #Freeze some layers from the pretrained model
    print('All layers',pretrained_model.layers)
    print('Selected layers',pretrained_model.layers[:-n_layers])

    for layer in pretrained_model.layers[:-n_layers]:
        layers.trainable = False
        
    #Create new model
    decoder = models.Sequential()
    decoder.add(pretrained_model)
    decoder.summary()
    return decoder

def Latent_Space_transfer_learning(pretrained_model):

    '''Obtain encoder part of the pretrained autoencoder'''

    model= models.Sequential()
    for layer in pretrained_model.layers[:-1]:
        model.add(layer)
    model.summary()
    return model


def reconstruction_BSPM_EGM(x_train, y_train):

    initializer = tf.keras.initializers.HeNormal()
    encoder = models.Sequential()
    encoder.add(layers.Conv2D(64, (2, 2), strides=1, padding='same', activation='leaky_relu',
    input_shape=x_train.shape[1:],kernel_regularizer =tf.keras.regularizers.l2( l=0.001), kernel_initializer=initializer))
    encoder.add(layers.Conv2D(32, (3, 3), strides=1, padding='same', activation='leaky_relu'))
    encoder.add(BatchNormalization())

    encoder.add(layers.Flatten())

    encoder.add(layers.Reshape((50, -1)))
    encoder.add(layers.TimeDistributed(layers.Flatten()))
    encoder.add(layers.LSTM(15, return_sequences=True))
    encoder.add(layers.Dense(512, activation='leaky_relu'))
    encoder.add(layers.Dropout(0.6))

    return encoder


class MultiOutput():
    """
    Used to generate our multi-output model. This CNN contains 2 branches, one for autoencoder, other for 
    regression from Bsps to EGMs. 
    """
    def build_autoencoder_branch(self, inputs, input_shape):
        """
        Used to optimize the BSPS feature extraction
        """

        initializer = tf.keras.initializers.HeNormal()
        encoder = layers.Conv3D(64, (2, 2, 2), strides=1, padding='same', activation='leaky_relu',
        input_shape = input_shape, kernel_initializer=initializer, kernel_regularizer =tf.keras.regularizers.l2( l=0.01))(inputs)
        encoder = layers.Conv3D(64, (2, 2, 2), strides=1, padding='same', activation='leaky_relu')(encoder)
        encoder = layers.Conv3D(32, (2, 2, 2), strides=1, padding='same', activation='leaky_relu')(encoder)
        encoder = layers.MaxPooling3D((1, 2, 2))(encoder)
        encoder = layers.Conv3D(12, (2, 2, 2), strides=1, padding='same', activation='leaky_relu',kernel_regularizer =tf.keras.regularizers.l2( l=0.001))(encoder)
        encoder = layers.MaxPooling3D((1, 2, 2))(encoder)
        encoder = layers.Conv3D(12, (2, 2, 2), strides=1, padding='same', activation='linear')(encoder)
        encoder = layers.MaxPooling3D((1, 1, 2))(encoder)
       
        decoder = layers.Conv3D(12, (2,2, 2), strides=1, padding='same', activation='leaky_relu')(encoder)
        decoder = layers.UpSampling3D((1, 1, 2))(decoder)
        decoder = layers.Conv3D(32, (2,2, 2), strides=1, padding='same', activation='leaky_relu')(decoder)
        decoder = layers.UpSampling3D((1, 2, 2))(decoder)
        decoder = layers.Conv3D(32, (2,2, 2), strides=1, padding='same', activation='leaky_relu')(decoder)
        decoder = layers.UpSampling3D((1, 2,2))(decoder)
        decoder = layers.Conv3D(1, (2,2, 2), strides=1, padding='same', activation='linear', kernel_regularizer =tf.keras.regularizers.l2( l=0.001))(decoder)

        return decoder
        
    def build_reconstruction_branch(self, inputs, input_shape):
        """
        Used to directly obtain EGMs from BSPS
        """
        initializer = tf.keras.initializers.HeNormal()

        encoder = layers.Conv3D(64, (2, 2, 2), strides=1, padding='same', activation='leaky_relu',
        input_shape = input_shape, kernel_initializer=initializer, kernel_regularizer =tf.keras.regularizers.l2( l=0.01))(inputs)
        encoder = layers.Conv3D(64, (2, 2, 2), strides=1, padding='same', activation='leaky_relu')(encoder)
        encoder = layers.Conv3D(32, (2, 2, 2), strides=1, padding='same', activation='leaky_relu')(encoder)
        encoder = layers.MaxPooling3D((1, 2, 2))(encoder)
        encoder = layers.Conv3D(12, (2, 2, 2), strides=1, padding='same', activation='leaky_relu',kernel_regularizer =tf.keras.regularizers.l2( l=0.001))(encoder)
        encoder = layers.MaxPooling3D((1, 2, 2))(encoder)
        encoder = layers.Conv3D(12, (2, 2, 2), strides=1, padding='same', activation='linear')(encoder)
        encoder = layers.MaxPooling3D((1, 1, 2))(encoder)

        x= layers.Conv2D(64, (2, 2), strides=1, padding='same', activation='leaky_relu',
        input_shape = input_shape, kernel_regularizer =tf.keras.regularizers.l2( l=0.001), kernel_initializer=initializer)(encoder)
        x = layers.Conv2D(32, (3, 3), strides=1, padding='same', activation='leaky_relu')(x)
        x = BatchNormalization()(x)
        x = layers.Flatten()(x)

        x = layers.Reshape((50, -1))(x)
        x = layers.TimeDistributed(layers.Flatten())(x)
        x = layers.LSTM(15, return_sequences=True)(x)
        x = layers.Dense(512, activation='leaky_relu')(x)
        x = layers.Dropout(0.6)(x)
        return x
    
    def assemble_full_model(self, input_shape):
        """
        Used to assemble our multi-output model CNN.
        """
        inputs = layers.Input(shape=input_shape)
        autoencoder_branch = self.build_autoencoder_branch(inputs, input_shape)
        reconstruction_branch = self.build_reconstruction_branch(inputs, input_shape)
        model = Model(inputs=inputs, outputs = [autoencoder_branch, reconstruction_branch], name="MultiOutput")
        return model
    


