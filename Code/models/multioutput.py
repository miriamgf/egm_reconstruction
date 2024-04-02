import tensorflow as tf
from keras import layers
from keras.layers import BatchNormalization
from tensorflow.keras import layers, Model
#referencia: https://towardsdatascience.com/building-a-multi-output-convolutional-neural-network-with-keras-ed24c7bc1178

class MultiOutput():
    """
    Used to generate our multi-output model. This CNN contains 2 branches, one for autoencoder, other for
    regression from Bsps to EGMs.
    """

    def build_encoder_module(self, inputs, input_shape):
        """
        Used to optimize the BSPS feature extraction
        """

        initializer = tf.keras.initializers.HeNormal()
        encoder = layers.Conv3D(64, (2, 2, 2), strides=1, padding='same', activation='leaky_relu',
                                input_shape=input_shape[2:], kernel_initializer=initializer,
                                kernel_regularizer=tf.keras.regularizers.l2(l=0.01))(inputs)
        encoder = layers.Conv3D(64, (2, 2, 2), strides=1, padding='same', activation='leaky_relu')(encoder)
        encoder = layers.Conv3D(32, (2, 2, 2), strides=1, padding='same', activation='leaky_relu')(encoder)
        encoder = layers.MaxPooling3D((1, 2, 2))(encoder)
        encoder = layers.Conv3D(12, (2, 2, 2), strides=1, padding='same', activation='leaky_relu',
                                kernel_regularizer=tf.keras.regularizers.l2(l=0.001))(encoder)
        encoder = layers.MaxPooling3D((1, 2, 2))(encoder)
        encoder = layers.Conv3D(12, (2, 2, 2), strides=1, padding='same', activation='linear')(encoder)
        encoder = layers.MaxPooling3D((1, 1, 2))(encoder)
        return encoder

    def build_decoder_module(self, inputs, input_shape, encoder):

        decoder = layers.Conv3D(12, (2, 2, 2), strides=1, padding='same', activation='leaky_relu')(encoder)
        decoder = layers.UpSampling3D((1, 1, 2))(decoder)
        decoder = layers.Conv3D(32, (2, 2, 2), strides=1, padding='same', activation='leaky_relu')(decoder)
        decoder = layers.UpSampling3D((1, 2, 2))(decoder)
        decoder = layers.Conv3D(32, (2, 2, 2), strides=1, padding='same', activation='leaky_relu')(decoder)
        decoder = layers.UpSampling3D((1, 2, 2))(decoder)
        decoder = layers.Conv3D(1, (2, 2, 2), strides=1, padding='same', activation='linear',
                                kernel_regularizer=tf.keras.regularizers.l2(l=0.001), name = 'Autoencoder_output')(decoder)

        return decoder
    def build_autoencoder_branch(self, inputs, input_shape):

        encoder = self.build_encoder_module(inputs, input_shape)
        decoder = self.build_decoder_module(inputs, input_shape, encoder)
        return decoder

    def build_reconstruction_branch(self, inputs, input_shape):
        """
        Used to directly obtain EGMs from BSPS
        """
        initializer = tf.keras.initializers.HeNormal()
        encoder = self.build_encoder_module(inputs, input_shape)

        x = layers.Conv2D(64, (2, 2), strides=1, padding='same', activation='leaky_relu',
                          input_shape=input_shape[2:], kernel_regularizer=tf.keras.regularizers.l2(l=0.001),
                          kernel_initializer=initializer)(encoder)
        x = layers.Conv2D(32, (3, 3), strides=1, padding='same', activation='leaky_relu')(x)
        x = BatchNormalization()(x)
        x = layers.Flatten()(x)
        x = layers.Reshape((50, -1))(x)
        x = layers.TimeDistributed(layers.Flatten())(x)
        x = layers.LSTM(15, return_sequences=True)(x)
        x = layers.Dropout(0.6)(x)
        x = layers.Dense(512, activation='leaky_relu', name = 'Regressor_output')(x)
        return x

    def assemble_full_model(self, input_shape):
        """
        Used to assemble our multi-output model CNN.
        """
        inputs = layers.Input(shape=input_shape)
        autoencoder_branch = self.build_autoencoder_branch(inputs, input_shape)
        reconstruction_branch = self.build_reconstruction_branch(inputs, input_shape)

        model = Model(inputs=inputs, outputs=[autoencoder_branch, reconstruction_branch], name="MultiOutput")
        return model

