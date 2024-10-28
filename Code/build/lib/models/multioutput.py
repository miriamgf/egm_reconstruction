import tensorflow as tf
from keras import layers
from keras.layers import BatchNormalization
from tensorflow.keras import layers, Model

# referencia: https://towardsdatascience.com/building-a-multi-output-convolutional-neural-network-with-keras-ed24c7bc1178


class MultiOutput:
    """
    Used to generate our multi-output model. This CNN contains 2 branches, one for autoencoder, other for
    regression from Bsps to EGMs.
    """

    def build_encoder_module(self, inputs, input_shape):
        """
        Used to optimize the BSPS feature extraction
        """

        initializer = tf.keras.initializers.HeNormal()
        encoder = layers.Conv3D(
            64,
            (5, 2, 2),
            strides=1,
            padding="same",
            activation="leaky_relu",
            input_shape=input_shape[2:],
            kernel_initializer=initializer,
            kernel_regularizer=tf.keras.regularizers.l2(l=0.1),
        )(inputs)
        encoder = layers.Conv3D(
            64, (5, 2, 2), strides=1, padding="same", activation="leaky_relu"
        )(encoder)
        encoder = layers.Conv3D(
            32, (5, 2, 2), strides=1, padding="same", activation="leaky_relu"
        )(encoder)
        encoder = layers.MaxPooling3D((1, 2, 2))(encoder)
        encoder = layers.Conv3D(
            12,
            (5, 2, 2),
            strides=1,
            padding="same",
            activation="leaky_relu",
            kernel_regularizer=tf.keras.regularizers.l2(l=0.1),
        )(encoder)
        encoder = layers.MaxPooling3D((1, 2, 2))(encoder)
        encoder = layers.Conv3D(
            12, (5, 2, 2), strides=1, padding="same", activation="linear"
        )(encoder)
        encoder = layers.MaxPooling3D((1, 1, 2))(encoder)
        return encoder

    def build_decoder_module(self, inputs, input_shape, encoder):

        decoder = layers.Conv3D(
            12, (5, 2, 2), strides=1, padding="same", activation="leaky_relu"
        )(encoder)
        decoder = layers.UpSampling3D((1, 1, 2))(decoder)
        decoder = layers.Conv3D(
            32, (5, 2, 2), strides=1, padding="same", activation="leaky_relu"
        )(decoder)
        decoder = layers.UpSampling3D((1, 2, 2))(decoder)
        decoder = layers.Conv3D(
            32, (5, 2, 2), strides=1, padding="same", activation="leaky_relu"
        )(decoder)
        decoder = layers.UpSampling3D((1, 2, 2))(decoder)
        decoder = layers.Conv3D(
            1,
            (5, 2, 2),
            strides=1,
            padding="same",
            activation="linear",
            kernel_regularizer=tf.keras.regularizers.l2(l=0.1),
            name="Autoencoder_output",
        )(decoder)

        return decoder

    def build_autoencoder_branch(self, inputs, input_shape):

        encoder = self.build_encoder_module(inputs, input_shape)
        decoder = self.build_decoder_module(inputs, input_shape, encoder)
        return encoder, decoder

    def build_reconstruction_branch(self, inputs, input_shape, encoder, n_nodes):
        initializer = tf.keras.initializers.HeNormal()

        x = layers.Conv3D(
            64,
            (5, 2, 2),
            strides=(1, 1, 1),
            padding="same",
            activation="leaky_relu",
            input_shape=input_shape[1:],
            kernel_regularizer=tf.keras.regularizers.l2(l=0.1),
            kernel_initializer=initializer,
        )(encoder)
        x = layers.UpSampling3D((1, 2, 2))(x)
        x = layers.Conv3D(
            32,
            (5, 3, 3),
            strides=(1, 1, 1),
            padding="same",
            activation="leaky_relu",
            kernel_regularizer=tf.keras.regularizers.l2(l=0.1),
        )(x)
        x = layers.UpSampling3D((1, 2, 2))(x)
        # Ajusta el kernel temporal a 1 para evitar cambio en la dimensión temporal
        x = layers.Conv3D(
            3, (5, 3, 3), strides=(1, 1, 1), padding="same", activation="leaky_relu"
        )(x)
        x = layers.TimeDistributed(layers.Flatten())(x)
        x = BatchNormalization(axis=1)(x)
        x = layers.LSTM(50, return_sequences=True)(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(n_nodes, activation="leaky_relu", name="Regressor_output")(x)

        return x

    def assemble_full_model(self, input_shape, n_nodes):
        """
        Used to assemble our multi-output model CNN.
        """
        inputs = layers.Input(shape=input_shape)
        encoder, autoencoder_branch = self.build_autoencoder_branch(inputs, input_shape)
        reconstruction_branch = self.build_reconstruction_branch(
            inputs, input_shape, encoder, n_nodes
        )

        model = Model(
            inputs=inputs,
            outputs=[autoencoder_branch, reconstruction_branch],
            name="MultiOutput",
        )
        return model
