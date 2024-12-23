import tensorflow as tf
from keras import layers
from keras import Model, layers

tf.config.experimental_run_functions_eagerly(True)


class MultiOutput_skip:
    """
    Used to generate our multi-output model. This CNN contains 2 branches, one for autoencoder, other for
    regression from Bsps to EGMs.
    """

    def __init__(self, params):
        self.params = params

    def build_encoder_module(self, inputs, input_shape):
        """
        Used to optimize the BSPS feature extraction and return the intermediate layers for skip connections.
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
            kernel_regularizer=tf.keras.regularizers.l2(
                l=self.params["l2_reg_encoder_1"]
            ),
        )(inputs)
        skip1 = encoder  # First skip connection
        encoder = layers.Conv3D(
            64, (5, 2, 2), strides=1, padding="same", activation="leaky_relu"
        )(encoder)
        skip2 = encoder  # Second skip connection
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
            kernel_regularizer=tf.keras.regularizers.l2(
                l=self.params["l2_reg_encoder_2"]
            ),
        )(encoder)
        encoder = layers.MaxPooling3D((1, 2, 2))(encoder)
        encoder = layers.Conv3D(
            4, (5, 2, 2), strides=1, padding="same", activation="linear"
        )(encoder)
        encoder = layers.MaxPooling3D((1, 1, 2))(encoder)
        skip3 = encoder  # Third skip connection
        return encoder, [skip1, skip2, skip3]

    def build_decoder_module(self, inputs, input_shape, encoder, skips):
        """
        Build the decoder module with skip connections. Skip connections are concatenated.
        """

        # Unpack skip connections
        skip1, skip2, skip3 = skips

        decoder = layers.Conv3D(
            12, (5, 2, 2), strides=1, padding="same", activation="leaky_relu"
        )(encoder)
        decoder = layers.Concatenate()([decoder, skip3])

        decoder = layers.UpSampling3D((1, 1, 2))(decoder)

        # Skip connection 1
        decoder = layers.Conv3D(
            32, (5, 2, 2), strides=1, padding="same", activation="leaky_relu"
        )(decoder)
        decoder = layers.UpSampling3D((1, 2, 2))(decoder)

        # Skip connection 2
        
        decoder = layers.Conv3D(
            32, (5, 2, 2), strides=1, padding="same", activation="leaky_relu"
        )(decoder)

        decoder = layers.UpSampling3D((1, 2, 2))(decoder)
        decoder = layers.Concatenate()([decoder, skip2])
        # Skip connection 3
        decoder = layers.Conv3D(
            1,
            (5, 2, 2),
            strides=1,
            padding="same",
            activation="linear",
            kernel_regularizer=tf.keras.regularizers.l2(
                l=self.params["l2_reg_decoder_1"]
            ),
            name="Autoencoder_output",
        )(decoder)

        return decoder

    def build_autoencoder_branch(self, inputs, input_shape):
        """
        Build the autoencoder branch with encoder and decoder using skip connections.
        """
        encoder, skips = self.build_encoder_module(inputs, input_shape)
        decoder = self.build_decoder_module(inputs, input_shape, encoder, skips)
        return encoder, decoder

    def build_reconstruction_branch(self, inputs, input_shape, encoder, n_nodes):
        """
        Build the reconstruction branch for regression tasks (Bsps to EGMs).
        """
        initializer = tf.keras.initializers.HeNormal()

        x = layers.Conv3D(
            64,
            (5, 2, 2),
            strides=(1, 1, 1),
            padding="same",
            activation="leaky_relu",
            kernel_regularizer=tf.keras.regularizers.l2(l=self.params["l2_reg_rec_1"]),
            kernel_initializer=initializer,
        )(encoder)
        x = layers.UpSampling3D((1, 2, 2))(x)
        x = layers.Conv3D(
            16,
            (5, 3, 3),
            strides=(1, 1, 1),
            padding="same",
            activation="leaky_relu",
            kernel_regularizer=tf.keras.regularizers.l2(l=self.params["l2_reg_rec_2"]),
        )(x)
        x = layers.UpSampling3D((1, 2, 2))(x)

        # Adjust temporal kernel to 1 to prevent changing temporal dimension
        x = layers.Conv3D(
            3, (5, 3, 3), strides=(1, 1, 1), padding="same", activation="leaky_relu"
        )(x)
        x = layers.TimeDistributed(layers.Flatten())(x)
        print("shape before normalization", x.shape)
        # x = BatchNormalization(axis=-1)(x)
        x = layers.LSTM(self.params["LSTM_units"], return_sequences=True)(x)
        x = layers.Dropout(self.params["dropout"])(x)
        x = layers.Dense(n_nodes, activation="leaky_relu", name="Regressor_output")(x)

        return x

    def assemble_full_model(self, input_shape, n_nodes):
        """
        Assemble the full model with the autoencoder (including skip connections) and reconstruction branch.
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
    
