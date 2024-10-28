import tensorflow as tf
from keras import models, layers
from tensorflow.keras import layers, models


def autoencoder_model_2D(x_train):
    """
    Autoencoder model with Conv2D
    """

    initializer = tf.keras.initializers.HeNormal()
    encoder = models.Sequential()
    encoder.add(
        layers.Conv2D(
            64,
            (2, 2),
            strides=1,
            padding="same",
            activation="leaky_relu",
            input_shape=x_train.shape[1:],
            kernel_regularizer=tf.keras.regularizers.l2(l=0.001),
            kernel_initializer=initializer,
        )
    )
    encoder.add(
        layers.Conv2D(32, (2, 2), strides=1, padding="same", activation="leaky_relu")
    )
    encoder.add(
        layers.Conv2D(32, (2, 2), strides=1, padding="same", activation="leaky_relu")
    )
    encoder.add(layers.MaxPooling2D((2, 2)))
    encoder.add(
        layers.Conv2D(12, (2, 2), strides=1, padding="same", activation="leaky_relu")
    )
    encoder.add(layers.MaxPooling2D((2, 2)))
    encoder.add(
        layers.Conv2D(
            12,
            (2, 2),
            strides=1,
            padding="same",
            activation="linear",
            kernel_regularizer=tf.keras.regularizers.l2(l=0.001),
        )
    )
    encoder.add(layers.MaxPooling2D((1, 2)))
    encoder.summary()

    decoder = models.Sequential()
    decoder.add(
        layers.Conv2D(
            12,
            (2, 2),
            strides=1,
            padding="same",
            activation="leaky_relu",
            input_shape=encoder.output.shape[1:],
        )
    )
    decoder.add(layers.UpSampling2D((1, 2)))
    decoder.add(layers.Dropout(0.2))
    decoder.add(
        layers.Conv2D(12, (2, 2), strides=1, padding="same", activation="leaky_relu")
    )
    decoder.add(layers.UpSampling2D((2, 2)))
    decoder.add(layers.Dropout(0.2))
    decoder.add(
        layers.Conv2D(32, (2, 2), strides=1, padding="same", activation="leaky_relu")
    )
    decoder.add(layers.UpSampling2D((2, 2)))
    decoder.add(layers.Dropout(0.2))
    decoder.add(
        layers.Conv2D(
            1,
            (2, 2),
            strides=1,
            padding="same",
            activation="linear",
            kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
        )
    )
    decoder.summary()

    return encoder, decoder
