import tensorflow as tf
from keras import models, layers
from tensorflow.keras import layers, models


def reconstruction(x_train_ls, y_train):
    """
    Reconstruction model with Conv3D
    """

    initializer = tf.keras.initializers.HeNormal()
    estimator = models.Sequential()
    estimator.add(
        layers.Conv3D(
            32,
            (2, 2, 2),
            strides=1,
            padding="same",
            activation="leaky_relu",
            input_shape=(x_train_ls.shape[1:]),
            kernel_initializer=initializer,
        )
    )

    estimator.add(layers.UpSampling3D((1, 2, 2)))
    # estimator.add(BatchNormalization())
    estimator.add(
        layers.Conv3D(
            16,
            (3, 3, 3),
            strides=1,
            padding="same",
            activation="leaky_relu",
            kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
        )
    )
    estimator.add(layers.UpSampling3D((1, 2, 2)))
    # estimator.add(BatchNormalization())
    estimator.add(
        layers.Conv3D(
            3,
            (1, 3, 3),
            strides=1,
            padding="valid",
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
        )
    )
    estimator.add(layers.TimeDistributed(layers.Flatten()))
    estimator.add(layers.BatchNormalization(axis=1))

    estimator.add(layers.LSTM(50, return_sequences=True))
    estimator.add(layers.Dropout(0.5))
    # estimator.add(layers.LSTM(20, return_sequences=True))
    # estimator.add(layers.Dropout(0.5))
    estimator.add(layers.Dense(y_train.shape[2], activation="linear"))
    estimator.summary()

    return estimator


# Define a custom callback to extract intermediate layer outputs
class IntermediateLayerOutputCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, layer_names):
        super(IntermediateLayerOutputCallback, self).__init__()
        self.model = model
        self.layer_names = layer_names
        self.outputs = {layer_name: [] for layer_name in layer_names}

    def on_epoch_end(self, epoch, logs=None):
        for layer_name in self.layer_names:
            intermediate_layer_model = tf.keras.Model(
                inputs=self.model.input, outputs=self.model.get_layer(layer_name).output
            )
            layer_output = intermediate_layer_model.predict(self.model.inputs)
            self.outputs[layer_name].append(layer_output)
