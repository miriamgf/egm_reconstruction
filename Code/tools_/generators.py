# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 18:18:11 2021

@author: Miguel Ángel
"""

import sys, os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import tensorflow as tf
import keras
import numpy as np
from sklearn.model_selection import train_test_split
from numpy.random import default_rng
if keras.__version__ > '2.15.0':
    from keras.src.legacy.preprocessing.image import ImageDataGenerator
else:
    from keras.preprocessing.image import ImageDataGenerator

from scipy import stats
from sklearn.utils import shuffle

# %% Path Models
current = os.path.dirname(os.path.realpath(__file__))
directory = str(os.path.dirname(current)) + "/Data/"
torsos_dir = str(os.path.dirname(current)) + "/Labeled_torsos/"


# %% Train-test split (by time and by model)
def train_test_val_split(
    Y_model, split_type="Time", test_size=0.2, validation_size=0.2, seed=None
):
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
    t_indexes = np.arange(0, len(Y_model))
    if split_type == "Splitted":
        # Check the number of models
        n_models = np.unique(Y_model)

        # Create train-val-test lists
        train_list = []
        val_list = []
        test_list = []

        # Split time instants in train-val-test
        start_value = 0
        for model in n_models:
            labels = Y_model[np.where(Y_model == model)]

            # Check length of signals
            len_sig = len(labels)
            len_test = int(np.floor(test_size * len_sig))
            len_train_temp = len_sig - len_test
            len_val = int(np.floor(validation_size * len_train_temp))
            len_train = len_train_temp - len_val

            train_vector = np.arange(start_value, start_value + len_train)
            val_vector = np.arange(
                start_value + len_train, start_value + len_train + len_val
            )
            test_vector = np.arange(
                start_value + len_train + len_val,
                start_value + len_train + len_val + len_test,
            )

            start_value += len_sig

            train_list.extend(train_vector)
            val_list.extend(val_vector)
            test_list.extend(test_vector)

        t_train = np.sort(np.array(train_list).ravel())
        np.random.shuffle(t_train)
        t_val = np.sort(np.array(val_list).ravel())
        t_test = np.sort(np.array(test_list).ravel())

    elif split_type == "Time":
        t_train_val, t_test = train_test_split(
            t_indexes, test_size=test_size, random_state=seed
        )
        t_train, t_val = train_test_split(
            t_train_val, test_size=validation_size, random_state=seed
        )

    else:
        if seed != None:
            rng = default_rng(seed)
        else:
            rng = default_rng()

        # Check the number of models.
        y_model_unique = np.unique(Y_model)

        # Select the models that will go to test
        n_models_test = int(np.ceil(len(y_model_unique) * (test_size)))
        models_to_test = set(
            (sorted(rng.choice(y_model_unique, size=n_models_test, replace=False)))
        )

        # Obtain the models that will go to training/val and test sets
        y_model_unique = set(y_model_unique.tolist())
        models_to_train_val = y_model_unique - models_to_test
        models_to_train_val = np.array(list(models_to_train_val))
        models_to_test = np.array(list(models_to_test))

        # Get the time instants associated with training/val and test sets.
        t_test = []
        for n_model in models_to_test:
            t_values = t_indexes[np.where(Y_model == n_model)]
            t_test.extend(t_values)

        t_train_val = set(t_indexes.tolist()) - set(t_test)

        t_test = np.array(t_test)
        t_train_val = np.array(list(t_train_val))

        # Obtain the indexes of the training and validation sets
        t_train, t_val = train_test_split(
            t_train_val, test_size=validation_size, random_state=seed
        )

    return t_train, t_val, t_test


def generator_Autoencoder(x_train, x_test, x_val):
    """Miriam Gutierrez 2023"""

    def train_gen(x_train):
        while True:
            for i in range(0, len(x_train)):
                sig = np.expand_dims(x_train[i], axis=0)
                sig_list = [sig, sig]
                yield sig_list

    def test_gen(x_test):
        while True:
            for i in range(0, len(x_test)):
                sig = np.expand_dims(x_test[i], axis=0)
                sig_list = [sig, sig]
                yield sig_list

    def val_gen(x_val):
        while True:
            for i in range(0, len(x_val)):
                sig = np.expand_dims(x_test[i], axis=0)
                sig_list = [sig, sig]
                yield sig_list

    return train_gen, test_gen, val_gen, len(x_train), len(x_test), len(x_val)


def generator_reconstruction(
    y_train, latent_vector_train, y_test, latent_vector_test, y_val, latent_vector_val
):
    """Miriam GUtierrez 2023"""

    def train_gen(latent_vector_train, y_train):
        while True:
            for i in range(1, len(latent_vector_train)):
                # add 1 dimension at the beggining
                sig1 = np.expand_dims(latent_vector_train[i], axis=0)
                sig2 = np.expand_dims(y_train[i], axis=0)
                sig_list = [sig1, y_train[i]]
                yield sig_list

    def test_gen(latent_vector_test, y_test):
        while True:
            for i in range(1, len(latent_vector_test)):
                sig1 = np.expand_dims(latent_vector_test[i], axis=0)
                sig2 = np.expand_dims(y_test[i], axis=0)
                sig_list = [sig1, y_test[i]]
                yield sig_list

    def val_gen(latent_vector_val, y_val):
        while True:
            for i in range(1, len(latent_vector_val)):
                sig1 = np.expand_dims(latent_vector_val[i], axis=0)
                sig2 = np.expand_dims(y_val[i], axis=0)
                sig_list = [sig1, y_val[i]]
                yield sig_list

    return (
        train_gen,
        test_gen,
        val_gen,
        len(latent_vector_train),
        len(latent_vector_test),
        len(latent_vector_val),
    )


from keras.utils import Sequence


class DataGen(Sequence):
    def __init__(self, inputs, targets, batch_size):
        self.inputs = inputs
        self.targets = targets
        self.batch_size = batch_size

    def __len__(self):
        # Return the number of batches in the dataset
        return len(self.inputs) // self.batch_size

    def __getitem__(self, index):
        # Generate one batch of data
        batch_inputs = self.inputs[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        batch_targets = self.targets[
            index * self.batch_size : (index + 1) * self.batch_size
        ]

        # Preprocess the batch if needed
        # ...

        return batch_inputs, batch_targets


# %% Add noise to signals
def add_noise(X, SNR=20):

    X_noisy, _, _ = addwhitenoise(X.T, SNR=SNR)
    # Normalizar
    mm = np.mean(X_noisy, axis=1)
    ss = np.std(X_noisy, axis=1)
    X_noisy_normalized = (X_noisy - mm[:, np.newaxis]) / ss[:, np.newaxis]

    return X_noisy_normalized.T


# %% Interpolate tensor
def preprocess_input(x_tensor, tensor_type):
    if tensor_type == "3channelTensor":
        x_reshaped = tf.cast(
            np.reshape(
                x_tensor, (1, x_tensor.shape[0], x_tensor.shape[1], x_tensor.shape[2])
            ),
            tf.float32,
        )
        x_interpolated = tf.keras.layers.UpSampling2D(
            size=(10, 10), interpolation="bilinear"
        )(x_reshaped)
    else:
        x_reshaped = tf.cast(
            np.reshape(x_tensor, (1, x_tensor.shape[0], x_tensor.shape[1], 1)),
            tf.float32,
        )
        x_interpolated = tf.keras.layers.UpSampling2D(
            size=(10, 10), interpolation="bilinear"
        )(x_reshaped)

    return x_interpolated.numpy()[0, :, :, :]


# %% Generator for RNN
def generator_batches_RNN(
    X,
    Y,
    Y_model,
    data_type="Flat",
    val_percentage=0.2,
    test_percentage=0.2,
    input_size=50,
    SNR=20,
    number_classes=3,
    val=True,
    shuffle_batches_train=True,
):
    # Check the number of models
    n_models = np.unique(Y_model)  # 0-139?

    # Create train-val-test lists
    train_list = []
    val_list = []
    test_list = []
    count = 0
    # Store signals
    for model in n_models:
        count += 1
        signals = X[np.where(Y_model == model)]
        labels = Y[np.where(Y_model == model)]

        # Check length of signals
        len_sig = signals.shape[0]
        len_test = int(np.floor(test_percentage * len_sig))
        if len_test < input_size:
            len_test = input_size

        len_val = int(np.floor(val_percentage * len_sig))
        if len_val < input_size:
            len_val = input_size

        len_train = len_sig - len_val - len_test
        if len_train < input_size:
            raise ValueError("The training size is less than the input size")

        # Split signals in training, val and test
        if data_type == "Flat":
            train_sigs = signals[0:len_train, :].reshape((len_train, 64, 1))
            val_sigs = signals[len_train : len_train + len_val, :].reshape(
                (len_val, 64, 1)
            )
            test_sigs = signals[
                len_train + len_val : len_train + len_val + len_test, :
            ].reshape((len_test, 64, 1))
        elif data_type == "1channelTensor":
            train_sigs = signals[0:len_train, :].reshape((len_train, 6, 16, 1))
            val_sigs = signals[len_train : len_train + len_val, :].reshape(
                (len_val, 6, 16, 1)
            )
            test_sigs = signals[
                len_train + len_val : len_train + len_val + len_test, :
            ].reshape((len_test, 6, 16, 1))
        elif data_type == "3channelTensor":
            train_sigs = signals[0:len_train, :].reshape((len_train, 6, 4, 3))
            val_sigs = signals[len_train : len_train + len_val, :].reshape(
                (len_val, 6, 4, 3)
            )
            test_sigs = signals[
                len_train + len_val : len_train + len_val + len_test, :
            ].reshape((len_test, 6, 4, 3))
        else:
            print("Incorrect data type.")

        # Same with labels
        train_labels = labels[0:len_train]
        val_labels = labels[len_train : len_train + len_val]
        test_labels = labels[len_train + len_val : len_train + len_val + len_test]

        # Check the number of full signals chunks (depends on the input size)
        train_chunks = len_train // input_size
        val_chunks = len_val // input_size
        test_chunks = len_test // input_size

        # Truncate signals depending on the input size
        if data_type == "Flat":
            train_sigs = np.split(
                train_sigs[0 : input_size * train_chunks, :, :], train_chunks, axis=1
            )
            val_sigs = np.split(
                val_sigs[0 : input_size * val_chunks, :, :], val_chunks, axis=1
            )
            test_sigs = np.split(
                test_sigs[0 : input_size * test_chunks, :, :], test_chunks, axis=1
            )
        elif data_type == "1channelTensor":
            train_sigs = np.split(
                train_sigs[0 : input_size * train_chunks, :, :, :], train_chunks, axis=0
            )
            val_sigs = np.split(
                val_sigs[0 : input_size * val_chunks, :, :, :], val_chunks, axis=0
            )
            test_sigs = np.split(
                test_sigs[0 : input_size * test_chunks, :, :, :], test_chunks, axis=0
            )
        elif data_type == "3channelTensor":
            train_sigs = np.split(
                train_sigs[0 : input_size * train_chunks, :, :, :], train_chunks, axis=0
            )
            val_sigs = np.split(
                val_sigs[0 : input_size * val_chunks, :, :, :], val_chunks, axis=0
            )
            test_sigs = np.split(
                test_sigs[0 : input_size * test_chunks, :, :, :], test_chunks, axis=0
            )
        else:
            print("Incorrect data type.")

        # Same with labels (for each chunk, I pick the mode)
        train_labels = (
            stats.mode(
                np.split(train_labels[0 : input_size * train_chunks], train_chunks),
                axis=1,
            )[0]
            .ravel()
            .tolist()
        )
        val_labels = (
            stats.mode(
                np.split(val_labels[0 : input_size * val_chunks], val_chunks), axis=1
            )[0]
            .ravel()
            .tolist()
        )
        test_labels = (
            stats.mode(
                np.split(test_labels[0 : input_size * test_chunks], test_chunks), axis=1
            )[0]
            .ravel()
            .tolist()
        )

        # Associate signals with labels DUDA num_classes=8?
        train_data = list(
            zip(
                train_sigs,
                tf.keras.utils.to_categorical(
                    train_labels, num_classes=number_classes
                ).reshape(train_chunks, 1, number_classes),
            )
        )
        val_data = list(
            zip(
                val_sigs,
                tf.keras.utils.to_categorical(
                    val_labels, num_classes=number_classes
                ).reshape(val_chunks, 1, number_classes),
            )
        )
        test_data = list(
            zip(
                test_sigs,
                tf.keras.utils.to_categorical(
                    test_labels, num_classes=number_classes
                ).reshape(test_chunks, 1, number_classes),
            )
        )

        # Store data in its corresponding list
        train_list.extend(train_data)
        val_list.extend(val_data)
        test_list.extend(test_data)

    if val == True:
        if shuffle_batches_train == True:
            train_list = shuffle(train_list)

        n_batches_train = len(train_list)
        n_batches_val = len(val_list)
        n_batches_test = len(test_list)

        train_gen = aux_generator_RNN(train_list, data_type)
        val_gen = aux_generator_RNN(val_list, data_type)
        test_gen = aux_generator_RNN(test_list, data_type)

        return (
            n_batches_train,
            n_batches_val,
            n_batches_test,
            train_gen,
            val_gen,
            test_gen,
        )
    else:
        train_list = train_list + val_list

        if shuffle_batches_train == True:
            train_list = shuffle(train_list)

        n_batches_train = len(train_list)
        n_batches_test = len(test_list)

        train_gen = aux_generator_RNN(train_list, data_type)
        test_gen = aux_generator_RNN(test_list, data_type)

        return n_batches_train, n_batches_test, train_gen, test_gen


def generator_batches_autoencoder(
    X,
    Y,
    Y_model,
    data_type="Flat",
    val_percentage=0.2,
    test_percentage=0.2,
    input_size=50,
    SNR=20,
    val=True,
    shuffle_batches_train=True,
):
    # Check the number of models
    n_models = np.unique(Y_model)  # 0-139?

    # Create train-val-test lists
    train_list = []
    val_list = []
    test_list = []
    count = 0

    # Check length of signals
    len_sig = X.shape[0]
    len_test = int(np.floor(test_percentage * len_sig))
    len_val = int(np.floor(val_percentage * len_sig))
    len_train = len_sig - len_val - len_test
    signals = X

    # Split signals in training, val and test
    if data_type == "Flat":
        train_sigs = signals[0:len_train, :].reshape((len_train, 64, 1))
        val_sigs = signals[len_train : len_train + len_val, :].reshape((len_val, 64, 1))
        test_sigs = signals[
            len_train + len_val : len_train + len_val + len_test, :
        ].reshape((len_test, 64, 1))
    elif data_type == "1channelTensor":
        train_sigs = signals[0:len_train, :].reshape((len_train, 6, 16, 1))
        val_sigs = signals[len_train : len_train + len_val, :].reshape(
            (len_val, 6, 16, 1)
        )
        test_sigs = signals[
            len_train + len_val : len_train + len_val + len_test, :
        ].reshape((len_test, 6, 16, 1))
    elif data_type == "3channelTensor":
        train_sigs = signals[0:len_train, :].reshape((len_train, 6, 4, 3))
        val_sigs = signals[len_train : len_train + len_val, :].reshape(
            (len_val, 6, 4, 3)
        )
        test_sigs = signals[
            len_train + len_val : len_train + len_val + len_test, :
        ].reshape((len_test, 6, 4, 3))
    else:
        print("Incorrect data type.")

    # Check the number of full signals chunks (depends on the input size)
    train_chunks = len_train // input_size
    val_chunks = len_val // input_size
    test_chunks = len_test // input_size

    # Truncate signals depending on the input size
    if data_type == "Flat":
        train_sigs = np.split(
            train_sigs[0 : input_size * train_chunks, :, :], train_chunks, axis=1
        )
        val_sigs = np.split(
            val_sigs[0 : input_size * val_chunks, :, :], val_chunks, axis=1
        )
        test_sigs = np.split(
            test_sigs[0 : input_size * test_chunks, :, :], test_chunks, axis=1
        )
    elif data_type == "1channelTensor":
        train_sigs = np.split(
            train_sigs[0 : input_size * train_chunks, :, :, :], train_chunks, axis=0
        )
        val_sigs = np.split(
            val_sigs[0 : input_size * val_chunks, :, :, :], val_chunks, axis=0
        )
        test_sigs = np.split(
            test_sigs[0 : input_size * test_chunks, :, :, :], test_chunks, axis=0
        )
    elif data_type == "3channelTensor":
        train_sigs = np.split(
            train_sigs[0 : input_size * train_chunks, :, :, :], train_chunks, axis=0
        )
        val_sigs = np.split(
            val_sigs[0 : input_size * val_chunks, :, :, :], val_chunks, axis=0
        )
        test_sigs = np.split(
            test_sigs[0 : input_size * test_chunks, :, :, :], test_chunks, axis=0
        )
    else:
        print("Incorrect data type.")

    train_list = list(train_sigs)
    test_list = list(test_sigs)
    val_list = list(val_sigs)

    if val == True:
        if shuffle_batches_train == True:
            train_list = shuffle(train_list)

        n_batches_train = len(train_list)
        n_batches_val = len(val_list)
        n_batches_test = len(test_list)

        train_gen = aux_generator_RNN(train_list, data_type)
        val_gen = aux_generator_RNN(val_list, data_type)
        test_gen = aux_generator_RNN(test_list, data_type)

        return (
            n_batches_train,
            n_batches_val,
            n_batches_test,
            train_gen,
            val_gen,
            test_gen,
            train_list,
        )
    else:
        train_list = train_list + val_list

        if shuffle_batches_train == True:
            train_list = shuffle(train_list)

        n_batches_train = len(train_list)
        n_batches_test = len(test_list)

        train_gen = aux_generator_RNN(train_list, data_type)
        test_gen = aux_generator_RNN(test_list, data_type)

        return n_batches_train, n_batches_test, train_gen, test_gen, train_list


def generator_batches_RNN_kfold(
    X,
    Y,
    Y_model,
    indices,
    data_type="Flat",
    input_size=50,
    SNR=20,
    number_classes=3,
    shuffle_batches_train=True,
):

    # Create set lists
    samples_list = []
    count = 0

    # Store signals
    for model in indices:
        count += 1
        sigs = X[np.where(Y_model == model)]
        labels = Y[np.where(Y_model == model)]

        len_sig = sigs.shape[0]
        len_split = len_sig // input_size

        # Split signals in training, val and test
        if data_type == "Flat":
            sigs = sigs.reshape((len_sig, X.shape[1], 1))
        elif data_type == "1channelTensor":
            sigs = sigs.reshape((len_sig, 6, 16, 1))
        elif data_type == "3channelTensor":
            sigs = sigs.reshape((len_sig, 6, 4, 3))
        else:
            print("Incorrect data type.")

        # Truncate signals depending on the input size
        if data_type == "Flat":
            sigs = np.split(sigs, len_split, axis=0)
            count = 0
            for s in sigs:
                sigs[count] = s.reshape(1, s.shape[0], s.shape[1], s.shape[2])
                count += 1
        elif data_type == "1channelTensor":
            sigs = np.split(sigs, len_split, axis=0)
        elif data_type == "3channelTensor":
            sigs = np.split(sigs, len_split, axis=0)
        else:
            print("Incorrect data type.")

        # Same with labels (for each chunk, I pick the mode)
        labels = stats.mode(np.split(labels, len_split), axis=1)[0].ravel().tolist()

        # Associate signals with labels
        data = list(
            zip(
                sigs,
                tf.keras.utils.to_categorical(
                    labels, num_classes=number_classes
                ).reshape(len(sigs), 1, number_classes),
            )
        )

        # Store data in its corresponding list
        samples_list.extend(data)

    if shuffle_batches_train == True:
        samples_list = shuffle(samples_list)

    n_batches = len(samples_list)
    gen = aux_generator_RNN(samples_list, data_type)

    return n_batches, gen


def aux_generator_RNN(list_data, t_type):
    while True:
        for batch in list_data:

            if t_type == "Flat":
                yield (batch)
            else:
                batch_matrix = batch[0]
                # batch_label = batch[1]
                batch_inter = []

                for i in range(0, batch_matrix.shape[0]):
                    batch_inter.append(
                        preprocess_input(batch_matrix[i, :, :, :], tensor_type=t_type)
                    )

                batch_inter = np.expand_dims(np.array(batch_inter), axis=0)

                yield ((batch_inter))


# %% Generator for MLP and CNN
def generator_batches(
    X,
    Y,
    t_indexes,
    batch_size=64,
    data_type="Tensor",
    SNR=None,
    imgAugm=False,
    tensor_type="3channel",
):
    # while True:
    if SNR != None:
        if type(SNR).__name__ == "int":
            X_noisy_full = add_noise(X, SNR)
            X_data_batches = X_noisy_full[t_indexes, :]
            Y_data_batches = Y[t_indexes]
        elif type(SNR).__name__ == "list":
            X_data_batches = []
            Y_data_batches = []
            for rate in SNR:
                X_noisy_full = add_noise(X, rate)
                X_data_batches.append(X_noisy_full[t_indexes, :])
                Y_data_batches.append(Y[t_indexes])
            X_data_batches = np.vstack(X_data_batches)
            Y_data_batches = np.concatenate(Y_data_batches)
    else:
        X_data_batches = X[t_indexes, :]
        Y_data_batches = Y[t_indexes]

    indexes_split = np.arange(0, X_data_batches.shape[0], batch_size)
    if indexes_split[-1] < (X_data_batches.shape[0] - 1):
        indexes_split = np.append(indexes_split, X_data_batches.shape[0] - 1)
    n_batches = len(indexes_split) - 1

    batch_generator = set_generator(
        X_data_batches,
        Y_data_batches,
        indexes_split,
        data_type,
        imgAugm,
        tensor_type,
        batch_size,
    )

    return n_batches, batch_generator


def set_generator(
    X_data_batches,
    Y_data_batches,
    indexes_split,
    data_type,
    imgAugm,
    tensor_type,
    batch_size,
):
    datagen = ImageDataGenerator(
        zoom_range=0.25,
        rotation_range=15,
        width_shift_range=0.15,
        height_shift_range=0.15,
    )
    while True:
        if data_type != "Flat" and imgAugm == True:
            x_batch = []
            y_batch = []

            chosen_instant = np.random.randint(0, len(Y_data_batches), size=batch_size)
            x_img = get_tensor_model(X_data_batches[chosen_instant, :].T, tensor_type)
            y_label = tf.keras.utils.to_categorical(
                Y_data_batches[chosen_instant], num_classes=8
            )
            for j in range(0, len(x_img)):
                if tensor_type == "1channel":
                    x_batch.append(preprocess_input(x_img[j, :, :], tensor_type))
                else:
                    x_batch.append(preprocess_input(x_img[j, :, :, :], tensor_type))
            DA_iter = datagen.flow(np.array(x_batch), y_label, batch_size)
            yield (next(DA_iter))
        else:
            for i in range(0, len(indexes_split) - 1):
                y_batch_prev = Y_data_batches[indexes_split[i] : indexes_split[i + 1]]
                y_batch = tf.keras.utils.to_categorical(y_batch_prev, num_classes=8)
                x_batch = X_data_batches[indexes_split[i] : indexes_split[i + 1], :]
                if data_type == "Flat":
                    yield (x_batch, y_batch)
                else:
                    x_batch_prev = get_tensor_model(x_batch.T, tensor_type)
                    x_batch = []
                    for j in range(0, len(x_batch_prev)):
                        if tensor_type == "1channel":
                            x_batch.append(
                                preprocess_input(x_batch_prev[j, :, :], tensor_type)
                            )
                        elif tensor_type == "1channel_repeated":
                            input_temp = preprocess_input(
                                x_batch_prev[j, :, :], tensor_type
                            )
                            rep_temp = np.repeat(input_temp[:, :, :], 3, 2)
                            x_batch.append(rep_temp)
                        else:
                            x_batch.append(
                                preprocess_input(x_batch_prev[j, :, :, :], tensor_type)
                            )
                    x_batch = np.array(x_batch)
                    yield (x_batch, y_batch)
