from numpy import reshape
from tools_.tools import *
import tensorflow as tf

tf.random.set_seed(1)


def preprocessing_autoencoder_input(x_train, x_test, x_val, n_batch):
    """
    Function to preprocess input to fit autoencoder shapes

    Autoencoder Input shape = [# batches, batch_size, 12, 32, 1]
    Autoencoder Output shape = [# batches, batch_size, 12, 32, 1]
    Autoencoder Latent space shape = [# batches, batch_size, 3, 4, 12]

    Parameters
    ----------
    x_train: numpy array containing training data (shape:
    x_test

    Returns
    x_train
    -------

    """
    try:
        # Reshape and batch_generation to fit Conv (Add 1 dimension)

        x_train_reshaped = reshape(
            x_train,
            (
                int(len(x_train) / n_batch),
                n_batch,
                x_train.shape[1],
                x_train.shape[2],
                1,
            ),
        )
        x_test_reshaped = reshape(
            x_test,
            (int(len(x_test) / n_batch), n_batch, x_test.shape[1], x_test.shape[2], 1),
        )
        x_val_reshaped = reshape(
            x_val,
            (int(len(x_val) / n_batch), n_batch, x_val.shape[1], x_val.shape[2], 1),
        )

    except:

        raise Exception(
            "Input shape for autoencoder 3D is [# batches, batch_size, 12, 32, 1]. Current input shape is: ",
            x_train.shape,
        )

    return x_train_reshaped, x_test_reshaped, x_val_reshaped


def preprocessing_regression_input(
    latent_vector_train,
    latent_vector_test,
    latent_vector_val,
    train_models,
    test_models,
    val_models,
    Y_model,
    egm_tensor,
    AF_models,
    n_batch,
    random_split=True,
    norm=False,
):
    """
    Regression Input shape: [# batches, batch_size, 3, 4, 12]
    Regression Output shape: [# batches, batch_size, #nodes]

    Parameters
    ----------
    latent_vector_train: [# batches, batch_size, 3, 4, 12]
    latent_vector_test: [# batches, batch_size, 3, 4, 12]
    latent_vector_val: [# batches, batch_size, 3, 4, 12]
    Y_model
    egm_tensor

    Returns
    y_train, y_test, y_val, x_train_ls, x_test_ls, x_val_ls: x and x
    n_nodes: number of nodes are predicted (original geometry: 2048 nodes in heart geom)

    """

    try:

        latent_space_n, egm_tensor_n = preprocess_latent_space(
            latent_vector_train,
            latent_vector_test,
            latent_vector_val,
            train_models,
            test_models,
            val_models,
            Y_model,
            egm_tensor,
            dimension=5,
            norm=True,
        )
    except:

        raise Exception(
            "Input shape for Regression network is [# batches, batch_size, 3, 4, 12]. Current input shape is: ",
            latent_vector_train.shape,
        )

    # Split egm_tensor
    if random_split:
        x_train = latent_space_n[np.in1d(AF_models, train_models)]
        x_test = latent_space_n[np.in1d(AF_models, test_models)]
        x_val = latent_space_n[np.in1d(AF_models, val_models)]
    else:
        x_train = latent_space_n[np.where((Y_model >= 1) & (Y_model <= 200))]
        x_test = latent_space_n[np.where((Y_model > 180) & (Y_model <= 244))]
        x_val = latent_space_n[np.where((Y_model > 244) & (Y_model <= 286))]

    # Split EGM (Label)
    if random_split:
        y_train = egm_tensor_n[np.in1d(AF_models, train_models)]
        y_test = egm_tensor_n[np.in1d(AF_models, test_models)]
        y_val = egm_tensor_n[np.in1d(AF_models, val_models)]

    else:

        y_train = egm_tensor_n[np.where((Y_model >= 1) & (Y_model <= 200))]
        y_test = egm_tensor_n[np.where((Y_model > 180) & (Y_model <= 244))]
        y_val = egm_tensor_n[np.where((Y_model > 244) & (Y_model <= 286))]

    # %% Subsample EGM nodes

    y_train_subsample = y_train[:, 0:2048:3]
    y_test_subsample = y_test[:, 0:2048:3]
    y_val_subsample = y_val[:, 0:2048:3]

    n_nodes = y_train_subsample.shape[1]

    # Batch generation
    x_train_ls = reshape(
        x_train,
        (
            int(len(x_train) / n_batch),
            n_batch,
            x_train.shape[1],
            x_train.shape[2],
            x_train.shape[3],
        ),
    )
    x_test_ls = reshape(
        x_test,
        (
            int(len(x_test) / n_batch),
            n_batch,
            x_test.shape[1],
            x_test.shape[2],
            x_test.shape[3],
        ),
    )
    x_val_ls = reshape(
        x_val,
        (
            int(len(x_val) / n_batch),
            n_batch,
            x_val.shape[1],
            x_val.shape[2],
            x_val.shape[3],
        ),
    )

    y_train = reshape(
        y_train_subsample,
        (int(len(y_train_subsample) / n_batch), n_batch, y_train_subsample.shape[1]),
    )
    y_test = reshape(
        y_test_subsample,
        (int(len(y_test_subsample) / n_batch), n_batch, y_test_subsample.shape[1]),
    )
    y_val = reshape(
        y_val_subsample,
        (int(len(y_val_subsample) / n_batch), n_batch, y_val_subsample.shape[1]),
    )

    return y_train, y_test, y_val, x_train_ls, x_test_ls, x_val_ls, n_nodes


def preprocessing_y(
    egm_tensor,
    Y_model,
    AF_models,
    train_models,
    test_models,
    val_models,
    n_batch,
    norm=True,
    random_split=True,
):
    # Normalize
    if norm:
        egm_tensor_n = []
        for model in np.unique(Y_model):

            # 2. Normalize egm (output)
            arr_to_norm_egm = egm_tensor[
                np.where((Y_model == model))
            ]  # select window of signal belonging to model i
            egm_tensor_norm = normalize_array(arr_to_norm_egm, 1, -1)
            egm_tensor_n.extend(egm_tensor_norm)  # Add to new norm array

        egm_tensor_n = np.array(egm_tensor_n)

    else:

        egm_tensor_n = egm_tensor

    # Split EGM (Label)
    if random_split:
        y_train = egm_tensor_n[np.in1d(AF_models, train_models)]
        y_test = egm_tensor_n[np.in1d(AF_models, test_models)]
        y_val = egm_tensor_n[np.in1d(AF_models, val_models)]

    else:

        y_train = egm_tensor_n[np.where((Y_model >= 1) & (Y_model <= 200))]
        y_test = egm_tensor_n[np.where((Y_model > 180) & (Y_model <= 244))]
        y_val = egm_tensor_n[np.where((Y_model > 244) & (Y_model <= 286))]

    # %% Subsample EGM nodes

    y_train_subsample = y_train[:, 0:2048]  #:, 0:2048:2] --> 1024
    y_test_subsample = y_test[:, 0:2048]
    y_val_subsample = y_val[:, 0:2048]

    n_nodes = y_train_subsample.shape[1]

    y_train = reshape(
        y_train_subsample,
        (int(len(y_train_subsample) / n_batch), n_batch, y_train_subsample.shape[1]),
    )
    y_test = reshape(
        y_test_subsample,
        (int(len(y_test_subsample) / n_batch), n_batch, y_test_subsample.shape[1]),
    )
    y_val = reshape(
        y_val_subsample,
        (int(len(y_val_subsample) / n_batch), n_batch, y_val_subsample.shape[1]),
    )

    return y_train, y_test, y_val
