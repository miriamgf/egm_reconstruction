from numpy import *
from fastdtw import fastdtw


def evaluate_function(x_train, y_train, x_test, y_test, model, batch_size=1):
    '''
    This function evaluates the model on the specified metrics
    Parameters
    ----------
    batch_size
    x_test
    x_train
    model
    y_test
    y_train

    Returns
    -------

    '''

    results_test = model.evaluate(x_test, y_test, batch_size=batch_size)
    results_train = model.evaluate(x_train, y_train, batch_size=batch_size)

    # Save
    mse_train, mae_train = results_train[0], results_train[2]
    mse_test, mae_test = results_test[0], results_test[2]

    dtw_test, dtw_train = DynamicTimeWarping(x_train, y_train, x_test, y_test, model)

    results = {'mse test': mse_test, 'mse train': mse_train,
               'mae test': mae_test, 'mae train': mae_train,
               'dtw_test': dtw_test, 'dtw_train': dtw_train}

    return results


def DynamicTimeWarping(pred_train, y_train, pred_test, y_test, model):
    '''
    Compues an implementation of fast of Dynamic Time Warping, a method for comparing two sequences that do not perfectly sync up
    Parameters.
    * It requires flattening the inputs!
    ----------
    x_train
    y_train
    x_test
    y_test

    Returns
    -------

    '''

    n_dim_output = 2
    pred_train_flat = reshape_tensor(pred_train, n_dim_input=pred_train.ndim, n_dim_output=n_dim_output)
    pred_test_flat = reshape_tensor(pred_test, n_dim_input=pred_test.ndim, n_dim_output=n_dim_output)
    y_train_flat = reshape_tensor(y_train, n_dim_input=y_train.ndim, n_dim_output=n_dim_output)
    y_test_flat = reshape_tensor(y_test, n_dim_input=y_test.ndim, n_dim_output=n_dim_output)

    dtw_test, path_test = fastdtw(pred_test_flat, y_test_flat)
    dtw_train, path_train = fastdtw(pred_train_flat, y_train_flat)

    return dtw_test, dtw_train


def reshape_tensor(tensor, n_dim_input, n_dim_output):
    '''
    Reshapes the tensors used during pipeline, considering that the first two dimensions are (#n batches, batch size).
    In the case of n_dim_input = 5, the last dimension is the number of channels.

    Parameters
    ----------
    tensor: tensor to reshape
    n_dim_input: input shape
    n_dim_output: desired output shape

    Returns
    -------

    '''

    try:

        # case of autoencoder output: first two dimensions
        if n_dim_input == 5 and n_dim_output == 2:
            reshaped_tensor = reshape(tensor, (
                tensor.shape[0] * tensor.shape[1],
                tensor.shape[2] * tensor.shape[3]))
            return reshaped_tensor

        # case of regression output
        elif n_dim_input == 3 and n_dim_output == 2:
            reshaped_tensor = reshape(tensor, (
                tensor.shape[0] * tensor.shape[1],
                tensor.shape[2]))
            return reshaped_tensor

    except:
        raise (
            ValueError("Input shape - Output shape combination is not implemented: check reshape_tensor documentation"))
