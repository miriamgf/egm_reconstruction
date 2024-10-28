from numpy import *
from fastdtw import fastdtw
from tools_.tools import reshape_tensor


def evaluate_function(
    x_train, y_train, x_test, y_test, pred_train, pred_test, model, batch_size=1
):
    """
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

    """

    results_test = model.evaluate(x_test, y_test, batch_size=batch_size)
    results_train = model.evaluate(x_train, y_train, batch_size=batch_size)

    # Save
    mse_train, mae_train = results_train[0], results_train[2]
    mse_test, mae_test = results_test[0], results_test[2]

    dtw_test, dtw_train = DynamicTimeWarping(
        y_train, pred_train, y_test, pred_test, model
    )

    results = {
        "mse test": mse_test,
        "mse train": mse_train,
        "mae test": mae_test,
        "mae train": mae_train,
        "dtw_test": dtw_test,
        "dtw_train": dtw_train,
    }

    return results


def evaluate_function_multioutput(
    x_train, y_train, x_test, y_test, pred_train, pred_test, model, batch_size=1
):
    """
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

    """

    (
        global_loss_test,
        mse_test_ae,
        mse_test_regressor,
        mae_test_ae,
        mae_test_regressor,
    ) = model.evaluate(x_test, [x_test, y_test], batch_size=batch_size)
    (
        global_loss_train,
        mse_train_ae,
        mse_train_regressor,
        mae_train_ae,
        mae_train_regressor,
    ) = model.evaluate(x_train, [x_train, y_train], batch_size=batch_size)

    dtw_test, dtw_train = [
        0,
        0,
    ]  # DynamicTimeWarping(y_train, pred_train, y_test, pred_test, model)

    results_autoencoder = {
        "mse test": mse_test_ae,
        "mse train": mse_train_ae,
        "mae test": mae_test_ae,
        "mae train": mae_train_ae,
        "dtw_test": dtw_test,
        "dtw_train": dtw_train,
        "global_loss_test": global_loss_test,
        "global_loss_train": global_loss_train,
    }

    results_regressor = {
        "mse test": mse_test_regressor,
        "mse train": mse_train_regressor,
        "mae test": mae_test_regressor,
        "mae train": mae_train_regressor,
        "dtw_test": dtw_test,
        "dtw_train": dtw_train,
        "global_loss_test": global_loss_test,
        "global_loss_train": global_loss_train,
    }

    return results_autoencoder, results_regressor


def DynamicTimeWarping(pred_train, y_train, pred_test, y_test, model):
    """
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

    """

    n_dim_output = 2
    pred_train_flat = reshape_tensor(
        pred_train, n_dim_input=pred_train.ndim, n_dim_output=n_dim_output
    )
    pred_test_flat = reshape_tensor(
        pred_test, n_dim_input=pred_test.ndim, n_dim_output=n_dim_output
    )
    y_train_flat = reshape_tensor(
        y_train, n_dim_input=y_train.ndim, n_dim_output=n_dim_output
    )
    y_test_flat = reshape_tensor(
        y_test, n_dim_input=y_test.ndim, n_dim_output=n_dim_output
    )

    dtw_test, path_test = fastdtw(pred_test_flat, y_test_flat)
    dtw_train, path_train = fastdtw(pred_train_flat, y_train_flat)

    return dtw_test, dtw_train
