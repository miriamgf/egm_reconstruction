import optuna
from tensorflow.keras.optimizers import Adam
from models.multioutput import MultiOutput
from tools_.load_dataset import LoadDataset
from tools_.preprocess_data import Preprocess_Dataset
from tools_.train_model import TrainModel
from config import ParseHiperparams
from optuna.samplers import TPESampler, RandomSampler, NSGAIISampler

import json
from pathlib import Path

class OptunaOpt:
    """
    A class to perform hyperparameter optimization using Optuna. It orchestrates the 
    entire process of searching for the best hyperparameters, training the model, and 
    storing the optimal hyperparameters for future reference.

    Attributes:
    -----------
    params : dict
        Dictionary containing the initial parameters to configure the model and the search space for optimization.
    X_1channel : numpy.array
        Data with one-channel input for model training.
    egm_tensor : numpy.array
        Tensor containing electrogram data used for model training.
    AF_models : list
        List of atrial fibrillation models to be used for training.
    Y_model : numpy.array
        Ground truth output values for the models.
    dic_vars : dict
        Dictionary containing various input variables to be used during training and preprocessing.
    Y : numpy.array
        Labels or output data for the model's predictions.
    all_model_names : list
        List of all model names used in the optimization process.
    transfer_matrices : list
        List of transfer matrices used for the training process.
    models_dir : str
        Path to the directory where models will be saved.
    experiment_dir : str
        Path to the directory where experiment outputs (e.g., results, plots) will be stored.
    hyperparams_path : str
        Path to the configuration file containing the hyperparameters to be optimized.
    
    Methods:
    --------
    __init__(self, params, X_1channel, egm_tensor, AF_models, Y_model, dic_vars, Y, all_model_names, transfer_matrices, models_dir, experiment_dir)
        Initializes the OptunaOpt class with the provided data and paths.
        
    parse_search_space(self, trial)
        Parses the search space for hyperparameters and returns a dictionary of hyperparameter suggestions using Optuna's trial object.

    hyperparameter_optimization_optuna(self) -> dict
        Executes the optimization process using Optuna to find the best hyperparameters.

    parse_best_params(self, best_params)
        Saves the best hyperparameters found during optimization to a JSON file.

    __call__(self)
        Initiates the hyperparameter optimization process and returns the best hyperparameters.
    """

    def __init__(
        self,
        params,
        X_1channel,
        egm_tensor,
        AF_models,
        Y_model,
        dic_vars,
        Y,
        all_model_names,
        transfer_matrices,
        models_dir,
        experiment_dir,
    ):
        """
        Constructor to initialize the OptunaOpt class with model parameters, datasets, and paths.

        Args:
        -----
        params : dict
            Initial configuration parameters for the model and optimization search space.
        X_1channel : numpy.array
            One-channel data for training the model.
        egm_tensor : numpy.array
            Electrogram data tensor used for the model.
        AF_models : list
            List of atrial fibrillation models for the dataset.
        Y_model : numpy.array
            Ground truth output labels for the models.
        dic_vars : dict
            Dictionary containing various variables for preprocessing and training.
        Y : numpy.array
            Output data used for model prediction.
        all_model_names : list
            Names of all the models being considered in the optimization.
        transfer_matrices : list
            Transfer matrices for model training.
        models_dir : str
            Directory path for saving trained models.
        experiment_dir : str
            Directory path for saving experiment outputs (e.g., logs, plots, and hyperparameters).
        """
        self.params = params
        self.X_1channel = X_1channel
        self.egm_tensor = egm_tensor
        self.AF_models = AF_models
        self.Y_model = Y_model
        self.dic_vars = dic_vars
        self.Y = Y
        self.all_model_names = all_model_names
        self.transfer_matrices = transfer_matrices
        self.models_dir = models_dir
        self.experiment_dir = experiment_dir
        self.hyperparams_path = ParseHiperparams().get_path()

    def parse_search_space(self, trial):
        """
        Parses the hyperparameter search space using Optuna's trial object.

        Args:
        -----
        trial : optuna.Trial
            An Optuna trial object that allows for the dynamic suggestion of hyperparameters.

        Returns:
        --------
        dict
            A dictionary containing the suggested hyperparameters for the current trial.
        """
        search_space = ParseHiperparams().parse_optuna_hyperparams()
        optuna_params = {}

        # Take sampler
        for param_name, param_config in search_space.items():
            if param_name == "tpe":
                param_config = TPESampler()
            elif param_name == "random":
                param_config = RandomSampler()
            elif param_name == "genetic":
                param_config = NSGAIISampler()
            optuna_params[param_name]=param_config
    
        # Sample 
        for param_name, param_config in search_space.items():
            if param_config[0] == "range":
                if param_name == "lr" or param_name == "learning_rate":
                    optuna_params[param_name] = trial.suggest_float(param_name, param_config[1], param_config[2], log=True)
                    # Overwrite in dictionary
                    self.params[param_name] = optuna_params[param_name]

                else:  
                    if type(param_config[1]) == int:
                        #with step explicited
                        if len(param_config) >3:
                            optuna_params[param_name] = trial.suggest_int(param_name, param_config[1], param_config[2], step = param_config[3] )
                            # Overwrite in dictionary
                            self.params[param_name] = optuna_params[param_name]

                        else: 
                            optuna_params[param_name] = trial.suggest_int(param_name, param_config[1], param_config[2])
                            # Overwrite in dictionary
                            self.params[param_name] = optuna_params[param_name]


                    elif type(param_config[1]) == float:
                        if len(param_config) >3:
                            optuna_params[param_name] = trial.suggest_float(param_name, param_config[1], param_config[2], step = param_config[3])
                            # Overwrite in dictionary
                            self.params[param_name] = optuna_params[param_name]

                        else:
                            optuna_params[param_name] = trial.suggest_float(param_name, param_config[1], param_config[2])
                            # Overwrite in dictionary
                            self.params[param_name] = optuna_params[param_name]


            elif param_config[0] == "grid":
                optuna_params[param_name] = trial.suggest_categorical(
                    param_name, param_config[1]
                
                )
                # Overwrite in dictionary
                self.params[param_name] = optuna_params[param_name]


        
        return self.params

    def hyperparameter_optimization_optuna(self) -> dict:
        """
        Executes the hyperparameter optimization process using Optuna.

        This method defines an objective function which is used by Optuna to search 
        for the best hyperparameters based on the model's performance on validation data.

        Returns:
        --------
        dict
            A dictionary containing the best hyperparameters found during the optimization process.
        """

        def objective(trial):
            """
            Defines the objective function for the Optuna optimization trial.

            Args:
            -----
            trial : optuna.Trial
                An Optuna trial object to suggest hyperparameters.

            Returns:
            --------
            float
                The validation loss after training the model with the trial's suggested hyperparameters.
            """
            self.params = self.parse_search_space(trial)

            # Preprocess data
            (
                x_train,
                x_test,
                x_val,
                y_train,
                y_test,
                y_val,
                dic_vars,
                BSPM_train,
                BSPM_test,
                BSPM_val,
                AF_models_train,
                AF_models_test,
                AF_models_val,
                train_models,
                test_models,
                val_models,
            ) = Preprocess_Dataset(
                self.params,
                self.X_1channel,
                self.egm_tensor,
                self.AF_models,
                self.Y_model,
                self.dic_vars,
                self.Y,
                self.all_model_names,
                self.transfer_matrices,
            )()

            # Train the model
            model, history = TrainModel(
                self.params,
                x_train,
                x_test,
                x_val,
                y_train,
                y_test,
                y_val,
                self.models_dir,
                self.experiment_dir,
            )()

            val_loss = history.history["val_Regressor_output_loss"]

            return val_loss[-1]

        # Perform optimization
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=3)

        # Get best hyperparameters
        best_params = study.best_params
        print("Best hyperparameters:", best_params)
        return best_params

    def parse_best_params(self, best_params):
        """

        Rewrite params variable and saves the best hyperparameters found during
        the optimization process to a JSON file.

        Args:
        -----
        best_params : dict
            Dictionary containing the best hyperparameters from the Optuna optimization.

        Returns:
        --------
        dict
            The updated parameters dictionary including the best hyperparameters.
        """
        for param_name, param_config in best_params.items():
            self.params[param_name] = best_params[param_name]

        # Save to JSON
        with open(Path(self.experiment_dir, 'best_params.json'), "w") as outfile:
            json.dump(self.params, outfile)

        return self.params

    def __call__(self):
        """
        Invokes the class, starting the hyperparameter optimization process and returning the best hyperparameters.

        Returns:
        --------
        dict
            The best hyperparameters after the optimization process.
        """
        return self.hyperparameter_optimization_optuna()
