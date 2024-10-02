# This script was developed Miriam Gutiérrez Fernández

import optuna
from tensorflow.keras.optimizers import Adam
from models.multioutput import MultiOutput
from tools_.load_dataset import LoadDataset
from tools_.preprocess_data import Preprocess_Dataset
from tools_.train_model import TrainModel
from config import ParseHiperparams 
from optuna.samplers import TPESampler


class OptunaOpt:
    """
    The OptunaOpt class performs hyperparameter optimization using Optuna, given a declared search space and a specified sampling algorithm.

    Attributes:
        search_space (dict): Dictionary defining the search space for hyperparameters.
        parameters (dict): Dictionary of general parameters.
        data: Data used for training and evaluation.
        signals: Signals used for training and evaluation.
        train_model_instance: Instance of TrainModel used for model training and evaluation.

    Methods:
        __init__(self, search_space, data, signals, parameters, train_model_instance=None)
            Initializes an instance of the OptunaOpt class.

        hyperparameter_optimization_optuna(self) -> dict
            Executes Optuna optimization to find the best hyperparameters based on model performance.

        plot_optuna_results(self, study)
            Creates and saves visualizations for the Optuna optimization results.
    """

    def __init__(self, search_space, X_1channel, egm_tensor, AF_models, Y_model,dic_vars,Y, all_model_names,transfer_matrices,parameters, models_dir, experiment_dir):
        """
        Initializes an instance of the OptunaOpt class.

        Args:
            search_space (dict): Dictionary defining the search space for hyperparameters.
            data: Data used for training and evaluation.
            signals: Signals used for training and evaluation.
            parameters (dict): Dictionary of general parameters.
            train_model_instance: Instance of TrainModel used for model training and evaluation.
        """
        self.search_space = search_space
        self.parameters = parameters
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
        self.hyperparams_path = ParseHiperparams.get_path()
    
    def parse_search_space(self, trial):

        search_space = ParseHiperparams.parse_optuna_hyperparams()
        optuna_params = {}
        for param_name, param_config in search_space.items():
            if param_name == 'batch_size':
                continue
            elif param_name == 'tpe':
                param_config = TPESampler()
            optuna_params[param_name] = trial.suggest_categorical(param_name, param_config)

            # Overwrite in dictionary
            self.parameters[param_name] = optuna_params[param_name]
           




    def hyperparameter_optimization_optuna(self) -> dict:
        """
        Executes Optuna optimization to find the best hyperparameters based on model performance.

        Returns:
            dict: Best hyperparameters obtained from the optimization.
        """

    

        def objective(trial):
            """
            Objective function for Optuna optimization.

            Args:
                trial: Optuna trial object for suggesting hyperparameters.

            Returns:
                float: Accuracy score of the model with the suggested hyperparameters.
            """
            
            #Preprocess data
            x_train, x_test, x_val, y_train, y_test, y_val, dic_vars, BSPM_train, BSPM_test, BSPM_val, AF_models_train, AF_models_test, AF_models_val, train_models, test_models, val_models = Preprocess_Dataset(
                self.X_1channel,
                self.egm_tensor,
                self.AF_models,
                self.Y_model, 
                self.dic_vars, 
                self.Y, self.all_model_names, self.transfer_matrices)()

            model, history = TrainModel(x_train, x_test, x_val, y_train, y_test, y_val, self.models_dir, self.experiment_dir)()

            val_loss = history.val_loss

            return val_loss
        
        # Optimización
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=50)

        # Obtener mejores hiperparámetros
        best_params = study.best_params
        print("Best hyperparameters:", best_params)
        return best_params
    

    def __call__(self):
        return self.hyperparameter_optimization_optuna()


