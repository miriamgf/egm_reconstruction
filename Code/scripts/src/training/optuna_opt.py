#import optuna
from tensorflow.keras.optimizers import Adam
from models.multioutput import MultiOutput
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

    def __init__(self, search_space, x_train, y_train, x_val, y_val, parameters, callbacks):
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
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.callbacks = callbacks

    

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
            '''
            # Automatically unpack dictionary into optuna params
            optuna_params = {}
            for param_name, param_config in self.search_space.items():
                if param_name == 'batch_size':
                    continue
                elif param_name == 'tpe':
                    param_config = TPESampler()
                optuna_params[param_name] = trial.suggest_categorical(param_name, param_config)

                # Overwrite in dictionary
                self.parameters[param_name] = optuna_params[param_name]
            '''
            # Espacio de búsqueda de hiperparámetros
            learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
            #batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
            n_epochs = trial.suggest_int('n_epochs', 10, 100)

            #TODO: Modify the load function for batch and downsampling outside

            # Crear el modelo
            optimizer = Adam(learning_rate=learning_rate)
            model = MultiOutput().assemble_full_model(input_shape=self.x_train.shape[1:], n_nodes=self.y_train.shape[-1])
            model.compile(optimizer=optimizer,
                        loss=['mean_squared_error', 'mean_squared_error'],
                        loss_weights=[1.0, 5.0],
                        metrics=['mean_absolute_error'])
            

            # Entrenamiento
            history = model.fit(x=self.x_train, y=[self.x_train, self.y_train], batch_size=batch_size, epochs=n_epochs,
                                validation_data=(self.x_val, [self.x_val, self.y_val]),
                                callbacks=self.callbacks, verbose=0)

            # Devuelve la métrica que se quiere optimizar
            val_loss = history.history['val_loss'][-1]
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


