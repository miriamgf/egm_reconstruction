# This script was developed Miriam Gutiérrez Fernández
# """
import os
import sys

from tools_.noise_simulation import *
from tools_.tools_1 import *

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
from add_white_noise import *
from generators import *
from numpy import reshape
from plots import *
from scipy.io import savemat


from tools_.noise_simulation import *
from tools_.tools_1 import *
from tools_.k_fold import KFold_Stratified
from tools_.stratified_split import StratifiedSplit

# from noise_simulation import *

# %% Path Models
current = os.path.dirname(os.path.realpath(__file__))
torsos_dir = "../../../Labeled_torsos/"
# directory = "/home/profes/miriamgf/tesis/Autoencoders/Data/"
torsos_dir = "/home/profes/miriamgf/tesis/Autoencoders/Labeled_torsos/"

fs = 500


class PreprocessSyntDataset:
    """
    The Preprocess_Dataset class preprocess dataset loaded previously. Among the tasks that it performs:
        - Temporal Downsampling
        - Split into train and test
        - Batch generation
        - Normalization
        - Reshape to fit into input layers of the network

    """

    def __init__(
        self,
        params,
        X_1channel,
        egm_tensor,
        AF_models,
        class_complexity_list,
        Y_model,
        dic_vars,
        Y,
        all_model_names,
        transfer_matrices,experiment_dir,split_mode="deterministic", norm_egm=True, inference = False, shuffle_patient=False
    ):
        self.params = params
        self.X_1channel = X_1channel
        self.egm_tensor = egm_tensor
        self.AF_models = AF_models
        self.class_complexity_list = class_complexity_list
        self.Y_model = Y_model
        self.dic_vars = dic_vars
        self.Y = Y
        self.all_model_names = all_model_names
        self.transfer_matrices = transfer_matrices
        self.experiment_dir=experiment_dir 
        self.norm_egm=norm_egm
        self.inference =inference
        self.shuffle_patient=shuffle_patient
        self.split_mode=split_mode
        try:
            self.SEED = self.params["seed"]
        except:
            self.SEED = 1234
        random.seed(self.SEED)        
        np.random.seed(self.SEED)     

    def preprocess_main(self):
        """
        This function defines the main steps to perform preprocessing over target signals
        1) Apply downsampling and truncate length according to batch size
        2) Normalization between -1 and 1
        3) Clean Nans generated during Noise Generation (zero-Patches)
        4) Train/Test/Val split
        5) Batch generation


        """

        print("Preprocessing...")
        print('preprocess_compression')
        # Downsampling and truncate
        self.X_1channel, self.egm_tensor, self.AF_models, self.class_complexity_list, self.Y_model = (
            self.preprocess_compression(
                fs_sub=self.params["fs_sub"],
                batch_size=self.params["batch_size"],
                downsampling=True,
            )
        )

        print('normalize')
        # Normalize BSPS and EGM
        self.X_1channel = normalize_by_models(self.X_1channel, self.Y_model)

        if self.norm_egm:
            self.egm_tensor = normalize_by_models(self.egm_tensor, self.Y_model)
        
        #Remove Nans
        self.X_1channel = np.nan_to_num(
            self.X_1channel, nan=0.0
        )  # Nans generated during noise addition

        #Save

        if self.inference:
            print("Inference mode, no train/test/val split")

            return self.X_1channel, self.egm_tensor, self.AF_models, self.Y_model
        
        plt.figure()
        plt.plot(self.X_1channel[0:200, 0, 0], label="bsps")
        plt.plot(self.egm_tensor[0:200, 0], label="egm")
        plt.legend()
        os.makedirs("output/figures/input_output/", exist_ok=True)
        plt.savefig("output/figures/input_output/norm.png")


        # Train/Test/Val Split
        print("Splitting...")
        (
            x_train,
            train_models,
            AF_models_train,
            class_complexity_list_train,
            BSPM_train,
        ) = self.train_test_val_split_Autoencoder(
            BSPM_Models=self.X_1channel,
            random_split=True,
            train_percentage=0.8,
            test_percentage=0.1,
        )

        print("TRAIN SHAPE:", x_train.shape, "models:", train_models)


        x_train, _, _ = self.preprocessing_autoencoder_input(
            x_train, x_train, x_train, self.params["batch_size"]
        )
        
        y_train, class_complexity_list_train= self.preprocessing_y(
            train_models,
            class_complexity_list_train,
            self.params["batch_size"],

        )

        #
       
        '''
        plt.figure()
        plt.plot(x_train[0, :, 0, 0, 0], label="bsps")
        plt.plot(y_train[0, :, 0], label="egm")
        plt.legend()
        '''
        os.makedirs("output/figures/input_output/", exist_ok=True)
        plt.savefig("output/figures/input_output/preprocessing.png")

        return (
            x_train,
            y_train,
            self.dic_vars,
            BSPM_train,
            AF_models_train,
            class_complexity_list_train,
            train_models
        )

    def preprocess_compression(
        self,
        fs_sub,
        batch_size,
        downsampling=True,
    ):
        """
        This function process loaded data (BSPS, EGMs and metadata) to perform downsampling and
        truncate the length of the arrays according to the specified batch size

        This operation must be accomplished by AF model to ensure that the truncation alineates when splitting
        into train - test - val

        """
        self.AF_models = np.array(self.AF_models)

        new_X_1channel = []
        new_egm_tensor = []
        new_AF_models = []
        new_Y_model = []
        new_class_complexity_list = []

        print('downsampling and truncating length by batch size...')
        unique_models = np.unique(self.AF_models)

        for AF_model_i in unique_models:
            mask = self.AF_models == AF_model_i

            X_1channel_i = self.X_1channel[mask]
            egm_tensor_i = self.egm_tensor[mask]
            AF_models_i = self.AF_models[mask]
            Y_model_i = self.Y_model[mask]

            if self.class_complexity_list is not None:
                complexity_i = np.array(self.class_complexity_list)[mask]

            if downsampling:
                factor = int(500 / fs_sub)
                X_1channel_i = X_1channel_i[::factor]
                egm_tensor_i = egm_tensor_i[::factor]
                AF_models_i = AF_models_i[::factor]  # labels don't need resampling
                Y_model_i = Y_model_i[::factor]
                if self.class_complexity_list is not None:
                    complexity_i = complexity_i[::factor]
                #X_1channel_i = signal.resample_poly(X_1channel_i, fs_sub, 500, axis=0)
                #egm_tensor_i = signal.resample_poly(egm_tensor_i, fs_sub, 500, axis=0)
                #AF_models_i = AF_models_i[::int(500/fs_sub)]  # labels don't need resampling
                #Y_model_i = Y_model_i[::int(500/fs_sub)]
                #if self.class_complexity_list is not None:
                    #complexity_i = complexity_i[::int(500/fs_sub)]

            X_1channel_i = self.truncate_length_by_batch_size(batch_size, X_1channel_i)
            egm_tensor_i = self.truncate_length_by_batch_size(batch_size, egm_tensor_i)
            AF_models_i = self.truncate_length_by_batch_size(batch_size, AF_models_i)
            Y_model_i = self.truncate_length_by_batch_size(batch_size, Y_model_i)
            if self.class_complexity_list is not None:
                complexity_i = self.truncate_length_by_batch_size(batch_size, complexity_i)

            new_X_1channel.append(X_1channel_i)
            new_egm_tensor.append(egm_tensor_i)
            new_AF_models.append(AF_models_i)
            new_Y_model.append(Y_model_i)
            if self.class_complexity_list is not None:
                new_class_complexity_list.append(complexity_i)
            del X_1channel_i, egm_tensor_i, AF_models_i, Y_model_i


        # Concatenar fuera del loop
        self.X_1channel = np.concatenate(new_X_1channel, axis=0)
        self.egm_tensor = np.concatenate(new_egm_tensor, axis=0)
        self.AF_models = np.concatenate(new_AF_models, axis=0)
        self.Y_model = np.concatenate(new_Y_model, axis=0)
        if self.class_complexity_list is not None:
            self.class_complexity_list = np.concatenate(new_class_complexity_list, axis=0)

        return (
            np.array(self.X_1channel),
            np.array(self.egm_tensor),
            list(self.AF_models),
            list(self.class_complexity_list),
            np.array(self.Y_model),
        )

    def truncate_length_by_batch_size(self, batch_size, signal_data):

        if signal_data.shape[0] % batch_size != 0:
            trunc_val = np.floor_divide(signal_data.shape[0], batch_size)
            signal_data = signal_data[0 : batch_size * trunc_val, ...]
        return signal_data

    def preprocessing_autoencoder_input(self, x_train, x_test, x_val, n_batch):
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
                (
                    int(len(x_test) / n_batch),
                    n_batch,
                    x_test.shape[1],
                    x_test.shape[2],
                    1,
                ),
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


    def preprocessing_y(
        self,
        train_models,
        class_complexity_list_train,
        n_batch
    ):
        

        egm_tensor_n = self.egm_tensor

        # Split EGM (Label)
        y_train = egm_tensor_n[np.in1d(self.AF_models, train_models)]

        
        # %% Subsample EGM nodes

        if self.params["n_nodes_regression"] == 2048:
            N = 1
        elif self.params["n_nodes_regression"] == 1024:
            N = 2
        elif self.params["n_nodes_regression"] == 682:
            N = 3
        elif self.params["n_nodes_regression"] == 512:
            N = 4
        else: #default
            N = 1

        y_train_subsample = y_train[:, 0:2048:N]  #:, 0:2048:2] --> 1024


        y_train = reshape(
            y_train_subsample,
            (
                int(len(y_train_subsample) / n_batch),
                n_batch,
                y_train_subsample.shape[1],
            ),
        )

        class_complexity_list_train = reshape(
            class_complexity_list_train,
            (
                int(len(class_complexity_list_train) / n_batch),
                n_batch
            ))

        return y_train, class_complexity_list_train

    def train_test_val_split_Autoencoder(
        self,
        BSPM_Models,
        random_split,
        train_percentage,
        test_percentage,
    ):
        """
        This function splits the input tensor into train, tets and validation
        Parameters:
            X_1channel-> input BSPs tensor
            AF_models -> list corresponding to the original AF model that corresponds to each BSP
            BSPM_Models -> list of BSP model values for each sample
            all_model_names -> Name of all AF models loaded
            random_split -> AF models are randomly shaffled and assigned to each subset (train, test, val)
            train_percentage -> train percentage of the input dataset dedicated to training the models
            test_percentage -> test percentage of the input dataset dedicated to testing the models.
            *Validation is computed as 100-train_percentage-test_percentage


        Return:
            x_train -> training tensor
            x_test -> testing tensor
            x_val -> validation tensor
            train_models, test_models, val_models  -> BSP Models in train, test and val (Only id of AF Model)
            AF_models_train, AF_models_test, AF_models_val  -> Tensor AF Models in train, test and val
            BSPM_train, BSPM_test, BSPM_val  -> Tensor BSP Models in train, test and val

        """
        caution_split = False  # Split in train and test taking into account the high corr between selected models in 'set_models'

        if caution_split:
            # Select indices of highly correlated signals

            set_models = {
                "190619",
                "190717",
                "191001",
                "200316_001",
                "200428",
                "200316",
                "200212",
            }  #
            indx = []
            for i in range(0, len(self.all_model_names)):
                for s in set_models:
                    if s in self.all_model_names[i]:
                        indx.append(i)

        AF_models_unique = np.unique(self.AF_models)

        # Random
        if random_split:

            if self.split_mode=="deterministic":
                if not self.params["cross_validation"]:
                    # Deterministic assignation
                    train_models_deterministic = self.all_model_names
                    val_models_deterministic = []
                    test_models_deterministic = []
                    
                elif self.params["cross_validation"]:
                    KFold_obj = KFold_Stratified(k=4)
                    fold_list=KFold_obj.select_fold(fold=self.params["fold"])
                    train_models_deterministic = fold_list["train"]
                    test_models_deterministic = fold_list["test"]
                    val_models_deterministic = fold_list["val"]

                train_models, test_models, val_models = [], [], []
                for elemento in train_models_deterministic:
                    if elemento in self.all_model_names:
                        train_models.append(self.all_model_names.index(elemento))

                for elemento in test_models_deterministic:
                    if elemento in self.all_model_names:
                        test_models.append(self.all_model_names.index(elemento))

                for elemento in val_models_deterministic:
                    if elemento in self.all_model_names:
                        val_models.append(self.all_model_names.index(elemento))
                
                #Save to dic
                with open(self.experiment_dir+"train_models.txt", "w") as file:
                    for model in train_models_deterministic:
                        file.write(model + "\n")

            
            elif self.split_mode=="stratified":
                print("Stratified split...")
                StratifiedSplit_obj = StratifiedSplit(classes_to_oversample=self.params["classes_to_oversample"],
                                                      discard_classes=self.params["discard_classes"],
                                                       oversampling=self.params["oversampling"])
                train_models_strat, test_models_strat, val_models_strat=StratifiedSplit_obj()

                train_models, test_models, val_models = [], [], []
                for elemento in train_models_strat:
                    if elemento in self.all_model_names:
                        train_models.append(self.all_model_names.index(elemento))

                for elemento in test_models_strat:
                    if elemento in self.all_model_names:
                        test_models.append(self.all_model_names.index(elemento))

                for elemento in val_models_strat:
                    if elemento in self.all_model_names:
                        val_models.append(self.all_model_names.index(elemento))
                
                #Save to dic
                with open(self.experiment_dir+"train_models.txt", "w") as file:
                    for model in train_models_strat:
                        file.write(model + "\n")
                with open(self.experiment_dir+"test_models.txt", "w") as file:
                    for model in test_models_strat:
                        file.write(model + "\n")
                with open(self.experiment_dir+"val_models.txt", "w") as file:
                    for model in val_models_strat:
                        file.write(model + "\n")


            if self.shuffle_patient:

                print('Random shuffling of patients applied')

                random.shuffle(train_models)

            x_train = self.X_1channel[np.in1d(self.AF_models, train_models)]

            BSPM_train = BSPM_Models[np.in1d(self.AF_models, train_models)]


            AF_models_arr = np.array(self.AF_models)
            AF_models_train = AF_models_arr[np.in1d(self.AF_models, train_models)]

            class_complexity_list_arr = np.array(self.class_complexity_list)
            class_complexity_list_train = class_complexity_list_arr[np.in1d(self.AF_models, train_models)]

        else:
            pass



        # Save the model names in train, test and val

        [self.all_model_names[index] for index in AF_models_train]

        return (
            x_train,
            train_models,
            AF_models_train,
            class_complexity_list_train,
            BSPM_train,
        )

    def preprocess_latent_space(
        self,
        latent_vector_train,
        latent_vector_test,
        latent_vector_val,
        train_models,
        test_models,
        val_models,
        Y_model,
        egm_tensor,
        dimension,
        norm=False,
    ):
        """
        This function preprocess the latent space, following the scheme:
        1) Center data at 0
        2) Reshape for normalization
        3) Normalization between -1 and 1
        """
        # Center latent space
        center_function = lambda x: x - x.mean(axis=0)

        latent_vector_train = center_function(latent_vector_train)
        latent_vector_test = center_function(latent_vector_test)
        latent_vector_val = center_function(latent_vector_val)

        if dimension == 5:

            # Reshape latent space --> Flatten 'nº batch' x 'batch size' to normalize
            latent_vector_train = reshape(
                latent_vector_train,
                (
                    latent_vector_train.shape[0] * latent_vector_train.shape[1],
                    latent_vector_train.shape[2],
                    latent_vector_train.shape[3],
                    latent_vector_train.shape[4],
                ),
            )
            latent_vector_test = reshape(
                latent_vector_test,
                (
                    latent_vector_test.shape[0] * latent_vector_test.shape[1],
                    latent_vector_test.shape[2],
                    latent_vector_test.shape[3],
                    latent_vector_test.shape[4],
                ),
            )
            latent_vector_val = reshape(
                latent_vector_val,
                (
                    latent_vector_val.shape[0] * latent_vector_val.shape[1],
                    latent_vector_val.shape[2],
                    latent_vector_val.shape[3],
                    latent_vector_val.shape[4],
                ),
            )

        # first we merge Latent Space train/test/val
        con = np.concatenate((latent_vector_train, latent_vector_test))
        latent_space = np.concatenate((con, latent_vector_val))

        # Normalize
        if norm:

            latent_space_n = normalize_by_models(latent_space, self.Y_model)
            egm_tensor_n = normalize_by_models(self.egm_tensor, self.Y_model)

        else:

            latent_space_n = latent_space
            egm_tensor_n = self.egm_tensor

        return latent_space_n, egm_tensor_n

    def __call__(self, verbose=False, all=False):
        """Calls the Preprocess class."""
        return self.preprocess_main()
