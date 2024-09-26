import os
from tools_.noise_simulation import *




class LoadDataset:
    '''
    The LoadDataset class loads signals from original format (.mat), performs basic preprocessing in EGMs
    and BSPs and creates variables that store metadata for tracking during the training and validation process

    
    '''
    def init(self, data_type,
        n_classes=2,
        SR=True,
        subsampling=True,
        fs_sub=50,
        SNR=20,
        SNR_bsps = True,
        SNR_em_noise=20,
        SNR_white_noise=20,
        norm=False,
        classification=False,
        sinusoid=False,
        n_batch=50,
        patches_oclussion=None,
        directory=directory, 
        unfold_code=1, 
        inference = True):

        self.data_type = data_type




    def load_data(self,
        data_type,
        n_classes=2,
        SR=True,
        subsampling=True,
        fs_sub=50,
        SNR=20,
        SNR_bsps = True,
        SNR_em_noise=20,
        SNR_white_noise=20,
        norm=False,
        classification=False,
        sinusoid=False,
        n_batch=50,
        patches_oclussion=None,
        directory=directory, 
        unfold_code=1, 
        inference = True
        ):
        """
        Returns y values depends of the classification model

        Parameters:
            data_type: '3channelTensor' -> tx6x4x3 tensor (used for CNN-based classification)
                    '1channelTensor' -> tx6x16 tensor (used for CNN-based classification)

                    'Flat' ->  tx64 matrix (used for MLP-based classification) or tx10
            n_classes: for classification task 1-->
                        '1' -> Rotor/no rotor
                        '2' ->  RA/LA/No rotor (2 classes)
                        '3' ->  7 regions (3 classes) + no rotor (8 classes)
            SR: If Sinus Ryhthm Model is included (True/False)
            subsampling: if downsampling from fs= 500Hz to a lower fs_sub is done (True/False)
            fs_sub: Freq sampl to downsample the signal from its original fs= 500Hz
            norm: if normalizing is performed to models when loading them (True/False)
            classification: if classification task is performed
            sinusoid: if a database of sinusoids is generated (True/False)
            n_batch: batch size (number of samples/batch)


        Returns:
            X -> input data.
            Y -> labels.
            Y_model -> model associated for each time instant.
            egm_tensor -> original EGM values associated to each BSP
            length_list -> length (samples) of each EGM model
            all_model_names -> Name of all AF models loaded
            transfer_matrices -> All transfer matrices used for Forward Problem (*for using them for Tikhonov later)
        """
        fs = 500

        # % Check models in directory
        all_model_names = []

        for subdir, dirs, files in os.walk(directory):
            if subdir != directory:
                model_name = subdir.split("/")[-1]
                if "Sinusal" in model_name and SR == False:
                    continue
                else:
                    all_model_names.append(model_name)

        if sinusoid:
            n = 80  # NUmber of sinusoid models generated
            all_model_names = ["Model {}".format(m) for m in range(n + 1)]
        print(len(all_model_names), "Models")

        # % Load models
        X = []
        Y = []
        Y_model = []
        n_model = 1
        egm_tensor = []
        length_list = []
        AF_models = []

        # Load corrected transfer matrices
        transfer_matrices = self.load_transfer(ten_leads=False, bsps_set=False)

        AF_model_i = 0

        #SNR_em_noise = 10
        #SNR_white_noise=20

        #hacer una estimación de la longitud del array de BSPMs
        len_target_signal = 2000
        Noise_Simulation = NoiseSimulation(SNR_em_noise = SNR_em_noise,
                                        SNR_white_noise=SNR_white_noise,
                                        oclusion = None,
                                        fs=DataConfig.fs_sub) #instance of class

        test_models_deterministic = ['LA_PLAW_140711_arm', 'LA_RSPV_CAF_150115', 'Simulation_01_200212_001_  5',
                'Simulation_01_200212_001_ 10', 'Simulation_01_200316_001_  3',
                'Simulation_01_200316_001_  4', 'Simulation_01_200316_001_  8',
                'Simulation_01_200428_001_004', 'Simulation_01_200428_001_008',
                'Simulation_01_200428_001_010', 'Simulation_01_210119_001_001',
                'Simulation_01_210208_001_002']
        if inference:
            all_model_names=test_models_deterministic
        
        noise_database= Noise_Simulation.configure_noise_database(len_target_signal, all_model_names,
                                                            em = True,
                                                            ma = False,
                                                            gn = True,
                                                            )

        for model_name in all_model_names:

            if inference:
                if model_name not in model_name:
                    break
            print('Loading model', model_name, '......')

            # %% 1)  Compute EGM of the model
            egms = load_egms(model_name, sinusoid)

            
            # 1.1) Discard models <1500
            # if len(egms[1])<1500:
            # continue

            # 1.2)  EGMs filtering.
            x = ECG_filtering(egms, 500)
        
            # 1.3 Normalize models
            if norm:

                high = 1
                low = -1

                mins = np.min(x, axis=0)
                maxs = np.max(x, axis=0)
                rng = maxs - mins

                bsps_64_n = high - (((high - low) * (maxs - x)) / rng)
            

            # 2) Compute the Forward problem with each of the transfer matrices
            for matrix in transfer_matrices:

                # Forward problem
                y = forward_problem(x, matrix[0])
                bsps_64 = y[matrix[1].ravel(), :]
                bsps_64_or = bsps_64
                bsps_64_filt=bsps_64_or

                plt.figure(figsize=(20, 7))
                plt.subplot(2, 1, 1)
                plt.plot(x[0, 0:2000])
                plt.subplot(2, 1, 2)
                plt.plot(y[0, 0:2000])

                plt.title(model_name)
                plt.savefig('output/figures/input_output/forward_problem.png')

                # RESAMPLING signal to fs= fs_sub
                if subsampling:
                    bsps_64 = signal.resample_poly(bsps_64_filt, fs_sub, 500, axis=1)
                    x_sub = signal.resample_poly(x, fs_sub, 500, axis=1)

                    plt.figure(figsize=(20, 7))
                    plt.plot(x[0, 0:2000])
                    plt.title(model_name)
                    plt.savefig('output/figures/input_output/subsample.png')

                else:

                    bsps_64 = bsps_64_filt
                    x_sub = x

                if classification:

                    y_labels = get_labels(n_classes, model_name)
                    y_labels_list = y_labels.tolist()

                    # RESAMPLING labels to fs= fs_sub
                    if subsampling:
                        y_labels = sample(y_labels, len(bsps_64[1]))

                    y_model = np.full(len(y_labels), n_model)
                    Y_model.extend(y_model)
                    Y.extend(y_labels)
                    Y_model.extend(y_model)

                else:

                    Y.extend(np.full(len(x_sub), 0))
                
                #RESHAPE TO TENSOR

                if data_type == "3channelTensor":

                    tensor_model = get_tensor_model(bsps_64, tensor_type="3channel")
                    X.extend(tensor_model)

                elif data_type == "1channelTensor":

                    tensor_model = get_tensor_model(bsps_64, tensor_type="1channel", unfold_code=unfold_code)
                    
                    # Interpo was here *

                    # Noise 
                    start_time = time.time()

                    if SNR_bsps != None:
                        #New noise module
                        #num_patches must be maximum 16 (for 64 electrodes)
                        tensor_model_noisy, map_distribution_noise = Noise_Simulation.add_noise(tensor_model,
                                                                                                AF_model_i,
                                                                                                noise_database,
                                                                                                num_patches = 10,
                                                                                                distribution_noise_mode = 2, 
                                                                                                n_noise_chunks_per_signal = 3)
                    
                    
                        
                    # 5) Filter AFTER adding noise

                    tensor_model_filt = ECG_filtering(tensor_model_noisy, order = 3, fs = fs_sub, f_low=3, f_high=30 )
                    tensor_model=tensor_model_filt 

                    plt.figure(figsize=(20,5))
                    plt.plot(tensor_model_noisy[0:1000, 0, 0], label = 'Noisy')
                    plt.plot(tensor_model[0:1000,   0, 0], label = 'Original')
                    plt.plot(tensor_model_filt[0:1000,   0, 0], label = 'Cleaned')
                    plt.legend()
                    plt.savefig('output/figures/Noise_module/filtered_vs_original.png')

                    #Turn off electrodes

                    if patches_oclussion != 'PT':
                        oclussion= Oclussion(tensor_model, patches_oclussion = patches_oclussion)
                        tensor_model = oclussion.turn_off_patches()

                    # Interpolate
                    tensor_model= interpolate_2D_array(tensor_model)

                    plt.figure(figsize=(20, 7))
                    plt.plot(x_sub[0, 0:2000]) 
                    plt.plot(tensor_model[0:2000, 0, 0])
                    plt.title(model_name)
                    plt.savefig('output/figures/input_output/before_truncate.png')

                    # Truncate length to be divisible by the batch size
                    tensor_model, length_list, x_sub = truncate_length_bsps(n_batch, tensor_model, length_list, x_sub)
                    
                    
                    X.extend(tensor_model)
                    egm_tensor.extend(x_sub.T)


                    #plt.figure(figsize=(20, 7))
                    #plt.plot(x_sub[0, 0:2000]) 
                    #plt.plot(tensors_model[0:2000, 0, 0])
                    #plt.title(model_name)
                    #plt.savefig(model_name)

                    #plt.savefig('output/figures/input_output/saving_truncate.png')

                else:
                    X.extend(bsps_64.T)
                    egm_tensor.extend(x_sub.T)



                if not classification:
                    y_model = np.full(len(tensor_model), n_model)
                    Y_model.extend(y_model)

                    # Count AF Model
                    AF_model_i_array = np.full(len(tensor_model), AF_model_i)
                    AF_models.extend(AF_model_i_array)

                n_model += 1

            AF_model_i += 1

        return (
            np.array(X),
            np.array(Y),
            np.array(Y_model),
            np.array(egm_tensor),
            length_list,
            AF_models,
            all_model_names,
            transfer_matrices,
        )


    def load_transfer(self, ten_leads=False, bsps_set=False):
        """
        Load the transfer matrix for atria and torso models

        Returns:
            MTransfer: Transfer matrix for atria and torso models
        """

        all_torsos_names = []
        for subdir, dirs, files in os.walk(torsos_dir):
            for file in files:
                if file.endswith(".mat"):
                    all_torsos_names.append(file)

        transfer_matrices = []
        for torso in all_torsos_names:
            MTransfer = scipy.io.loadmat(torsos_dir + torso).get("TransferMatrix")

            if ten_leads == True:
                BSP_pos = scipy.io.loadmat(torsos_dir + torso).get("torso")["leads"][0, 0]
            elif bsps_set == True:
                torso_electrodes = loadmat(directory + "torso_electrodes.mat")
                BSP_pos = torso_electrodes["torso_electrodes"][0]
                # BSP_pos = get_bsps_192 (torso, False)
            else:
                BSP_pos = scipy.io.loadmat(torsos_dir + torso).get("torso")["bspmcoord"][
                    0, 0
                ]

            # Transform transfer matrix to account for WCT correction. A matrix is the result of
            # referencing MTransfer to a promediated MTransfer. THe objective is to obtain an ECG
            # referenced to the WCT, following the next expression:
            # ECG_CTW = MTransfer * EGM - M_ctw * MTransfer * EGM =
            # = (MTransfer - M_ctw * MTransfer) * EGM = MTransfer_ctw * EGM

            M_wct = (1 / (MTransfer.shape)[0]) * np.ones(
                ((MTransfer.shape)[0], (MTransfer.shape)[0])
            )
            A = MTransfer - np.matmul(M_wct, MTransfer)

            transfer_matrices.append((A, BSP_pos - 1))

        return transfer_matrices      