% wavelet_filter - Perform wavelet decomposition and reconstruction on signals
%
% Author: Angélica Quadros
% HeartLab
% 2024
%
% This function performs wavelet filtering by decomposition and reconstruction on signals.
% Different wavelet types and sets of reconstruction levels can be applied.
% It decomposes each signal using the specified wavelet type and number of levels,
% and then reconstructs the signal using the selected reconstruction levels.
% Optionally, it can save the MRA matrices for each electrode. 
% 
%
% Usage:
%   wavelet_filter(D_EL, fs, electrodes, waveletTypes, numLevels, reconstructionLevelsSets, saveMRA)
%
% Inputs:
%   - D_EL: Raw signals, being each row an electrode and each column one sample.
%   - fs: Sampling frequency of the signals.
%   - electrodes: Vector containing the electrode numbers, can be one or more.
%   - waveletTypes: Cell array of strings containing the wavelet types to be used.
%   - numLevels: Number of decomposition levels.
%   - reconstructionLevelsSets: Cell array of vectors specifying which levels will be used for reconstruction.
%   - saveMRA (optional): Boolean indicating whether to save the MRA matrices (decomposition levels variables). Default is false.
%                         If the goal is only filtering, leave this parameter empty.
%
% Outputs:
%     - It returns the variable data_filtered containing the filtered signals.
%     - Reconstructed signals named according to the wavelet type, set of reconstruction levels,
%     and the suffix "_Reconstructed". For example, "Signal_reconstructed_db4_Levels_1_5".
%     - If saveMRA is true, the function creates MRA matrices and stores them as variables
%     in the workspace. The variables are named according to the electrode, wavelet type,
%     set of reconstruction levels, and the suffix "_MRA". For example, "MRA_db4_Levels_1_5_Electrode_1".
%
%
% This function performs wavelet decomposition and reconstruction on signals
% from electrodes using different wavelet types and sets of reconstruction levels.
% It decomposes each signal using the specified wavelet type and number of levels,
% and then reconstructs the signal using the selected reconstruction levels.
% Optionally, it can save the MRA matrices for each electrode and configuration.
%
% Example:
%   wavelet_filter(D_EL, 3600, [1:16], {'db4', 'coif1'}, 10, {[6,7], [7,8,9]});
%
% This example performs wavelet decomposition and reconstruction on signals from
% electrodes 1 to 16 using 'db4' and 'coif1' wavelet types, 10 decomposition levels,
% and using two sets of reconstruction levels: [6,7] and [7,8,9].
%
%
%
function data_filtered = wavelet_filter(D_EL, fs, electrodes, waveletTypes, numLevels, reconstructionLevelsSets, saveMRA)
    
    % get the original frequency sample and set the target frequency sample
    original_fs = fs;  
    new_fs = 3600;

    % check if saveMRA is provided, if not, set to false
    if nargin < 7
        saveMRA = false;
    end
    
    % Initialize data_filtered to store all filtered signals
    data_filtered = [];
    
    % Iterate over each wavelet type
    for j = 1:numel(waveletTypes)
        currentWavelet = waveletTypes{j};
        
        % Iterate over each set of reconstruction levels
        for i = 1:numel(reconstructionLevelsSets)
            currentLevels = reconstructionLevelsSets{i};

            % Format levels into a string to name the variables
            levelString = strjoin(arrayfun(@num2str, currentLevels, 'UniformOutput', false), '_');
            
            % Initialize variable to store reconstructed signals for all electrodes
            Signal_reconstructed_all = [];
            
            % Iterate over each electrode
            for k = 1:numel(electrodes)
                electrode = electrodes(k);
                
                % Original signal
                Signal = D_EL(electrode, :);

                %resampling the signal
                Signal = resample(Signal, new_fs, original_fs);

                % Applying conversion factor to microvolts
                factor_conv = 0.1949999928474426;
                Signal = Signal * factor_conv;

                % Decomposition using modwt
                wt = modwt(Signal, currentWavelet, numLevels);

                % Construct MRA matrix using modwtmra
                mra_temp = modwtmra(wt, currentWavelet);
                
                % Saving on workspace MRA matrix for each electrode if saveMRA is true
                if saveMRA
                    variable_name_mra = ['MRA_' currentWavelet '_Levels_' levelString '_Electrode_' num2str(electrode)];
                    assignin('caller', variable_name_mra, mra_temp);
                end

                % Create the logical array for selecting levels to be used in reconstruction
                levelForReconstruction = false(1, numLevels);
                levelForReconstruction(currentLevels) = true; % set to true the desired ones

                % Signal reconstruction by summing the selected rows
                Signal_reconstructed = sum(mra_temp(levelForReconstruction, :), 1);

                %resampling the signal to the original frequency
                Signal_reconstructed = resample(Signal_reconstructed, original_fs, new_fs);
                
                % Concatenate reconstructed signal from the current electrode to Signal_reconstructed_all
                Signal_reconstructed_all = [Signal_reconstructed_all; Signal_reconstructed];
            end
            
            % Save the reconstructed signals on workspace
            variable_name_signals = ['Signal_reconstructed_' currentWavelet '_Levels_' levelString];
            assignin('caller', variable_name_signals, Signal_reconstructed_all);

            % Assigning to data_filtered all the reconstructed signals
            data_filtered = Signal_reconstructed_all;
        end
    end
end