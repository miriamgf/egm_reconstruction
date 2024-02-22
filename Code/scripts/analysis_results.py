

norm = True
# Normalize (por muestras)
if norm:
    estimate_egms_n = []

    for model in np.unique(BSPM_test):
        print(model)
        print(model)
        print(model)

        # 1. Normalize Reconstruction
        arr_to_norm_estimate = reconstruction_flat_test[
            np.where((BSPM_test == model))]  # select window of signal belonging to model i
        estimate_egms_norm = normalize_array(arr_to_norm_estimate, 1, -1, 0)  # 0: por muestas
        estimate_egms_n.extend(estimate_egms_norm)  # Add to new norm array
    estimate_egms_n = np.array(estimate_egms_n)

# Plots

for i in range(0, 20):
    interv = random.randrange(0, len(estimate_egms_test) - 1, 50)
    # interv= 6600+i
    node = random.randrange(0, n_nodes, 1)
    # node= 69
    normalize_ = True
    rango = 4 * fs
    # normalize between -1 and 1
    estimate_signal = estimate_egms_n[interv: interv + rango, :]
    # estimate_egms_norm_represent = normalize_array(estimate_signal, 1, -1, 0)
    estimate_egms_norm_represent = estimate_signal

    plt.figure(figsize=(15, 3))

    plt.plot(estimate_egms_norm_represent[:, node], label='Estimation Test')
    # plt.plot(estimate_egms_norm[interv: interv+200 ,node], label='Estimation Test')
    plt.plot(y_test_subsample[interv:interv + rango, node], label='Test', alpha=0.5)
    # plt.plot(latent_vector_test[200:400, 0, 0, 0], label = 'Latent Vector')
    text = 'Node {} in second {} to {}'.format(node, interv / fs, interv / fs + rango / fs)
    plt.legend()
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.title(text)
    plt.savefig('output/figures/' + str(fs_sub) + '/Reconstruction' + str(i))
    plt.show()

plt.figure(layout='tight', figsize=(15, 10))
plt.subplot(1, 3, 1)
plt.imshow(normalize_array(y_test_subsample[0:1000, :].T, 1, 0, 0), cmap='Greys')
plt.title('y test (egm)')
plt.xlabel('time')
plt.ylabel('nodes')
plt.colorbar(orientation="horizontal", pad=0.2)
plt.subplot(1, 3, 2)
plt.imshow(normalize_array(estimate_egms_n[0:1000, :].T, 1, 0, 0), cmap='Greys')
plt.title('Reconstruction')
plt.xlabel('time')
plt.ylabel('nodes')
plt.colorbar(orientation="horizontal", pad=0.2)
plt.subplot(1, 3, 3)
dif = normalize_array(y_test_subsample[0:1000, :].T, 1, 0, 0) - normalize_array(estimate_egms_n[0:1000, :].T, 1, 0, 0)
dif = abs(dif)
plt.imshow(dif, cmap='Greys')
plt.title('y_test - reconstruction')
plt.xlabel('time')
plt.ylabel('nodes')
plt.colorbar(orientation="horizontal", pad=0.2)
plt.savefig('output/figures/' + str(fs_sub) + '/2D_Rec.png')
plt.show()

# Correlation and RMSE

# correlation_pearson_time=corr_pearson_cols(estimate_egms_n.T, y_test_subsample.T)
# correlation_pearson_nodes=corr_pearson_cols(estimate_egms_n, y_test_subsample)
# correlation_spearman_time=corr_spearman_cols(estimate_egms_n.T, y_test_subsample.T)
# correlation_spearman_nodes=corr_spearman_cols(estimate_egms_n, y_test_subsample)


# DF Mapping
DF_Mapping = False

if DF_Mapping:
    unique_test_models = np.unique(AF_models_test)
    for model in unique_test_models:
        # model= AF_models_test[1]
        estimation_array = estimate_egms_n[
            np.where((AF_models_test == model))]  # select window of signal belonging to model i
        y_array = y_test_subsample[np.where((AF_models_test == model))]  # select window of signal belonging to model i

        DF_rec, sig_k_rec, phase_rec = freq_pha.kuklik_DF_phase(estimation_array.T, fs)
        DF_label, sig_k_label, phase_label = freq_pha.kuklik_DF_phase(y_array.T, fs)

        # Interpolate DF Mapping to 2048 nodes

        sig_k_rec_i = interpolate_fun(sig_k_rec.T, len(sig_k_rec.T), 2048, sig=False)
        sig_k_label_i = interpolate_fun(sig_k_label.T, len(sig_k_label.T), 2048, sig=False)
        phase_rec_i = interpolate_fun(sig_k_rec.T, len(phase_rec.T), 2048, sig=False)
        phase_label_i = interpolate_fun(sig_k_label.T, len(phase_label.T), 2048, sig=False)

        print(sig_k_rec_i.shape, 'interpo df')

        # %%
        dic_DF = {"DF_rec": DF_rec, "sig_k_rec": sig_k_rec_i, "phase_rec": phase_rec_i,
                  "DF_label": DF_label, "sig_k_label": sig_k_label_i, "phase_label": phase_label_i}
        savemat("saved_var/" + str(fs_sub) + "/DF_Mapping_variables_" + str(model) + ".mat", dic_DF)

'''
plt.figure(figsize=(15, 5), layout='tight')
plt.subplot(3, 1, 1)
plt.plot(correlation_spearman_time, label='spearman')
plt.plot(correlation_pearson_time, alpha=0.5, label='pearson')
plt.legend()
plt.title('Time')
plt.subplot(2, 1, 2)
plt.plot(correlation_spearman_nodes, label='spearman')
plt.plot(correlation_pearson_nodes, alpha=0.5, label='pearson')
plt.title('Nodes')
plt.savefig('output/figures/'+ str(fs_sub) + '/Correlation.png')
plt.show()

'''
# %%
correlation_list = []

unique_AF_test_models = np.unique(AF_models_test)

for model in unique_AF_test_models:
    # 1. Normalize Reconstruction
    estimation_array = estimate_egms_n[
        np.where((AF_models_test == model))]  # select window of signal belonging to model i
    y_array = y_test_subsample[np.where((AF_models_test == model))]  # select window of signal belonging to model i
    correlation_pearson_nodes = corr_spearman_cols(estimation_array, y_array)
    correlation_list.extend([correlation_pearson_nodes])
correlation_array = np.array(correlation_list)

plt.figure(figsize=(15, 7), layout='tight')

for i in range(0, len(correlation_list) - 1):
    plt.subplot(len(correlation_list) - 1, 1, i + 1)
    plt.plot(correlation_list[i], label=all_model_names[unique_AF_test_models[i]])
    plt.title(all_model_names[unique_AF_test_models[i]])
    plt.xlabel('Nodes')
plt.ylabel('Coefficient of correlation')
plt.suptitle('Spearson correlation')
plt.savefig('output/figures/' + str(fs_sub) + '/Correlation_2.png')
plt.show()

# %%

dtw_array, dtw_array_random = DTW_by_AFModels(AF_models_test, estimate_egms_n, y_test_subsample)
rmse_array = RMSE_by_AFModels(AF_models_test, estimate_egms_n, y_test_subsample)
correlation_array = correlation_by_AFModels(AF_models_test, estimate_egms_n, y_test_subsample)

# %%
# Mean and STD of Spearman Correlation, DTW and RMSE
corr_mean = np.mean(correlation_array, axis=1)
corr_std = np.std(correlation_array, axis=1)
dtw_mean = np.mean(dtw_array, axis=1)
dtw_std = np.std(dtw_array, axis=1)
dtw_mean_random = np.mean(dtw_array_random, axis=1)
dtw_std_random = np.std(dtw_array_random, axis=1)
rmse_mean = np.mean(rmse_array, axis=1)
rmse_std = np.std(rmse_array, axis=1)

# Labels for plotting
# Labels for plotting
labels = []
for i in range(0, len(corr_mean)):
    labels.extend([all_model_names[unique_AF_test_models[i] - 1]])

x_pos = np.arange(len(labels))
x = np.linspace(0, len(labels), len(labels))

# Corr
plt.figure(figsize=(7, 2))
# fig, ax = plt.subplots()
plt.errorbar(x, corr_mean,
             yerr=corr_std,
             fmt='--o', label='Test')

# ax.set_xticks(x_pos)
# plt.xticks(rotation=90)
# ax.set_xticklabels(labels)
plt.tight_layout()
plt.xlabel('AF Model')
plt.title('Spearman Correlation in test models')
plt.savefig('output/figures/' + str(fs_sub) + '/Corr3.png')

plt.show()

# DTW
plt.figure(figsize=(7, 2))
# fig, ax = plt.subplots()
plt.errorbar(x, dtw_mean,
             yerr=dtw_std,
             fmt='--o', label='Test')
plt.errorbar(x, dtw_mean_random,
             yerr=dtw_std_random,
             fmt='--o', label='Random')
# plt.xticks(rotation=90)
# ax.set_xticklabels(labels)
plt.tight_layout()
plt.xlabel('AF Model')
plt.title('DTW in test models')
plt.legend()
plt.savefig('output/figures/' + str(fs_sub) + '/DTW.png')

plt.show()

# RMSE
plt.figure(figsize=(7, 2))
# fig, ax = plt.subplots()
plt.errorbar(x, rmse_mean,
             yerr=rmse_std,
             fmt='--o')
# plt.xticks(rotation=90)
# ax.set_xticklabels(labels)
plt.tight_layout()
plt.xlabel('AF Model')
plt.title('RMSE in test models')
plt.savefig('output/figures/' + str(fs_sub) + '/RMSE.png')
plt.show()

# plot_Welch_reconstruction(latent_vector_test, estimate_egms_n, fs, n_nodes,y_test_subsample, nperseg_value=100)

LS_signal = x_test
output_signal = estimate_egms_n
nperseg_value = 2 * fs

fig = plt.figure(layout='tight', figsize=(10, 6))

plt.subplot(3, 1, 1)

# reconstruction
for height in range(0, 511, 5):
    f, Pxx_den = scipy.signal.welch(output_signal[:, height], fs, nperseg=nperseg_value,
                                    noverlap=nperseg_value // 2, scaling='density', detrend='linear')
    plt.plot(f, Pxx_den, linewidth=0.5)
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD [V**2/Hz]')
    # plt.ylim([0,0.2])

    titlee = "PSD Welch of EGM signal estimation. node {}".format(height)
    plt.title('EGM signals reconstruction')

# Latent Space
plt.subplot(3, 1, 2)
for height in range(0, 3):
    for width in range(0, 4):
        for filters in range(0, 12, 2):
            f, Pxx_den = scipy.signal.welch(LS_signal[:, height, width, filters], fs, nperseg=nperseg_value,
                                            noverlap=nperseg_value // 2, scaling='density', detrend='linear')
            plt.plot(f, Pxx_den, linewidth=0.5)
            plt.xlabel('frequency [Hz]')
            plt.ylabel('PSD [V**2/Hz]')
            titlee = "PSD Welch of Latent Space signal. nodes ( - )".format(height, width)
            plt.title('latent Space')
            # plt.ylim([0,0.2])

fig.suptitle("Welch Periodogram (window size=200 samples)", fontsize=15)

# Input
plt.subplot(3, 1, 3)

for height in range(0, 511, 5):
    f, Pxx_den = scipy.signal.welch(y_test_subsample[:, height], fs, nperseg=nperseg_value,
                                    noverlap=nperseg_value // 2, scaling='density', detrend='linear')
    plt.plot(f, Pxx_den, linewidth=0.5)
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD [V**2/Hz]')
    titlee = "PSD Welch of EGM signal estimation. node {}".format(height)
    plt.title('EGM signals Real')
    # plt.ylim([0,0.2])

plt.savefig('output/figures/' + str(fs_sub) + '/Periodogram_rec.png')
plt.show()