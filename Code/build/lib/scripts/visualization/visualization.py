import sys

sys.path.append("../Code")
import matplotlib.pyplot as plt
import pickle
from tools_.preprocessing_network import *
from tools_.plots import *
from config import TrainConfig_1
from config import TrainConfig_2
from config import DataConfig
import random
from evaluate_function import reshape_tensor

fs = 50
figs_dir = "output/figures/20240410-192639"

experiment_dir = figs_dir
if not os.path.exists(experiment_dir):
    os.makedirs(experiment_dir)
    print("Directory for experiment", experiment_dir, "created")
else:
    experiment_dir = experiment_dir + "_1"
    print(
        "Existing directory, name changed for avoiding rewriting information to:",
        experiment_dir,
    )
    figs_dir = experiment_dir
    os.makedirs(experiment_dir)


dict_var_dir = "output/variables/"
dict_results_dir = "output/results/"


# Load dictionary

data_to_read = open(dict_var_dir + "variables_P3.pkl", "rb")
variables = pickle.load(data_to_read)
print("variables loaded...")

data_to_read = open(dict_results_dir + "dict_results_autoencoder_P3.pkl", "rb")
dict_results_autoencoder = pickle.load(data_to_read)

data_to_read = open(dict_results_dir + "dict_results_reconstruction_P3.pkl", "rb")
dict_results_reconstruction = pickle.load(data_to_read)

# Extraer las variables al espacio de trabajo global
globals().update(variables)
globals().update(dict_results_autoencoder)
globals().update(dict_results_reconstruction)


# ------------------------- Plot before training ------------------------------

# Batch generation

# Batch generation
x_train_batch = reshape(
    x_train_raw,
    (
        int(len(x_train_raw) / TrainConfig_1.batch_size_1),
        TrainConfig_1.batch_size_1,
        x_train_raw.shape[1],
        x_train_raw.shape[2],
        1,
    ),
)
x_test_batch = reshape(
    x_test_raw,
    (
        int(len(x_test_raw) / TrainConfig_1.batch_size_1),
        TrainConfig_1.batch_size_1,
        x_test_raw.shape[1],
        x_test_raw.shape[2],
        1,
    ),
)
x_val_batch = reshape(
    x_val_raw,
    (
        int(len(x_val_raw) / TrainConfig_1.batch_size_1),
        TrainConfig_1.batch_size_1,
        x_val_raw.shape[1],
        x_val_raw.shape[2],
        1,
    ),
)


# Reshape AF_models to compute corresponding AF model to each sample
x_train_batch_AF = reshape(
    AF_models_train,
    (
        int(len(AF_models_train) / TrainConfig_1.batch_size_1),
        TrainConfig_1.batch_size_1,
    ),
)
x_test_batch_AF = reshape(
    AF_models_test,
    (int(len(AF_models_test) / TrainConfig_1.batch_size_1), TrainConfig_1.batch_size_1),
)
x_val_batch_AF = reshape(
    AF_models_val,
    (int(len(AF_models_val) / TrainConfig_1.batch_size_1), TrainConfig_1.batch_size_1),
)

# Plot random batches
a = randint(0, x_train_batch.shape[0] - 1)  # Random batch


plt.figure(layout="tight", figsize=(10, 10))
for i in range(0, TrainConfig_1.batch_size_1):
    plt.subplot(10, 10, i + 1)
    plt.imshow(x_train_batch[a, i, :, :, 0])
    plt.title(str(i) + "- AF model" + str(x_train_batch_AF[a, i]))
plt.suptitle("Train Batch" + str(a))
plt.savefig(figs_dir + "/Batch1.png")
plt.show()


a = randint(0, x_test_batch.shape[0] - 1)  # Random batch
plt.figure(layout="tight", figsize=(10, 10))
for i in range(0, TrainConfig_1.batch_size_1):
    plt.subplot(10, 10, i + 1)
    plt.imshow(x_test_batch[a, i, :, :, 0])
    plt.title("AF model" + str(x_test_batch_AF[a, i]))
plt.suptitle("Test Batch" + str(a))
plt.savefig(figs_dir + "/Batch2.png")
plt.show()

X_1channel = Original_X_1channel
plotting_before_AE(X_1channel, x_train_raw, x_test_raw, x_val_raw, Y_model, AF_models)

# ------------------------- 2. Plot after training AE ------------------------------


#
time_instant = random.randint(0, 49)
batch = random.randint(0, x_test.shape[0])

print(batch, time_instant)

plt.figure(figsize=(5, 5), layout="tight")
plt.subplot(3, 1, 1)
# image1=define_image(time_instant,x_test )
image1 = x_test[batch, time_instant, ::]
min_val, max_val = np.amin(image1), np.amax(image1)
plt.imshow(image1, vmin=min_val, vmax=max_val)  # , cmap='jet')
plt.colorbar()
plt.title("Original")
plt.subplot(3, 1, 2)
# image2=define_image(time_instant,pred_test)
image2 = pred_test[batch, time_instant, ::]
plt.imshow(image2, vmin=min_val, vmax=max_val)  # , cmap='jet')
plt.colorbar()
plt.title("Reconstructed")
plt.subplot(3, 1, 3)
plt.imshow(image2 - image1, vmin=min_val, vmax=max_val)
plt.title("Error (Reconstructed-Original)")
plt.colorbar()
plt.savefig(figs_dir + "/AE_reconstruction.png")
plt.show()

# PLOTS 50 samples --> 1 second
batch = random.randint(0, x_test.shape[0])

# Random signal visualizer: in each execution plots the first second of the BSPS in random nodes
plt.figure(figsize=(10, 3))
a = randint(0, pred_test.shape[2] - 1)
b = randint(0, pred_test.shape[3] - 1)
# s=randint(0, len(pred_test)-fs-1 )

title = "50 samples from BSP from nodes {} x {} in the model {}".format(
    a, b, x_test_batch_AF[batch, 0]
)
plt.plot(x_test[batch, :, a, b], label=("Input"))
plt.plot(pred_test[batch, :, a, b], label=("Reconstruction"))
plt.fill_between(
    np.arange(TrainConfig_1.batch_size_1),
    pred_test[batch, :, a, b, 0],
    x_test[batch, :, a, b, 0],
    color="lightgreen",
)
plt.ylabel("Amplitude")
plt.xlabel("Samples")
plt.legend()
plt.title(title)
plt.savefig(figs_dir + "/AE_Recontruction2.png")

plt.show()

# %%

# create function to center data
center_function = lambda x: x - x.mean(axis=0)

# Center latent space
latent_vector = center_function(latent_vector_test)


# Plot latent space frames in random time instant

time_instant = randint(0, 49)
batch = random.randint(0, x_test.shape[0])

plt.figure(figsize=(6, 6), layout="tight")
title = "Filters from Latent Space in time instant {} ".format(time_instant)
for i in range(0, 12):
    plt.subplot(6, 2, i + 1)
    # plt.plot(6, 2, i)
    plt.imshow(latent_vector[batch, time_instant, :, :, i])  # , cmap='jet')
    plt.colorbar()
plt.suptitle(title)
plt.savefig(figs_dir + "/Latentspace1.png")

plt.show()


plt.figure()
plt.plot(x_test[batch, :, 1, 1], label="Input")  # , cmap='jet')

for i in range(0, 12):
    # plt.plot(6, 2, i)
    plt.plot(latent_vector[batch, :, 1, 1, i], alpha=0.5, label=i)  # , cmap='jet')
    plt.title("Feature vector in time (50 samples) of 12 filters")
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.savefig(figs_dir + "/Latentspace2.png")

plt.show()


# Flatten
x_train_fl = reshape(
    x_train, (x_train.shape[0] * x_train.shape[1], x_train.shape[2], x_train.shape[3])
)
x_test_fl = reshape(
    x_test, (x_test.shape[0] * x_test.shape[1], x_test.shape[2], x_test.shape[3])
)
x_val_fl = reshape(
    x_val, (x_val.shape[0] * x_val.shape[1], x_val.shape[2], x_val.shape[3])
)
pred_test_fl = reshape(
    pred_test,
    (pred_test.shape[0] * pred_test.shape[1], pred_test.shape[2], pred_test.shape[3]),
)
# latent_vector = reshape(latent_vector, (
# latent_vector.shape[0] * latent_vector.shape[1], latent_vector.shape[2], latent_vector.shape[3],
# latent_vector.shape[4]))

# Plot of reconstruction (orange) over the context of 14 seconds of random input signal

# PLOT INPUT VS OUTPUT
window_length_test = 500  # samples
window_length_subsegment = 200
t_vector = np.array(np.linspace(0, window_length_test / fs, window_length_test))
t_vector_reshaped = t_vector.reshape(t_vector.shape[0], 1, 1)

# Random big window to show x_test
randomonset_test = random.randrange(len(x_test_fl) - window_length_test)
random_window_test = [randomonset_test, randomonset_test + window_length_test]

x_test_cut = x_test_fl[random_window_test[0] : random_window_test[1], 1, 1]
pred_test_cut = pred_test_fl[random_window_test[0] : random_window_test[1], 1, 1]

# smaller window to show subsegment inside x_test
randomonset = random.randrange(len(x_test_cut) - window_length_subsegment)
random_window_subsegment = [randomonset, randomonset + window_length_subsegment]
print(pred_test_cut.size, "total length", random_window_subsegment, "subwindow size")
copy_pred_test_cut = pred_test_cut
pred_test_cut[: random_window_subsegment[0]] = None
pred_test_cut[random_window_subsegment[1] :] = None

plt.figure(figsize=(10, 2))
plt.plot(t_vector, x_test_cut, alpha=0.5, label=("Test"))
plt.plot(t_vector, pred_test_cut, label=("Reconstruction"))
plt.xlabel("Time(s)")
plt.ylabel("mV")
plt.title("14 seconds of Test BSPS. Over it, 2 seconds of BSPS reconstruction")
plt.savefig(figs_dir + "/BSPS_AE.png")
plt.show()

# PSD Welch of input, output and Latent Space

# plot_Welch_periodogram(x_test_fl, latent_vector, pred_test_fl, fs=fs)
# plt.savefig(figs_dir  + '/Periodogram_AE.png')


# ------------------------- 3. Plot after training Reconstruction ------------------------------
y_test = y_test_ls
pred_test = pred_test_egm

y_test_flat = reshape_tensor(y_test, n_dim_input=y_test.ndim, n_dim_output=2)
reconstruction_flat_test = reshape_tensor(
    pred_test, n_dim_input=pred_test.ndim, n_dim_output=2
)

y_test_subsample = y_test_flat
estimate_egms_test = reconstruction_flat_test
estimate_egms_n = normalize_by_models(reconstruction_flat_test, BSPM_test)

for i in range(0, 30):
    interv = random.randrange(0, len(estimate_egms_test) - 1, 50)
    # interv= 6600+i
    node = random.randrange(0, TrainConfig_2.batch_size_2, 1)
    # node= 69
    normalize_ = True
    rango = 4 * fs
    # normalize between -1 and 1
    estimate_signal = estimate_egms_n[interv : interv + rango, :]
    # estimate_egms_norm_represent = normalize_array(estimate_signal, 1, -1, 0)
    estimate_egms_norm_represent = estimate_signal

    plt.figure(figsize=(15, 3))

    plt.plot(estimate_egms_norm_represent[:, node], label="Estimation Test")
    # plt.plot(estimate_egms_norm[interv: interv+200 ,node], label='Estimation Test')
    plt.plot(y_test_subsample[interv : interv + rango, node], label="Test", alpha=0.5)
    # plt.plot(latent_vector_test[200:400, 0, 0, 0], label = 'Latent Vector')
    text = "Node {} in second {} to {}".format(
        node, interv / fs, interv / fs + rango / fs
    )
    plt.legend()
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.title(text)
    plt.savefig(figs_dir + "/Reconstruction" + str(i))
    plt.show()

plt.figure(layout="tight", figsize=(15, 10))
plt.subplot(1, 3, 1)
plt.imshow(normalize_array(y_test_subsample[0:1000, :].T, 1, 0, 0), cmap="Greys")
plt.title("y test (egm)")
plt.xlabel("time")
plt.ylabel("nodes")
plt.colorbar(orientation="horizontal", pad=0.2)
plt.subplot(1, 3, 2)
plt.imshow(normalize_array(estimate_egms_n[0:1000, :].T, 1, 0, 0), cmap="Greys")
plt.title("Reconstruction")
plt.xlabel("time")
plt.ylabel("nodes")
plt.colorbar(orientation="horizontal", pad=0.2)
plt.subplot(1, 3, 3)
dif = normalize_array(y_test_subsample[0:1000, :].T, 1, 0, 0) - normalize_array(
    estimate_egms_n[0:1000, :].T, 1, 0, 0
)
dif = abs(dif)
plt.imshow(dif, cmap="Greys")
plt.title("y_test - reconstruction")
plt.xlabel("time")
plt.ylabel("nodes")
plt.colorbar(orientation="horizontal", pad=0.2)
plt.savefig(figs_dir + "/2D_Rec.png")
plt.show()

# Correlation and RMSE

# correlation_pearson_time=corr_pearson_cols(estimate_egms_n.T, y_test_subsample.T)
# correlation_pearson_nodes=corr_pearson_cols(estimate_egms_n, y_test_subsample)
# correlation_spearman_time=corr_spearman_cols(estimate_egms_n.T, y_test_subsample.T)
# correlation_spearman_nodes=corr_spearman_cols(estimate_egms_n, y_test_subsample)

# ------------------------- 4. Plot CORRELATION and Frequency ------------------------------

#!!!!!!!!!!!!!!!!!!!! PLOT[3] !!!!!!!!!!!!!!!!!!!!!!!!!!!!
unique_AF_test_models = np.unique(AF_models_test)
# Labels for plotting
labels = []
for i in range(0, len(corr_mean)):
    labels.extend([all_model_names[unique_AF_test_models[i] - 1]])

x_pos = np.arange(len(labels))
x = np.linspace(0, len(labels), len(labels))

# Corr
plt.figure(figsize=(7, 2))
# fig, ax = plt.subplots()
plt.errorbar(x, corr_mean, yerr=corr_std, fmt="--o", label="Test")

# ax.set_xticks(x_pos)
# plt.xticks(rotation=90)
# ax.set_xticklabels(labels)
plt.tight_layout()
plt.xlabel("AF Model")
plt.title("Spearman Correlation in test models")
plt.savefig(figs_dir + "/Corr3.png")

plt.show()

"""
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
plt.savefig(figs_dir  + '/DTW.png')
plt.show()
"""

# RMSE
plt.figure(figsize=(7, 2))
# fig, ax = plt.subplots()
plt.errorbar(x, rmse_mean, yerr=rmse_std, fmt="--o")
# plt.xticks(rotation=90)
# ax.set_xticklabels(labels)
plt.tight_layout()
plt.xlabel("AF Model")
plt.title("RMSE in test models")
plt.savefig(figs_dir + "/RMSE.png")
plt.show()

# plot_Welch_reconstruction(latent_vector_test, estimate_egms_n, fs, n_nodes,y_test_subsample, nperseg_value=100)


LS_signal = latent_vector_test
output_signal = estimate_egms_n
nperseg_value = 2 * fs
LS_signal = reshape_tensor(LS_signal, LS_signal.ndim, 4)

fig = plt.figure(layout="tight", figsize=(10, 6))

plt.subplot(3, 1, 1)

# reconstruction
for height in range(0, 511, 5):
    f, Pxx_den = scipy.signal.welch(
        output_signal[:, height],
        fs,
        nperseg=nperseg_value,
        noverlap=nperseg_value // 2,
        scaling="density",
        detrend="linear",
    )
    plt.plot(f, Pxx_den, linewidth=0.5)
    plt.xlabel("frequency [Hz]")
    plt.ylabel("PSD [V**2/Hz]")
    # plt.ylim([0,0.2])

    titlee = "PSD Welch of EGM signal estimation. node {}".format(height)
    plt.title("EGM signals reconstruction")

# Latent Space
plt.subplot(3, 1, 2)
for height in range(0, 3):
    for width in range(0, 4):
        for filters in range(0, 12, 2):
            f, Pxx_den = scipy.signal.welch(
                LS_signal[:, height, width, filters],
                fs,
                nperseg=nperseg_value,
                noverlap=nperseg_value // 2,
                scaling="density",
                detrend="linear",
            )
            plt.plot(f, Pxx_den, linewidth=0.5)
            plt.xlabel("frequency [Hz]")
            plt.ylabel("PSD [V**2/Hz]")
            titlee = "PSD Welch of Latent Space signal. nodes ( - )".format(
                height, width
            )
            plt.title("latent Space")
            # plt.ylim([0,0.2])

fig.suptitle("Welch Periodogram (window size=200 samples)", fontsize=15)

# Input
plt.subplot(3, 1, 3)

for height in range(0, 511, 5):
    f, Pxx_den = scipy.signal.welch(
        y_test_subsample[:, height],
        fs,
        nperseg=nperseg_value,
        noverlap=nperseg_value // 2,
        scaling="density",
        detrend="linear",
    )
    plt.plot(f, Pxx_den, linewidth=0.5)
    plt.xlabel("frequency [Hz]")
    plt.ylabel("PSD [V**2/Hz]")
    titlee = "PSD Welch of EGM signal estimation. node {}".format(height)
    plt.title("EGM signals Real")
    # plt.ylim([0,0.2])

plt.savefig(figs_dir + "/Periodogram_rec.png")
plt.show()
