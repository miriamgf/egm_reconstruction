import sys

sys.path.append(
    "/home/profes/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/"
)

from models.autoencoder_model import autoencoder
from models.autoencoder_model import autoencoder
from models.autoencoder_model import autoencoder
from models.reconstruction_model import reconstruction
from config import TrainConfig_1
from config import TrainConfig_2
from config import DataConfig
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.optimizers import Adam
from tools_ import freq_phase_analysis as freq_pha
from tools_ import plots
from tools_.preprocessing_network import *
from tools_.tools import *
from tools_.df_mapping import *
import tensorflow as tf
import os
import scipy
import datetime
import time
from evaluate_function import *
from numpy import *
import pickle
import sys
import mlflow
import itertools

sys.path.append("/code")
tf.random.set_seed(42)

root_logdir = "../output/logs/"
log_dir = root_logdir + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
data_dir = "../../../Data/"
torsos_dir = "../../../Labeled_torsos/"
figs_dir = "../output/"
models_dir = "../output/model/"
dict_var_dir = "../output/variables/"
dict_results_dir = "../output/results/"


dic_vars = {}
dict_results = {}

# GPU Configuration
physical_devices = tf.config.list_physical_devices("GPU")
print("Num GPUs:", len(physical_devices))
for gpu in tf.config.experimental.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

start = time.time()

all_torsos_names = []
for subdir, dirs, files in os.walk(torsos_dir):
    for file in files:
        if file.endswith(".mat"):
            all_torsos_names.append(file)


all_model_names = []
directory = data_dir
for subdir, dirs, files in os.walk(directory):
    # print(subdir, directory, files)

    if subdir != directory:
        model_name = subdir.split("/")[-1]
        all_model_names.append(model_name)

all_model_names = sorted(all_model_names)
# print(len(all_model_names))

mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
mlflow.autolog()

# Load data

if DataConfig.fs == DataConfig.fs_sub:
    DataConfig.fs = DataConfig.fs_sub

Transfer_model = False  # Transfer learning from sinusoids
sinusoids = False


# Load data
df = load_egms_df(directory)

df.head(10)
df = df.sort_values(by="id")


signals = df["AF_signal"].values
labels = df["id"].values

signals_array = np.array(signals)
for i, signal in enumerate(signals_array):
    signals_array[i] = signal[:, :500]

signals_array_reshaped = np.stack(signals_array)
signals_array_reshaped_flattened = signals_array_reshaped.reshape(df.shape[0], -1)
correlation_matrix = np.corrcoef(signals_array_reshaped_flattened)

plt.figure(figsize=(20, 20))
plt.imshow(correlation_matrix, cmap="coolwarm", interpolation="nearest")
plt.colorbar()  # Agrega una barra de color para mostrar los valores
plt.title("Matriz de correlación")
plt.xlabel("Señal")
plt.ylabel("Señal")
plt.xticks(ticks=np.arange(len(labels)), labels=labels, rotation=90)
plt.yticks(ticks=np.arange(len(labels)), labels=labels)
plt.savefig("output/figures/" + "matrix.png")
plt.show()
sys.path.append("../Code")


# Crear una matriz con ceros y unos donde 1 representa una correlación mayor que el umbral
correlation_mask = np.where(correlation_matrix > 0.5, 1, 0)

# Visualizar la matriz de correlación resaltando solo las correlaciones mayores que el umbral
plt.figure(figsize=(8, 6))
plt.imshow(correlation_mask, cmap="binary", interpolation="nearest")
plt.title("Matriz de correlación (Umbral: > 0.4)")
plt.xlabel("Señal")
plt.ylabel("Señal")
plt.savefig("output/figures/" + "matrix_umbral.png")
plt.show()

# Obtener todas las combinaciones posibles de pares de señales
combinaciones = itertools.combinations(range(len(labels)), 2)

# Obtener las parejas de señales que superan el umbral de correlación sin tener en cuenta las parejas invertidas
parejas_correlacion = []
for i, j in combinaciones:
    if correlation_matrix[i, j] > 0.6:
        # Agregar la pareja de señales solo si la señal i es menor que la señal j
        if i < j:
            parejas_correlacion.append((labels[i], labels[j], correlation_matrix[i, j]))

# Imprimir las parejas de señales con correlación superior al umbral
for pareja in parejas_correlacion:
    print(f"Pareja: {pareja[0]} - {pareja[1]}, Correlación: {pareja[2]}")

df_parejas_correlacion = pd.DataFrame(
    parejas_correlacion, columns=["Señal 1", "Señal 2", "Correlación"]
)

# Imprimir el DataFrame resultante
df_parejas_correlacion = df_parejas_correlacion.sort_values(by="Señal 1")
print(df_parejas_correlacion)

# Guardar el DataFrame como un archivo CSV
df_parejas_correlacion.to_csv(
    "output/figures/" + "parejas_correlacion.csv", index=False
)
valores_unicos_lista = df_parejas_correlacion["Señal 1"].unique().tolist()
with open("output/figures/problematic_signals.txt", "w") as archivo:
    # Escribir cada valor de la lista en una línea del archivo
    for valor in valores_unicos_lista:
        archivo.write(str(valor) + "\n")


plt.figure()
plt.plot()


# Load egms and compute DF

# Load data
df = load_egms_df(data_dir)
df = df.sort_values(by="id")
fs = 500
fun_freq = []
for i in range(0, df.shape[0]):
    id = df["id"][i]
    sig = np.array(df["AF_signal"][df["id"] == id])[0]
    fft_signal = np.fft.fft(sig[0, :])
    fundamental_index = np.argmax(np.abs(fft_signal[1 : len(fft_signal) // 2])) + 1
    fundamental_freq = fundamental_index * fs / len(sig)
    fun_freq.append(fundamental_freq)

df["fun_freq"] = fun_freq
df = df.drop(columns=["AF_signal"])

df.to_csv("output/figures/" + "df_freq.csv", index=False)
fun_freq


"""
# Datos de ejemplo
x = range(10000)
y = [i**2 for i in x]

# Crear la figura y los ejes
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)

# Dibujar los datos
line, = ax.plot(x, y)

# Agregar un slider para ajustar la posición de la ventana
ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
window_slider = Slider(ax_slider, 'Ventana', 0, 10000, valinit=0)

def update(val):
    window_start = int(window_slider.val)
    window_end = window_start + 200  # Tamaño de la ventana
    ax.set_xlim(window_start, window_end)
    fig.canvas.draw_idle()

window_slider.on_changed(update)

plt.show()
"""
