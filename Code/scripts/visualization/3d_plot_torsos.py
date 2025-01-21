import sys
sys.path.append("../Code")
import numpy as np
import scipy.io as sio
import vtk
from scipy.interpolate import interp1d
from vtk.util.numpy_support import numpy_to_vtk
from scripts.visualization.utils.renderizer import EGMRenderer_BSP
import os
import cv2
import imageio
import matplotlib.pyplot as plt
from PIL import Image
from scripts.config import ParseHiperparams
from tools_.load_dataset import LoadDataset_BSPS

torso_num=2
model_path=f"/home/pdi/miriamgf/tesis/Autoencoders/Labeled_torsos/Torso{torso_num}_mod.mat"
model_name = ["Simulation_01_200316_001_  3"]


# Cargar datos
#model_path = "/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/experiments/experiments_VAE/OMAMI_weighted/reconstructions_by_model_OMAMI_weighted.mat"

geom_path_CF = "/home/pdi/miriamgf/tesis/Autoencoders/geometries/Atria_geom/Modelos_computacionales_Carlos_Fambuena/Atria.mat"
geom_path_edgar= "/home/pdi/miriamgf/tesis/Autoencoders/geometries/Atria_geom/Modelos_Edgar/Atria.mat"
model_path_database= "/home/pdi/miriamgf/tesis/Autoencoders/Data/modelLA_RSPV_CAF_150115/EGMs.mat"
output_directory = "/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/renderized_heart/OMAMI_bs_200"
os.makedirs(output_directory, exist_ok=True)
data_dir = "/home/profes/miriamgf/tesis/Autoencoders/Data/"
torsos_dir = "/home/profes/miriamgf/tesis/Autoencoders/Labeled_torsos/"


# Cargar datos del modelo y la geometría

all_torsos_names = []
for subdir, dirs, files in os.walk(torsos_dir):
    for file in files:
        if file.endswith(".mat"):
            all_torsos_names.append(file)

#Load geometry



SNR_em_noise = None
SNR_white_noise = 100
patches_oclussion = "PT"
experiment_number = 0
unfold_code = 1

params = ParseHiperparams().parse_default_hyperparams()

# Load data
(
    X_1channel,
    Y,
    Y_model,
    egm_tensor,
    length_list,
    AF_models,
    all_model_names,
    transfer_matrices,
    y_list
) = LoadDataset_BSPS(
    params,
    directory=data_dir,
    data_type="1channelTensor",
    n_classes=params["n_classes"],
    downsampling=False,
    fs=params["fs"],
    norm=False,
    SR=True,
    n_batch=params["batch_size"],
    sinusoid=False,
    SNR_em_noise=SNR_em_noise,
    SNR_white_noise=SNR_white_noise,
    patches_oclussion=patches_oclussion,
    unfold_code=unfold_code,
    inference=False,
    select_model = model_name
)()

sys.exit()


torso_name=f"Torso{torso_num}_mod.mat"
torso_index = all_torsos_names.index(torso_name)
bspm_signal=y_list[torso_index]['y']

# Plot BSPM in image format (64 electrodes)
X_1channel_split = np.array_split(X_1channel, 10)
bsp_64 =  X_1channel_split[torso_index]
bsp_64_reshaped = bsp_64.reshape(2001, -1)
normalizar=True
if normalizar:
    bsp_64_n = np.zeros_like(bsp_64_reshaped)
    for nodo in range(bsp_64_reshaped.shape[1]):  
        min_val = np.min(bsp_64_reshaped[:, nodo])
        max_val = np.max(bsp_64_reshaped[:, nodo])
        
        if max_val != min_val:  
            bsp_64_n[:, nodo] = (bsp_64_reshaped[:, nodo] - min_val) / (max_val - min_val)
        else:
            bsp_64_n[:, nodo] = 0  
bsp_64= bsp_64_n.reshape(2001, 12,32)

output_video = os.path.join(output_directory, f"video_bsp_64.gif")
filenames=[]
for instant in range(0, 500, 1):  # Cada 500 instantes
    path_frames = os.path.join(output_directory, f"frame_{instant:04d}.png")
    frame=bsp_64[instant, :, :]
    plt.figure()
    plt.imshow(frame)
    plt.colorbar(label='mV')  # Puedes añadir una etiqueta opcional
    plt.savefig(path_frames)
    filenames.append(path_frames)
    plt.close()

# Crea el GIF usando las imágenes guardadas
with imageio.get_writer(output_video, mode='I', duration=0.1) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

if normalizar:
    y_label_n = np.zeros_like(bspm_signal)
    for nodo in range(bspm_signal.shape[1]):  
        min_val = np.min(bspm_signal[:, nodo])
        max_val = np.max(bspm_signal[:, nodo])
        
        if max_val != min_val:  
            # Normalización entre 0 y 1
            normalized = (bspm_signal[:, nodo] - min_val) / (max_val - min_val)
            # Transformación de 0-1 a -1-1
            y_label_n[:, nodo] = 2 * normalized - 1
        else:
            y_label_n[:, nodo] = 0


    var_represent = y_label_n
    var_represent_original = y_label_n
else:
    var_represent=bspm_signal
    var_represent_original=bspm_signal





# Crear renderizadores para var_represent y var_represent_original
renderer_var = EGMRenderer_BSP(faces, vertices, min_val=-0.6, max_val=0.6, view="front")
renderer_label = EGMRenderer_BSP(faces, vertices,  min_val=-0.6, max_val=0.6, view="back")

# Configurar renderizadores como subplots
renderer_var.renderer.SetViewport(0.0, 0.0, 0.5, 1.0)  # Subplot izquierdo
renderer_label.renderer.SetViewport(0.5, 0.0, 1.0, 1.0)  # Subplot derecho

# Configurar fondo blanco
renderer_var.renderer.SetBackground(1, 1, 1)  # Gris claro
renderer_label.renderer.SetBackground(1, 1, 1)  # Gris claro

# Create a text actor for var_represent
title_var = vtk.vtkTextActor()
title_var.SetInput(f"Front")
title_varprop = title_var.GetTextProperty()
title_varprop.SetFontFamilyToArial()
title_varprop.SetFontSize(25)
#title_varprop.BoldOn()
title_varprop.SetColor(0, 0, 0)  # Black color
title_var.SetPosition(300, 500)  # Adjust position manually as needed
renderer_var.renderer.AddActor2D(title_var)

# Create a text actor for label
title_label = vtk.vtkTextActor()
title_label.SetInput("Back")
title_labelprop = title_label.GetTextProperty()
title_labelprop.SetFontFamilyToArial()
title_labelprop.SetFontSize(25)
#title_labelprop.BoldOn()
title_labelprop.SetColor(0, 0, 0)  # Black color
title_label.SetPosition(300, 500)  # Adjust position manually as needed

# Add the titles to the renderers

renderer_label.renderer.AddActor2D(title_label)

# Scalar bars remain unchanged but without titles
scalar_bar_var = vtk.vtkScalarBarActor()
scalar_bar_var.SetLookupTable(renderer_var.mapper.GetLookupTable())
scalar_bar_var.GetLabelTextProperty().SetColor(0, 0, 0)  # Set font color to black
scalar_bar_var.GetLabelTextProperty().SetItalic(False)  # Ensure text is not italicized
scalar_bar_var.GetLabelTextProperty().SetShadow(False) 
scalar_bar_var.GetLabelTextProperty().SetFontFamilyToArial()  # Set font to Arial
scalar_bar_var.GetLabelTextProperty().SetFontSize(15) 
scalar_bar_var.SetNumberOfLabels(5)

scalar_bar_label = vtk.vtkScalarBarActor()
scalar_bar_label.SetLookupTable(renderer_label.mapper.GetLookupTable())
scalar_bar_label.GetLabelTextProperty().SetColor(0, 0, 0)  # Set font color to black
scalar_bar_label.SetNumberOfLabels(5)

renderer_var.renderer.AddActor2D(scalar_bar_var)
renderer_label.renderer.AddActor2D(scalar_bar_label)


# Configurar la ventana de renderizado conjunta
render_window = vtk.vtkRenderWindow()
render_window.SetSize(1600, 600)  # Tamaño total de la ventana
render_window.AddRenderer(renderer_var.renderer)
render_window.AddRenderer(renderer_label.renderer)

# Configurar la cámara para ambos subplots
center = renderer_var.mesh.GetCenter()
for renderer in [renderer_var, renderer_label]:
    renderer.camera.SetPosition(center[0], center[1], center[2]  +200)  # Alejar la cámara
    renderer.camera.SetFocalPoint(center[0], center[1], center[2])  # Enfocar al centro
    renderer.camera.SetViewUp(1, 0, 0)  # Eje vertical hacia arriba
    renderer.camera.Elevation(270)  # Flip de 180 grados sobre el eje horizontal
    renderer.camera.Azimuth(0)  # Rotar 0 grados en azimut
    renderer.renderer.ResetCameraClippingRange()

# Configurar el filtro para capturar imágenes
window_to_image_filter = vtk.vtkWindowToImageFilter()
window_to_image_filter.SetInput(render_window)

#configure video
fps = 10  # Frames por segundo

array_frames=[]
# Iterar sobre instantes y generar frames
for instant in range(0, time, 1):  # Cada 500 instantes
    # Actualizar los datos escalares de los dos renderizadores
    vtk_scalars_var = numpy_to_vtk(var_represent[ :, instant], deep=True)
    vtk_scalars_label = numpy_to_vtk(var_represent_original[:, instant], deep=True)

    renderer_var.mesh.GetPointData().SetScalars(vtk_scalars_var)
    renderer_label.mesh.GetPointData().SetScalars(vtk_scalars_label)

    renderer_var.mesh.Modified()
    renderer_label.mesh.Modified()

    # Renderizar la ventana con subplots
    render_window.Render()

    # Guardar la imagen como PNG
    window_to_image_filter.Modified()
    window_to_image_filter.Update()

    output_file = os.path.join(output_directory, f"frame_{instant:04d}.png")
    writer = vtk.vtkPNGWriter()
    writer.SetFileName(output_file)
    writer.SetInputData(window_to_image_filter.GetOutput())
    writer.Write()

    # Obtener la imagen como un array numpy
    image_data = window_to_image_filter.GetOutput()
    dims = image_data.GetDimensions()
    vtk_array = vtk.util.numpy_support.vtk_to_numpy(image_data.GetPointData().GetScalars())
    frame = vtk_array.reshape((dims[1], dims[0], 3))[::-1]  # Invertir eje Y

    # Escribir el frame en el video
    array_frames.append(frame)

    #print(f"Frame guardado: {output_file}")
    #os.remove(output_file)

output_gif = os.path.join(output_directory, f"video_torso.gif")
imageio.mimsave(output_gif, array_frames,format='GIF', fps=10)
print(f"Video guardado en {output_gif}")


  



