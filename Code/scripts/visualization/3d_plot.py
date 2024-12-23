import numpy as np
import scipy.io as sio
import vtk
from scipy.interpolate import interp1d
from vtk.util.numpy_support import numpy_to_vtk
from renderizer import EGMRenderer
import os
import cv2

# Cargar datos
model_path = "/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/experiments/experiments_VAE/OMAMI_bs_200/reconstructions_by_model_OMAMI_bs_200.mat"
geom_path_CF = "/home/pdi/miriamgf/tesis/Autoencoders/geometries/Atria_geom/Modelos_computacionales_Carlos_Fambuena/Atria.mat"
model_path_database= "/home/pdi/miriamgf/tesis/Autoencoders/Data/Simulation_01_200428_001_010/EGMs.mat"
output_directory = "/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/renderized_heart/OMAMI_bs_200"
os.makedirs(output_directory, exist_ok=True)

# Cargar datos del modelo y la geometría
model = sio.loadmat(model_path)["modelSimulation_01_200428_001_010"]

try:
    geom = sio.loadmat(geom_path_CF)["geometries"]
except:
    geom = sio.loadmat(geom_path_CF)

# Extraer datos
y_reconstructed = model["reconstruction"][0, 0]
y_label = model["label"][0, 0]
heart = geom["heart"]
faces = heart["faces"][0, 0] - 1  # Convertir a índice base 0
vertices = heart["vertices"][0, 0]

# Crear renderizadores para var_represent y var_represent_original
renderer_var = EGMRenderer(faces, vertices, min_val=-1, max_val=1)
renderer_label = EGMRenderer(faces, vertices, min_val=0, max_val=np.max(y_label))

# Generar datos suavizados
label_true = sio.loadmat(model_path_database)['x'].T
var_represent = renderer_var.smoothing_plot(10, y_reconstructed)
var_represent_original = renderer_label.smoothing_plot(10, label_true)
var_represent=y_reconstructed
var_represent_original=label_true


# Configurar renderizadores como subplots
renderer_var.renderer.SetViewport(0.0, 0.0, 0.5, 1.0)  # Subplot izquierdo
renderer_label.renderer.SetViewport(0.5, 0.0, 1.0, 1.0)  # Subplot derecho

# Configurar fondo blanco
renderer_var.renderer.SetBackground(1.0, 1.0, 1.0)  # Blanco
renderer_label.renderer.SetBackground(1.0, 1.0, 1.0)  # Blanco

# Crear barras de color (leyendas)
scalar_bar_var = vtk.vtkScalarBarActor()
scalar_bar_var.SetLookupTable(renderer_var.mapper.GetLookupTable())
scalar_bar_var.SetTitle("Var Represent")
scalar_bar_var.SetNumberOfLabels(5)

scalar_bar_label = vtk.vtkScalarBarActor()
scalar_bar_label.SetLookupTable(renderer_label.mapper.GetLookupTable())
scalar_bar_label.SetTitle("Label")
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
    renderer.camera.SetPosition(center[0], center[1], center[2] + 50)  # Alejar la cámara
    renderer.camera.SetFocalPoint(center[0], center[1], center[2])  # Enfocar al centro
    renderer.camera.SetViewUp(1, 0, 0)  # Eje vertical hacia arriba
    renderer.camera.Elevation(270)  # Flip de 180 grados sobre el eje horizontal
    renderer.camera.Azimuth(0)  # Rotar 0 grados en azimut
    renderer.renderer.ResetCameraClippingRange()

# Configurar el filtro para capturar imágenes
window_to_image_filter = vtk.vtkWindowToImageFilter()
window_to_image_filter.SetInput(render_window)

# Iterar sobre instantes y generar frames
for instant in range(0, var_represent.shape[0], 500):  # Cada 500 instantes
    # Actualizar los datos escalares de los dos renderizadores
    vtk_scalars_var = numpy_to_vtk(var_represent[instant, :], deep=True)
    vtk_scalars_label = numpy_to_vtk(var_represent_original[instant, :], deep=True)

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

    print(f"Frame guardado: {output_file}")
