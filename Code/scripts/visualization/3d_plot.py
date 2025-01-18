import numpy as np
import scipy.io as sio
import vtk
from scipy.interpolate import interp1d
from vtk.util.numpy_support import numpy_to_vtk
from renderizer import EGMRenderer
import os
import cv2
import imageio


# Cargar datos
#model_path = "/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/experiments/experiments_VAE/OMAMI_weighted/reconstructions_by_model_OMAMI_weighted.mat"
model_path="/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/experiments/experiments_VAE/pruebas interpol/reconstructions_by_model_pruebas interpol.mat"
model_path="/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/experiments/experiments_VAE/OMAMI_Optuna/reconstructions_by_model_OMAMI_Optuna.mat"
geom_path_CF = "/home/pdi/miriamgf/tesis/Autoencoders/geometries/Atria_geom/Modelos_computacionales_Carlos_Fambuena/Atria.mat"
geom_path_edgar= "/home/pdi/miriamgf/tesis/Autoencoders/geometries/Atria_geom/Modelos_Edgar/Atria.mat"
model_path_database= "/home/pdi/miriamgf/tesis/Autoencoders/Data/modelLA_RSPV_CAF_150115/EGMs.mat"
output_directory = "/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/renderized_heart/OMAMI_bs_200"
os.makedirs(output_directory, exist_ok=True)

# Cargar datos del modelo y la geometría

model_name="modelSimulation_01_200316_001_  4"

try:
    model = sio.loadmat(model_path)[model_name]
    print('Loaded model ', model_name)
except:
    model = sio.loadmat(model_path)

try:
    geom = sio.loadmat(geom_path_CF)["geometries"]
except:
    geom = sio.loadmat(geom_path_CF)

# Extraer datosy
y_reconstructed = model["reconstruction"][0,0]
y_label = model["label"][0,0]
heart = geom["heart"]
faces = heart["faces"][0, 0] - 1  # Convertir a índice base 0
vertices = heart["vertices"][0, 0]

# Crear renderizadores para var_represent y var_represent_original
renderer_var = EGMRenderer(faces, vertices, min_val=-0.6, max_val=0.6)
try:
    renderer_label = EGMRenderer(faces, vertices,  min_val=-0.6, max_val=0.6)
except:
    y_label=y_label[0][0]
    renderer_label = EGMRenderer(faces, vertices,  min_val=-10, max_val=10)

# Generar datos suavizados
#label_true = sio.loadmat(model_path_database)['x'].T

if y_reconstructed.ndim>2:
    y_reconstructed = y_reconstructed.reshape(-1, y_reconstructed.shape[2]) 

#y_label=y_reconstructed 

#y_reconstructed = renderer_var.smoothing_plot(10, y_reconstructed)
#ylabel = renderer_label.smoothing_plot(10, y_label)

#probaf normalizacion
# Inicializa una matriz para las señales estandarizadas

estandarizar=False
normalizar=False

if estandarizar:
    y_label_std = np.zeros_like(y_label)
    for nodo in range(y_label.shape[1]): 
        mean_val = np.mean(y_label[:, nodo])  
        std_val = np.std(y_label[:, nodo])    
        if std_val != 0:  
            y_label_std[:, nodo] = (y_label[:, nodo] - mean_val) / std_val
        else:
            y_label_std[:, nodo] = 0  
    var_represent=y_reconstructed
    var_represent_original=y_label_std


if normalizar:
    y_label_n = np.zeros_like(y_label)
    for nodo in range(y_label.shape[1]):  
        min_val = np.min(y_label[:, nodo])
        max_val = np.max(y_label[:, nodo])
        
        if max_val != min_val:  
            y_label_n[:, nodo] = (y_label[:, nodo] - min_val) / (max_val - min_val)
        else:
            y_label_n[:, nodo] = 0  

    var_represent = y_reconstructed
    var_represent_original = y_label_n

var_represent=y_reconstructed
var_represent_original=y_label

elevation_values_range = [270, 30]

for elevation_value in elevation_values_range:
    '''
    if var_represent.shape[1]!=2048:
        var_represent=var_represent[0][0]
        var_represent_original=var_represent_original[0][0]
    '''
    # Configurar renderizadores como subplots
    renderer_var.renderer.SetViewport(0.0, 0.0, 0.5, 1.0)  # Subplot izquierdo
    renderer_label.renderer.SetViewport(0.5, 0.0, 1.0, 1.0)  # Subplot derecho

    # Configurar fondo blanco
    renderer_var.renderer.SetBackground(1, 1, 1)  # Gris claro
    renderer_label.renderer.SetBackground(1, 1, 1)  # Gris claro

    # Configurar bordes negros y deshabilitar interpolación en renderer_var y renderer_label



    # Create a text actor for var_represent
    title_var = vtk.vtkTextActor()
    title_var.SetInput(f"Reconstructed")
    title_varprop = title_var.GetTextProperty()
    title_varprop.SetFontFamilyToArial()
    title_varprop.SetFontSize(25)
    #title_varprop.BoldOn()
    title_varprop.SetColor(0, 0, 0)  # Black color
    title_var.SetPosition(300, 500)  # Adjust position manually as needed
    renderer_var.renderer.AddActor2D(title_var)

    # Create a text actor for label
    title_label = vtk.vtkTextActor()
    title_label.SetInput("Real")
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
        renderer.camera.SetPosition(center[0], center[1], center[2] + 35)  # Alejar la cámara
        renderer.camera.SetFocalPoint(center[0], center[1], center[2])  # Enfocar al centro
        renderer.camera.SetViewUp(1, 0, 0)  # Eje vertical hacia arriba
        renderer.camera.Elevation(elevation_value)  # Flip de 180 grados sobre el eje horizontal
        renderer.camera.Azimuth(0)  # Rotar 0 grados en azimut
        renderer.renderer.ResetCameraClippingRange()

    # Configurar el filtro para capturar imágenes
    window_to_image_filter = vtk.vtkWindowToImageFilter()
    window_to_image_filter.SetInput(render_window)

    #configure video
    if elevation_value == 270:
        view = "front"
    elif elevation_value == 30:
        view = "back"
    output_video = os.path.join(output_directory, f"video_{view}.gif")
    sample_frame = os.path.join(output_directory, "sample_frame.png")
    renderer.render_frame(var_represent[0, :], sample_frame)
    framei = cv2.imread(sample_frame)
    height, width, layers = framei.shape
    fps = 10  # Frames por segundo
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec para AVI
    video = cv2.VideoWriter(output_video, fourcc, fps, (1600, 600))
    array_frames=[]
    # Iterar sobre instantes y generar frames
    for instant in range(0, 500, 1):  # Cada 500 instantes
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

        # Obtener la imagen como un array numpy
        image_data = window_to_image_filter.GetOutput()
        dims = image_data.GetDimensions()
        vtk_array = vtk.util.numpy_support.vtk_to_numpy(image_data.GetPointData().GetScalars())
        frame = vtk_array.reshape((dims[1], dims[0], 3))[::-1]  # Invertir eje Y

        # Escribir el frame en el video
        #video.write(frame)
        array_frames.append(frame)

        print(f"Frame guardado: {output_file}")
        #os.remove(output_file)

    # Liberar el objeto VideoWriter
    imageio.mimsave(output_video, array_frames,format='GIF', fps=10)


    #video.release()

    print(f"Video guardado en {output_video}")