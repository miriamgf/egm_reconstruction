import vtk
import numpy as np
import os
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
import numpy as np
import scipy.io as sio
import vtk
from vtk.util.numpy_support import numpy_to_vtk
from renderizer import EGMRenderer
import os

# Cargar datos
model_path = "/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/experiments/experiments_VAE/OMAMI_bs_200/reconstructions_by_model_OMAMI_bs_200.mat"
geom_path_CF = "/home/pdi/miriamgf/tesis/Autoencoders/geometries/Atria_geom/Modelos_computacionales_Carlos_Fambuena/Atria.mat"
model_path_database= "/home/pdi/miriamgf/tesis/Autoencoders/Data/Simulation_01_200428_001_010/EGMs.mat"
output_directory = "/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/renderized_heart/OMAMI_bs_200"
os.makedirs(output_directory, exist_ok=True)

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

# Cargar datos del modelo y la geometría
model = sio.loadmat(model_path)["modelSimulation_01_200428_001_010"]

# Crear los valores escalares para los nodos reducidos
num_nodes = var_represent.shape[1]  # Número de nodos con valores conocidos
num_vertices = vertices.shape[0]  # Número total de vértices en la malla

# Crear un array de valores iniciales para un subconjunto de nodos
initial_values = np.zeros(num_vertices)
indices = np.linspace(0, num_vertices - 1, num_nodes, dtype=int)
initial_values[indices] = var_represent[0, :]  # Asignar valores conocidos

# Crear un vtkPolyData para los puntos conocidos
known_points = vtk.vtkPoints()
known_scalars = vtk.vtkDoubleArray()
known_scalars.SetName("Scalars")
for i, idx in enumerate(indices):
    known_points.InsertNextPoint(vertices[idx])
    known_scalars.InsertNextValue(var_represent[0, i])

known_polydata = vtk.vtkPolyData()
known_polydata.SetPoints(known_points)
known_polydata.GetPointData().SetScalars(known_scalars)

# Crear un vtkPolyData para la malla completa
mesh_points = vtk.vtkPoints()
for vertex in vertices:
    mesh_points.InsertNextPoint(vertex)

mesh_polydata = vtk.vtkPolyData()
mesh_polydata.SetPoints(mesh_points)
mesh_polydata.SetPolys(faces)

# Usar vtkPointInterpolator para interpolar valores en toda la malla
interpolator = vtk.vtkPointInterpolator()
interpolator.SetSourceData(known_polydata)  # Datos conocidos
interpolator.SetInputData(mesh_polydata)  # Malla completa
interpolator.Update()

# Obtener los valores interpolados
interpolated_polydata = interpolator.GetOutput()
interpolated_scalars = vtk_to_numpy(interpolated_polydata.GetPointData().GetScalars())

print(f"Interpolación completada. Escalares interpolados: {interpolated_scalars.shape}")
