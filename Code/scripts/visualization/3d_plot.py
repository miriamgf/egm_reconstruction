import numpy as np
import scipy.io as sio
import vtk
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter1d
from vtk.util.numpy_support import numpy_to_vtk

# Cargar datos
model_path = "/Users/miriamgutierrez/Library/CloudStorage/OneDrive-UniversidadReyJuanCarlos/Escritorio/URJC/Doctorado/Congresos/CINC24/Paper/Modelo/20240827-112359_EXP_0/recontruction_by_model.mat"
geom_path = "data/geometries.mat"

model = sio.loadmat(model_path)["modelSimulation_01_200428_001_010"]
geom = sio.loadmat(geom_path)["geometries"]

# Extraer datos
y_reconstructed = model["reconstruction"][0, 0]
y_label = model["label"][0, 0]
heart = geom["heart"][0, 0]
faces = heart["faces"][0, 0] - 1  # Convertir a índice base 0
vertices = heart["vertices"][0, 0]

# Interpolación si es necesario
if y_reconstructed.shape[1] != 2048:
    x_original = np.linspace(0, y_reconstructed.shape[1] - 1, y_reconstructed.shape[1])
    x_nuevo = np.linspace(0, y_reconstructed.shape[1] - 1, 2048)

    interp_func_reconstructed = interp1d(
        x_original, y_reconstructed, axis=1, kind="cubic"
    )
    interp_func_label = interp1d(x_original, y_label, axis=1, kind="cubic")

    y_reconstructed = interp_func_reconstructed(x_nuevo)
    y_label = interp_func_label(x_nuevo)


# Suavizado
def smoothing_plot(window_size, data):
    return uniform_filter1d(data, size=window_size, axis=1)


var_represent = smoothing_plot(10, y_reconstructed)
var_represent_original = smoothing_plot(10, y_label)

# Configuración de tiempo
Fs = 200
L = var_represent.shape[1]
time = np.arange(0, L / Fs, 1 / Fs)


# Función para renderizar EGM con VTK
def egm_representation_vtk(
    reconstruction, label, faces, vertices, min_val, max_val, instant
):
    # Convertir vértices y caras a formato VTK
    points = vtk.vtkPoints()
    for vertex in vertices:
        points.InsertNextPoint(vertex)

    polys = vtk.vtkCellArray()
    for face in faces:
        polys.InsertNextCell(len(face))
        for vertex_index in face:
            polys.InsertCellPoint(int(vertex_index))

    # Crear objeto de malla
    mesh = vtk.vtkPolyData()
    mesh.SetPoints(points)
    mesh.SetPolys(polys)

    # Crear y asignar datos de reconstrucción como escalares
    reconstruction_vtk = numpy_to_vtk(reconstruction[:, instant], deep=True)
    mesh.GetPointData().SetScalars(reconstruction_vtk)

    # Configurar mapper y actor para reconstrucción
    mapper_reconstruction = vtk.vtkPolyDataMapper()
    mapper_reconstruction.SetInputData(mesh)
    mapper_reconstruction.SetScalarRange(min_val, max_val)

    actor_reconstruction = vtk.vtkActor()
    actor_reconstruction.SetMapper(mapper_reconstruction)

    # Crear y asignar datos de etiquetas como escalares
    label_vtk = numpy_to_vtk(label[:, instant], deep=True)
    mesh.GetPointData().SetScalars(label_vtk)

    mapper_label = vtk.vtkPolyDataMapper()
    mapper_label.SetInputData(mesh)
    mapper_label.SetScalarRange(min_val, max_val)

    actor_label = vtk.vtkActor()
    actor_label.SetMapper(mapper_label)

    # Configurar renderers
    renderer_reconstruction = vtk.vtkRenderer()
    renderer_label = vtk.vtkRenderer()

    renderer_reconstruction.AddActor(actor_reconstruction)
    renderer_label.AddActor(actor_label)

    renderer_reconstruction.SetViewport(0.0, 0.0, 0.5, 1.0)
    renderer_label.SetViewport(0.5, 0.0, 1.0, 1.0)

    # Configurar ventana de renderizado
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer_reconstruction)
    render_window.AddRenderer(renderer_label)
    render_window.SetSize(1920, 1080)

    # Interactor
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)

    # Renderizar
    render_window.Render()
    interactor.Start()


# Generar visualización para un instante inicial
instant = 0
min_val, max_val = -1, 1
egm_representation_vtk(
    var_represent, var_represent_original, faces, vertices, min_val, max_val, instant
)

# Generar video con VTK
video_filename = "video_egm_representation_vtk.avi"
fps = 10
video_writer = cv2.VideoWriter(
    video_filename, cv2.VideoWriter_fourcc(*"XVID"), fps, (1920, 1080)
)

for instant in range(len(time)):
    egm_representation_vtk(
        var_represent,
        var_represent_original,
        faces,
        vertices,
        min_val,
        max_val,
        instant,
    )
    # Capturar la pantalla renderizada
    window_to_image_filter = vtk.vtkWindowToImageFilter()
    window_to_image_filter.SetInput(render_window)
    window_to_image_filter.Update()

    vtk_image = window_to_image_filter.GetOutput()
    width, height, _ = render_window.GetSize()
    vtk_array = vtk.util.numpy_support.vtk_to_numpy(
        vtk_image.GetPointData().GetScalars()
    ).reshape(height, width, -1)
    vtk_array = (vtk_array[:, :, :3] * 255).astype(np.uint8)
    video_writer.write(cv2.cvtColor(vtk_array, cv2.COLOR_RGB2BGR))

video_writer.release()
