from scipy.ndimage import uniform_filter1d
import vtk
from scipy.interpolate import interp1d
from vtk.util.numpy_support import numpy_to_vtk
import os


class EGMRenderer_EGM:
    def __init__(self, faces, vertices, min_val, max_val, view=None):
        self.faces = faces
        self.vertices = vertices
        self.min_val = min_val
        self.max_val = max_val
        self.view=view

        # Inicializar malla
        self.mesh = vtk.vtkPolyData()
        points = vtk.vtkPoints()
        for vertex in vertices:
            points.InsertNextPoint(vertex)
        polys = vtk.vtkCellArray()
        for face in faces:
            polys.InsertNextCell(len(face))
            for vertex_index in face:
                polys.InsertCellPoint(int(vertex_index))
        self.mesh.SetPoints(points)
        self.mesh.SetPolys(polys)

        # Configurar mapper y actor
        self.mapper = vtk.vtkPolyDataMapper()
        self.mapper.SetInputData(self.mesh)
        self.mapper.SetScalarRange(min_val, max_val)
        self.actor = vtk.vtkActor()
        self.actor.SetMapper(self.mapper)
        #self.actor.RotateZ(-90)  # Rota 30 grados hacia la derecha
        #self.actor.RotateX(10)  # Rota 30 grados hacia la derecha
        if self.view == "back":
            self.actor.RotateZ(180)  # Rota 30 grados hacia la derecha


        # Configure renderer and camera
        self.renderer = vtk.vtkRenderer()
        self.renderer.AddActor(self.actor)
        self.renderer.SetBackground(0.1, 0.2, 0.4)

        # Camera settings
        self.camera = vtk.vtkCamera()
        self.camera.SetPosition(0, 0, 50)  # Adjust position (x, y, z)
        self.camera.SetFocalPoint(0, 0, 0)  # Adjust focal point
        #self.camera.Azimuth(180)  # Rotar 90 grados hacia la derecha
        self.camera.Elevation(45)  # Rotar 90 grados hacia arriba
        #self.camera.OrthogonalizeViewUp()  # Asegurar que el view-up no sea paralelo al normal
        #self.camera.SetViewUp(normal)  # Configurar un view-up seguro


        self.renderer.SetActiveCamera(self.camera)
        self.renderer.ResetCamera()  # Asegurar que la cámara se ajuste al objeto


        # Configure render window
        self.render_window = vtk.vtkRenderWindow()
        self.render_window.AddRenderer(self.renderer)
        self.render_window.SetSize(1920, 1080)
        self.render_window.SetOffScreenRendering(1)
        # Filtro para capturar imágenes
        self.window_to_image_filter = vtk.vtkWindowToImageFilter()
        self.window_to_image_filter.SetInput(self.render_window)

    def render_frame(self, scalars, output_file):
       
        # Crear nueva malla para evitar problemas de caché
        mesh = vtk.vtkPolyData()
        mesh.SetPoints(self.mesh.GetPoints())
        mesh.SetPolys(self.mesh.GetPolys())
        
        vtk_scalars = numpy_to_vtk(scalars, deep=True)
        mesh.GetPointData().SetScalars(vtk_scalars)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(mesh)
        mapper.SetScalarRange(self.min_val, self.max_val)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        renderer = vtk.vtkRenderer()
        renderer.AddActor(actor)
        renderer.SetBackground(0.1, 0.2, 0.4)

        render_window = vtk.vtkRenderWindow()
        render_window.AddRenderer(renderer)
        render_window.SetSize(1920, 1080)

        window_to_image_filter = vtk.vtkWindowToImageFilter()
        window_to_image_filter.SetInput(render_window)

        render_window.SetOffScreenRendering(1)  # Habilita el modo offscreen

        render_window.Render()
        window_to_image_filter.Update()

        writer = vtk.vtkPNGWriter()
        writer.SetFileName(output_file)
        writer.SetInputData(window_to_image_filter.GetOutput())
        writer.Write()

    def rotate_camera_right(self, angle):
        """
        Rota la cámara hacia la derecha.
        :param angle: Ángulo en grados para rotar la cámara.
        """
        self.camera.Azimuth(angle)  # Rotar horizontalmente
        self.renderer.ResetCameraClippingRange()  # Ajustar rango de recorte
        self.render_window.Render()  # Renderizar nuevamente la ventana

    def smoothing_plot(self, window_size, data):
        return uniform_filter1d(data, size=window_size, axis=1)


    def egm_representation_vtk(
        self, reconstruction, label, faces, vertices, min_val, max_val, instant
    ):
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

        renderer_reconstruction.SetViewport(0.0, 0.5, 0.5, 1.0)
        renderer_label.SetViewport(0.5, 0.0, 1.0, 1.0)

        # Configurar ventana de renderizado
        render_window = vtk.vtkRenderWindow()
        render_window.AddRenderer(renderer_reconstruction)
        render_window.AddRenderer(renderer_label)
        render_window.SetSize(1920, 1080)

        

        # Renderizar
        render_window.Render()


class EGMRenderer_BSP:
    def __init__(self, faces, vertices, min_val, max_val, view):
        self.faces = faces
        self.vertices = vertices
        self.min_val = min_val
        self.max_val = max_val
        self.view = view

        # Inicializar malla
        self.mesh = vtk.vtkPolyData()
        points = vtk.vtkPoints()
        for vertex in vertices:
            points.InsertNextPoint(vertex)
        polys = vtk.vtkCellArray()
        for face in faces:
            polys.InsertNextCell(len(face))
            for vertex_index in face:
                polys.InsertCellPoint(int(vertex_index))
        self.mesh.SetPoints(points)
        self.mesh.SetPolys(polys)

        # Configurar mapper y actor
        self.mapper = vtk.vtkPolyDataMapper()
        self.mapper.SetInputData(self.mesh)
        self.mapper.SetScalarRange(min_val, max_val)
        self.actor = vtk.vtkActor()
        self.actor.SetMapper(self.mapper)
        self.actor.RotateZ(-90)  # Rota 30 grados hacia la derecha
        self.actor.SetScale(2)

        normals=vtk.vtkPolyDataNormals()
        normals.SetInputData(self.mesh)
        normals.ConsistencyOn()
        normals.SplittingOff()
        normals.Update()



        if self.view == "back":
            self.actor.RotateZ(180)  # Rota 30 grados hacia la derecha

        # Configure renderer and camera
        self.renderer = vtk.vtkRenderer()
        self.renderer.AddActor(self.actor)
        self.renderer.SetBackground(0.1, 0.2, 0.4)

        # Camera settings
        self.camera = vtk.vtkCamera()
        self.camera.SetPosition(0, 0, 50)  # Adjust position (x, y, z)
        self.camera.SetFocalPoint(0, 0, 0)  # Adjust focal point
        #self.camera.Azimuth(180)  # Rotar 90 grados hacia la derecha
        self.camera.Elevation(45)  # Rotar 90 grados hacia arriba
        #self.camera.OrthogonalizeViewUp()  # Asegurar que el view-up no sea paralelo al normal
        #self.camera.SetViewUp(normal)  # Configurar un view-up seguro


        self.renderer.SetActiveCamera(self.camera)
        self.renderer.ResetCamera()  # Asegurar que la cámara se ajuste al objeto


        # Configure render window
        self.render_window = vtk.vtkRenderWindow()
        self.render_window.AddRenderer(self.renderer)
        self.render_window.SetSize(1920, 1080)
        self.render_window.SetOffScreenRendering(1)
        # Filtro para capturar imágenes
        self.window_to_image_filter = vtk.vtkWindowToImageFilter()
        self.window_to_image_filter.SetInput(self.render_window)

    def render_frame(self, scalars, output_file):
       
        # Crear nueva malla para evitar problemas de caché
        mesh = vtk.vtkPolyData()
        mesh.SetPoints(self.mesh.GetPoints())
        mesh.SetPolys(self.mesh.GetPolys())
        
        vtk_scalars = numpy_to_vtk(scalars, deep=True)
        mesh.GetPointData().SetScalars(vtk_scalars)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(mesh)
        mapper.SetScalarRange(self.min_val, self.max_val)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        renderer = vtk.vtkRenderer()
        renderer.AddActor(actor)
        renderer.SetBackground(0.1, 0.2, 0.4)

        render_window = vtk.vtkRenderWindow()
        render_window.AddRenderer(renderer)
        render_window.SetSize(1920, 1080)

        window_to_image_filter = vtk.vtkWindowToImageFilter()
        window_to_image_filter.SetInput(render_window)

        render_window.SetOffScreenRendering(1)  # Habilita el modo offscreen

        render_window.Render()
        window_to_image_filter.Update()

        writer = vtk.vtkPNGWriter()
        writer.SetFileName(output_file)
        writer.SetInputData(window_to_image_filter.GetOutput())
        writer.Write()

    def rotate_camera_right(self, angle):
        """
        Rota la cámara hacia la derecha.
        :param angle: Ángulo en grados para rotar la cámara.
        """
        self.camera.Azimuth(angle)  # Rotar horizontalmente
        self.renderer.ResetCameraClippingRange()  # Ajustar rango de recorte
        self.render_window.Render()  # Renderizar nuevamente la ventana

    def smoothing_plot(self, window_size, data):
        return uniform_filter1d(data, size=window_size, axis=1)


    def egm_representation_vtk(
        self, reconstruction, label, faces, vertices, min_val, max_val, instant
    ):
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

        renderer_reconstruction.SetViewport(0.0, 0.5, 0.5, 1.0)
        renderer_label.SetViewport(0.5, 0.0, 1.0, 1.0)

        # Configurar ventana de renderizado
        render_window = vtk.vtkRenderWindow()
        render_window.AddRenderer(renderer_reconstruction)
        render_window.AddRenderer(renderer_label)
        render_window.SetSize(1920, 1080)

        

        # Renderizar
        render_window.Render()


    