import sys
sys.path.append("../Code")
import numpy as np
import scipy.io as sio
import vtk
from scipy.interpolate import interp1d
from vtk.util.numpy_support import numpy_to_vtk
import os
import cv2
import imageio
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import h5py
import matplotlib.colors as mcolors




from scripts.visualization.utils.renderizer import EGMRenderer_EGM


class EGM_3D_PLOTTER:

    def __init__(self, model_name, model_path, geom_path_CF, output_directory, labels_mode, tikhonov=False, time=500, all_egms=False):
        self.model_name=model_name
        self.model_path=model_path
        self.geom_path_CF=geom_path_CF
        self.output_directory=output_directory
        self.duration = time
        self.labels_mode = labels_mode
        self.tikhonov=tikhonov
        self.all_egms=all_egms

    
    def load_geometry_and_egm(self):

        if self.labels_mode:
            self.model_path="/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/experiments/experiments_VAE/pruebas interpol/egm_names_all.mat"
            #Load
            try:
                model = sio.loadmat(self.model_path)[self.model_name]
            except:
                try:
                    with h5py.File(self.model_path, 'r') as f:
                        model = {key: f[key][()] for key in f.keys()}
                except OSError as e:
                    print(f"Error loading model file: {e}")
                    model = None

            try:
                geom = sio.loadmat(self.geom_path_CF)["geometries"]
            except:
                geom = sio.loadmat(self.geom_path_CF)
            
            # Generate df from dictionary

            egm_tensor = model["egm"]#[0, 0]
            AF_models = model["AF_models"]#[0, 0]
            names=model["all_model_names"]#[0, 0]
            try:
                heart = geom["heart"]
            except:
                heart = geom
            faces = heart["faces"][0, 0] - 1  # Convertir a índice base 0
            vertices = heart["vertices"][0, 0]

            if len(vertices.shape) == 0:
                faces = heart["faces"] - 1  # Convertir a índice base 0
                vertices = heart["vertices"]
            
            patient_names_list = []
            patient_egm_list = []
            for patient in np.unique(AF_models):
                print('patient:', patient)
                patient_indices = np.where(AF_models == patient)[0]
                patient_name = names[patient].decode('utf-8')
                patient_egm = egm_tensor[patient_indices]
                a=int(len(patient_egm)/10)
                print(a)
                patient_egm = patient_egm[0: a-1, :] #repeated N times (N confussion matrices)
                patient_names_list.append(patient_name)
                patient_egm_list.append(patient_egm)

                print(len(patient_egm), patient_name)
            #Create a DataFrame from the synchronized data
            df = pd.DataFrame({
                "name": patient_names_list,
                "egm": patient_egm_list
            })

            df.head()

            #Filter df with desired AF model
        
            df_filtered = df[df['name'] == self.model_name[0]]
     
            # Get the row with the desired name
            row = df_filtered.iloc[0]

            # Extract values from each column
            name = row['name']
            y_reconstructed = row['egm']

            y_label=y_reconstructed
            # Crear renderizadores para var_represent y var_represent_original
            renderer_var = EGMRenderer_EGM(faces, vertices, min_val=-1, max_val=1)
            try:
                renderer_label = EGMRenderer_EGM(faces, vertices, min_val=0, max_val=1)
            except:
                y_label=y_label[0][0]
                renderer_label = EGMRenderer_EGM(faces, vertices, min_val=0, max_val=1)

            # Generar datos suavizados
            #label_true = sio.loadmat(model_path_database)['x'].T
            if y_reconstructed.ndim>2:
                y_reconstructed = y_reconstructed.reshape(-1, y_reconstructed.shape[2]) 
            
        else:
            

            try:
                geom = sio.loadmat(self.geom_path_CF)["geometries"]
            except:
                geom = sio.loadmat(self.geom_path_CF)

            
            
            y_reconstructed =None
            y_label = None
            heart = geom["heart"]
            faces = heart["faces"][0, 0] - 1  # Convertir a índice base 0
            vertices = heart["vertices"][0, 0]

        return y_reconstructed, y_label, faces, vertices
    
    def plot_only_label(self, y_label, faces, vertices, normalizar=True, frames=1):
        '''
        This function plots only labels with front (left) and back (right) in a single figure.
        '''
        # Crear dos renderizadores para front y back
        renderer_front = EGMRenderer_EGM(faces, vertices, min_val=-1, max_val=1)
        renderer_back = EGMRenderer_EGM(faces, vertices, min_val=-1, max_val=1)

        # Normalización opcional
        if normalizar:
            y_label_n = np.zeros_like(y_label)
            for nodo in range(y_label.shape[1]):  
                min_val = np.min(y_label[:, nodo])
                max_val = np.max(y_label[:, nodo])
                if max_val != min_val:  
                    y_label_n[:, nodo] = (y_label[:, nodo] - min_val) / (max_val - min_val)
                else:
                    y_label_n[:, nodo] = 0  
            var_represent_original = y_label_n
        else:
            var_represent_original = y_label

        # Configurar Viewport: Front a la izquierda, Back a la derecha
        renderer_front.renderer.SetViewport(0.0, 0.0, 0.5, 1.0)  # Izquierda
        renderer_back.renderer.SetViewport(0.5, 0.0, 1.0, 1.0)  # Derecha

        # Crear colormap
        custom_lut, cmap = self.create_custom_colormap_voltage()
        renderer_front.mapper.SetLookupTable(custom_lut)
        renderer_back.mapper.SetLookupTable(custom_lut)

        # Configurar fondo blanco
        renderer_front.renderer.SetBackground(1, 1, 1)
        renderer_back.renderer.SetBackground(1, 1, 1)

        # Crear títulos para cada vista
        title_front = vtk.vtkTextActor()
        title_front.SetInput("Front (270°)")
        title_front.GetTextProperty().SetFontSize(25)
        title_front.GetTextProperty().SetColor(0, 0, 0)
        title_front.SetPosition(200, 500)
        renderer_front.renderer.AddActor2D(title_front)

        title_back = vtk.vtkTextActor()
        title_back.SetInput("Back (30°)")
        title_back.GetTextProperty().SetFontSize(25)
        title_back.GetTextProperty().SetColor(0, 0, 0)
        title_back.SetPosition(200, 500)
        renderer_back.renderer.AddActor2D(title_back)

        # Crear barra de color
        scalar_bar = vtk.vtkScalarBarActor()
        scalar_bar.SetLookupTable(renderer_front.mapper.GetLookupTable())
        scalar_bar.GetLabelTextProperty().SetColor(0, 0, 0)
        scalar_bar.SetNumberOfLabels(5)
        scalar_bar.SetTitle("mV")
        scalar_bar.SetPosition(0.9, 0.2)  # Alineada a la derecha
        scalar_bar.SetWidth(0.05)
        scalar_bar.SetHeight(0.6)

        renderer_back.renderer.AddActor2D(scalar_bar)

        # Crear ventana de renderizado combinada
        render_window = vtk.vtkRenderWindow()
        render_window.SetOffScreenRendering(True)
        render_window.SetSize(1600, 600)  # Doble de ancho para ambas vistas
        render_window.AddRenderer(renderer_front.renderer)
        render_window.AddRenderer(renderer_back.renderer)

        # Configurar las cámaras
        center = renderer_front.mesh.GetCenter()
        for renderer, elevation in zip([renderer_front, renderer_back], [270, 30]):
            renderer.camera.SetPosition(center[0], center[1], center[2] + 35)
            renderer.camera.SetFocalPoint(center[0], center[1], center[2])
            renderer.camera.SetViewUp(1, 0, 0)
            renderer.camera.Elevation(elevation)
            renderer.renderer.ResetCameraClippingRange()

        # Configurar filtro de imagen
        window_to_image_filter = vtk.vtkWindowToImageFilter()
        window_to_image_filter.SetInput(render_window)

        # Configurar salida de video
        output_video = os.path.join(self.output_directory, f"{self.model_name}.gif")
        fps = 10  # Frames por segundo

        array_frames = []
        for instant in range(0, self.duration, frames):
            vtk_scalars_label = numpy_to_vtk(var_represent_original[instant, :], deep=True)
            renderer_front.mesh.GetPointData().SetScalars(vtk_scalars_label)
            renderer_back.mesh.GetPointData().SetScalars(vtk_scalars_label)
            
            renderer_front.mesh.Modified()
            renderer_back.mesh.Modified()

            render_window.Render()
            window_to_image_filter.Modified()
            window_to_image_filter.Update()

            image_data = window_to_image_filter.GetOutput()
            dims = image_data.GetDimensions()
            vtk_array = vtk.util.numpy_support.vtk_to_numpy(image_data.GetPointData().GetScalars())
            frame = vtk_array.reshape((dims[1], dims[0], 3))[::-1]  # Invertir eje Y
            array_frames.append(frame)

        # Guardar GIF
        imageio.mimsave(output_video, array_frames, format='GIF', fps=5)
        print(f"Video guardado en {output_video}")



    def plot_3d_mesh_prediction(self, y_reconstructed, y_label, faces, vertices, normalizar=True):
        # Crear renderizadores para var_represent y var_represent_original
        renderer_var = EGMRenderer_EGM(faces, vertices, min_val=-1, max_val=1)
        renderer_label = EGMRenderer_EGM(faces, vertices,  min_val=-1, max_val=1)
       

        if y_reconstructed.ndim>2:
            y_reconstructed = y_reconstructed.reshape(-1, y_reconstructed.shape[2]) 

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
        var_represent_original=y_label_n

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

            custom_lut,cmap = self.create_custom_colormap_voltage()

            # Asignar la paleta al mapper de ambos renderizadores
            renderer_var.mapper.SetLookupTable(custom_lut)
            renderer_label.mapper.SetLookupTable(custom_lut)

            # Configurar fondo blanco
            renderer_var.renderer.SetBackground(1, 1, 1)  # Gris claro
            renderer_label.renderer.SetBackground(1, 1, 1)  # Gris claro

            # Create a text actor for var_represent
            title_var = vtk.vtkTextActor()
            title_var.SetInput(f"Reconstructed ({self.model_name})")
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

            renderer_label.renderer.AddActor2D(title_label)
            # Barra de color para var_represent
            scalar_bar_var = vtk.vtkScalarBarActor()
            scalar_bar_var.SetLookupTable(renderer_var.mapper.GetLookupTable())
            scalar_bar_var.GetLabelTextProperty().SetColor(0, 0, 0)  # Color de la fuente a negro
            scalar_bar_var.GetLabelTextProperty().SetItalic(False)  # Asegurarse de que no esté en cursiva
            scalar_bar_var.GetLabelTextProperty().SetShadow(False)
            scalar_bar_var.GetLabelTextProperty().SetFontFamilyToArial()  # Fuente Arial
            scalar_bar_var.GetLabelTextProperty().SetFontSize(10)  # Tamaño de fuente ajustado para las etiquetas
            scalar_bar_var.SetNumberOfLabels(5)
            scalar_bar_var.SetTextPositionToPrecedeScalarBar()
            scalar_bar_var.SetTitle("mV")  # Título de la barra

            # Cambiar tamaño del título de la barra lateral
            title_text_property = scalar_bar_var.GetTitleTextProperty()
            title_text_property.SetFontSize(10)  # Tamaño de fuente del título ajustado aquí
            title_text_property.SetColor(0, 0, 0)  # Color negro para el título
            title_text_property.SetFontFamilyToArial()  # Fuente Arial
            title_text_property.SetBold(True)  # Opcional: poner en negrita



            # Barra de color para var_represent_original
            scalar_bar_label = vtk.vtkScalarBarActor()
            scalar_bar_label.SetLookupTable(renderer_label.mapper.GetLookupTable())
            scalar_bar_label.GetLabelTextProperty().SetColor(0, 0, 0)  # Color de la fuente a negro
            scalar_bar_label.SetNumberOfLabels(5)
            scalar_bar_label.SetTitle("mV")  # Título de la barra
            scalar_bar_var.SetPosition(0.8, 0.2)  # Ajustar la posición de la barra de color
            scalar_bar_var.SetWidth(0.05)  # Ajustar el ancho de la barra
            scalar_bar_var.SetHeight(0.6)  # Ajustar la altura de la barra

            scalar_bar_label.SetPosition(0.8, 0.2)  # Ajustar la posición de la barra de color
            scalar_bar_label.SetWidth(0.05)  # Ajustar el ancho de la barra
            scalar_bar_label.SetHeight(0.6)  # Ajustar la altura de la barra
            # Cambiar tamaño del título de la barra lateral
            title_text_property_label = scalar_bar_label.GetTitleTextProperty()
            title_text_property_label.SetFontSize(5)  # Tamaño de fuente del título ajustado aquí
            title_text_property_label.SetColor(0, 0, 0)  # Color negro para el título
            title_text_property_label.SetFontFamilyToArial()  # Fuente Arial
            title_text_property_label.SetBold(True)  # Opcional: poner en negrita
            # Para var_represent
            scalar_bar_var.SetTitle("mV")
            title_text_property = scalar_bar_var.GetTitleTextProperty()
            title_text_property.SetFontSize(5)  # Tamaño de fuente del título
            title_text_property.SetColor(0, 0, 0)  # Color negro
            title_text_property.SetFontFamilyToArial()  # Fuente Arial
            title_text_property.SetBold(False)  # No en negrita

            # Para var_represent_original
            scalar_bar_label.SetTitle("mV")
            title_text_property_label = scalar_bar_label.GetTitleTextProperty()
            title_text_property_label.SetFontSize(5)  # Tamaño de fuente del título
            title_text_property_label.SetColor(0, 0, 0)  # Color negro
            title_text_property_label.SetFontFamilyToArial()  # Fuente Arial
            title_text_property_label.SetBold(False)  # No en negrita

            renderer_var.renderer.AddActor2D(scalar_bar_var)
            renderer_label.renderer.AddActor2D(scalar_bar_label)


            custom_lut.SetNumberOfTableValues(256)
            custom_lut.Build()

            for i in range(255):
                r, g, b, _ = cmap(i / 255.0)  # Obtener color del colormap
                custom_lut.SetTableValue(i, r, g, b, 1.0)  # Asignar color en la tabla de VTK

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetLookupTable(custom_lut)
            #mapper.SetScalarRange(-0.6, 0.6)  # Ajustar al rango de valores de tus datos

            # Configurar la ventana de renderizado conjunta
            render_window = vtk.vtkRenderWindow()
            render_window.SetOffScreenRendering(True)

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
            if self.tikhonov:
                output_video = os.path.join(self.output_directory, f"egm_rec_tikhonov_{view}2.gif")
            else:
                output_video = os.path.join(self.output_directory, f"egm_rec_DL_{view}2.gif")
            
            if self.all_egms:
                output_video = os.path.join(self.output_directory, f"{self.model_name}.gif")

            fps = 10  # Frames por segundo

            array_frames=[]
            # Iterar sobre instantes y generar frames
            for instant in range(0, self.duration):  # Cada 500 instantes
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

                output_file = os.path.join(self.output_directory, f"frame_{instant:04d}.png")
                writer = vtk.vtkPNGWriter()
                writer.SetFileName(output_file)
                writer.SetInputData(window_to_image_filter.GetOutput())
                writer.Write()

                # Obtener la imagen como un array numpy
                image_data = window_to_image_filter.GetOutput()
                dims = image_data.GetDimensions()
                vtk_array = vtk.util.numpy_support.vtk_to_numpy(image_data.GetPointData().GetScalars())
                frame = vtk_array.reshape((dims[1], dims[0], 3))[::-1]  # Invertir eje Y
                array_frames.append(frame)

                #print(f"Frame guardado: {output_file}")
                os.remove(output_file)

            # Liberar el objeto VideoWriter
            imageio.mimsave(output_video, array_frames,format='GIF', fps=10)
            print(f"Video guardado en {output_video}")



    


    
    def plot_3d_mesh_label(self, y_reconstructed, y_label, faces, vertices, normalizar=True ):
        # Crear renderizadores para var_represent y var_represent_original
        renderer_var = EGMRenderer_EGM(faces, vertices, min_val=-0.6, max_val=0.6)
        try:
            renderer_label = EGMRenderer_EGM(faces, vertices,  min_val=-0.6, max_val=0.6)
        except:
            y_label=y_label[0][0]
            renderer_label = EGMRenderer_EGM(faces, vertices,  min_val=-10, max_val=10)

        if y_reconstructed.ndim>2:
            y_reconstructed = y_reconstructed.reshape(-1, y_reconstructed.shape[2]) 

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

        var_represent=y_label
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
            scalar_bar_var.GetLabelTextProperty().SetFontSize(5) 
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

            output_video = os.path.join(self.output_directory, f"labels_{view}.gif")

            array_frames=[]
            # Iterar sobre instantes y generar frames
            for instant in range(0, self.duration, 1):  # Cada 500 instantes
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

                output_file = os.path.join(self.output_directory, f"frame_{instant:04d}.png")
                writer = vtk.vtkPNGWriter()
                writer.SetFileName(output_file)
                writer.SetInputData(window_to_image_filter.GetOutput())
                writer.Write()

                # Obtener la imagen como un array numpy
                image_data = window_to_image_filter.GetOutput()
                dims = image_data.GetDimensions()
                vtk_array = vtk.util.numpy_support.vtk_to_numpy(image_data.GetPointData().GetScalars())
                frame = vtk_array.reshape((dims[1], dims[0], 3))[::-1]  # Invertir eje Y
                array_frames.append(frame)
                #print(f"Frame guardado: {output_file}")
                os.remove(output_file)

            # Liberar el objeto VideoWriter
            imageio.mimsave(output_video, array_frames,format='GIF', fps=10)

            print(f"Video guardado en {output_video}")
    


    def plot_3d_regions(self, y_reconstructed, y_label, faces, vertices, normalizar=True):

        print("Rango de valores escalares reconstruidos:", np.min(y_reconstructed), np.max(y_reconstructed))

    # Crear renderizadores para var_represent y var_represent_original
        renderer_var = EGMRenderer_EGM(faces, vertices, min_val=1, max_val=7)
        try:
            renderer_label = EGMRenderer_EGM(faces, vertices, min_val=1, max_val=7)
        except:
            y_label = y_label[0][0]
            renderer_label = EGMRenderer_EGM(faces, vertices, min_val=1, max_val=7)

        # Verificar el rango de la tabla de colores
        print("Rango de la tabla de colores:", renderer_var.mapper.GetLookupTable().GetRange())

        # Asegurar que y_reconstructed sea unidimensional
        if y_reconstructed.ndim > 2:
            y_reconstructed = y_reconstructed.reshape(-1, y_reconstructed.shape[2])

        # Asegurarse de que los valores sean enteros y estén aplanados
        var_represent = y_reconstructed.flatten().astype(np.int32)
        var_represent_original = y_label.flatten().astype(np.int32)

        print("Rango de valores escalares reconstruidos:", np.min(var_represent), np.max(var_represent))

        elevation_values_range = [270, 30]

        for elevation_value in elevation_values_range:
            # Configurar los subplots
            renderer_var.renderer.SetViewport(0.0, 0.0, 0.5, 1.0)  # Subplot izquierdo
            renderer_label.renderer.SetViewport(0.5, 0.0, 1.0, 1.0)  # Subplot derecho

            # Configurar fondo blanco
            renderer_var.renderer.SetBackground(1, 1, 1)
            renderer_label.renderer.SetBackground(1, 1, 1)

            # Configurar texto
            title_var = vtk.vtkTextActor()
            title_var.SetInput(f"Front")
            title_varprop = title_var.GetTextProperty()
            title_varprop.SetFontFamilyToArial()
            title_varprop.SetFontSize(25)
            title_varprop.SetColor(0, 0, 0)
            title_var.SetPosition(200, 500)
            renderer_var.renderer.AddActor2D(title_var)

            title_label = vtk.vtkTextActor()
            title_label.SetInput("Real")
            title_labelprop = title_label.GetTextProperty()
            title_labelprop.SetFontFamilyToArial()
            title_labelprop.SetFontSize(25)
            title_labelprop.SetColor(0, 0, 0)
            title_label.SetPosition(300, 500)
            renderer_label.renderer.AddActor2D(title_label)

            

            # Configurar la tabla de colores
            lookup_table = vtk.vtkLookupTable()
            lookup_table.SetNumberOfTableValues(7)
            lookup_table.SetRange(1, 7)
            colors = [
               
                (1.0, 0.0, 0.0),  # Rojo
                (0.0, 1.0, 0.0),  # Verde
                (0.0, 0.0, 1.0),  # Azul
                (1.0, 1.0, 0.0),  # Amarillo
                (0.0, 1.0, 1.0),  # Cyan
                (1.0, 0.0, 1.0),  # Magenta
                (0.5, 0.5, 0.5),  # Gris
            ]

            for i, color in enumerate(colors):
                lookup_table.SetTableValue(i, *color, 1.0)
            lookup_table.Build()
            # Configurar sombreado plano para eliminar la interpolación
            renderer_var.actor.GetProperty().SetInterpolationToFlat()


            # Configurar el mapper y actor para renderizador
            renderer_var.mapper.SetLookupTable(lookup_table)
            renderer_var.mapper.SetScalarVisibility(True)
            renderer_var.mapper.SetScalarModeToUsePointData()
            renderer_var.mapper.SetColorModeToMapScalars()
            renderer_var.mapper.SetScalarRange(1, 7)

            renderer_label.mapper.SetLookupTable(lookup_table)
            renderer_label.mapper.SetScalarVisibility(True)
            renderer_label.mapper.SetScalarModeToUsePointData()
            renderer_label.mapper.SetColorModeToMapScalars()
            renderer_label.mapper.SetScalarRange(1, 7)

            # Configurar barras de colores
            scalar_bar_var = vtk.vtkScalarBarActor()
            scalar_bar_var.SetLookupTable(renderer_var.mapper.GetLookupTable())
            scalar_bar_var.GetLabelTextProperty().SetColor(0, 0, 0)
            scalar_bar_var.SetNumberOfLabels(5)

            scalar_bar_label = vtk.vtkScalarBarActor()
            scalar_bar_label.SetLookupTable(renderer_label.mapper.GetLookupTable())
            scalar_bar_label.GetLabelTextProperty().SetColor(0, 0, 0)
            scalar_bar_label.SetNumberOfLabels(5)

            renderer_var.renderer.AddActor2D(scalar_bar_var)
            renderer_label.renderer.AddActor2D(scalar_bar_label)

            # Configurar la ventana de renderizado
            render_window = vtk.vtkRenderWindow()
            render_window.SetOffScreenRendering(True)
            render_window.SetSize(1600, 600)
            render_window.AddRenderer(renderer_var.renderer)
            render_window.AddRenderer(renderer_label.renderer)

            # Configurar la cámara
            center = renderer_var.mesh.GetCenter()
            for renderer in [renderer_var, renderer_label]:
                renderer.camera.SetPosition(center[0], center[1], center[2] + 35)
                renderer.camera.SetFocalPoint(center[0], center[1], center[2])
                renderer.camera.SetViewUp(1, 0, 0)
                renderer.camera.Elevation(elevation_value)
                renderer.renderer.ResetCameraClippingRange()

            # Captura y guarda de frames
            array_frames = []
            for instant in range(1):
                vtk_scalars_var = numpy_to_vtk(var_represent, deep=True)
                vtk_scalars_label = numpy_to_vtk(var_represent_original, deep=True)

                renderer_var.mesh.GetPointData().SetScalars(vtk_scalars_var)
                renderer_label.mesh.GetPointData().SetScalars(vtk_scalars_label)

                renderer_var.mesh.Modified()
                renderer_label.mesh.Modified()

                render_window.Render()
                window_to_image_filter = vtk.vtkWindowToImageFilter()
                window_to_image_filter.SetInput(render_window)
                window_to_image_filter.Modified()
                window_to_image_filter.Update()

                image_data = window_to_image_filter.GetOutput()
                dims = image_data.GetDimensions()
                vtk_array = vtk.util.numpy_support.vtk_to_numpy(image_data.GetPointData().GetScalars())
                frame = vtk_array.reshape((dims[1], dims[0], 3))[::-1]
                array_frames.append(frame)

            # Guardar video
            output_video = os.path.join(self.output_directory, f"REGIONS_{elevation_value}.gif")
            imageio.mimsave(output_video, array_frames, format="GIF", fps=10)
            print(f"Video guardado en {output_video}")

    def create_custom_colormap_voltage(self):

        # Definir los colores con rosa añadido
        colors = [
            (0.0, "darkblue"),   
            (0.25, "cyan"), 
            (0.50, "green"),  
            (0.75, "yellow"),   
            (0.90, "red"),   
            (1.0, "darkred"),  
        ]
        cmap = mcolors.LinearSegmentedColormap.from_list("custom_colormap", colors, N=700)
        lut = vtk.vtkLookupTable()
        return lut, cmap
    



    def __call__(self):
        y_reconstructed, y_label, faces, vertices=self.load_geometry_and_egm()      

        if self.labels_mode:
            self.plot_3d_mesh_label(y_reconstructed, y_label, faces, vertices)
        else:
            self.plot_3d_mesh_prediction(y_reconstructed, y_label, faces, vertices)


