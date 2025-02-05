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
from scipy.stats import pearsonr, spearmanr
import math

import matplotlib.colors as mcolors

from scripts.visualization.utils.renderizer import EGMRenderer_EGM


class DF_MAPS_3D_PLOTTER:

    def __init__(self, model_name, model_path, geom_path_CF, output_directory, labels_mode, tikhonov=False, time=500):
        self.model_name=model_name
        self.model_path=model_path
        self.geom_path_CF=geom_path_CF
        self.output_directory=output_directory
        self.duration = time
        self.labels_mode = labels_mode
        self.tikhonov=tikhonov

    
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
                model = sio.loadmat(self.model_path)[self.model_name]
                print('Loaded model ', self.model_name)
            except:
                modified_name = self.model_name[0]
                self.model_name = "model" + modified_name.strip()
                model = sio.loadmat(self.model_path)[self.model_name]
                print('Loaded model ', self.model_name)                

            try:
                geom = sio.loadmat(self.geom_path_CF)["geometries"]
            except:
                geom = sio.loadmat(self.geom_path_CF)

            # Extraer datosy
            try:
                y_reconstructed = model["reconstruction"][0,0]
            except:
                normalized_name = " ".join(self.model_name.split())
                self.model_name = "model" + normalized_name.strip()

            y_label = model["label"][0,0]
            heart = geom["heart"]
            faces = heart["faces"][0, 0] - 1  # Convertir a índice base 0
            vertices = heart["vertices"][0, 0]

        return y_reconstructed, y_label, faces, vertices
    
    def plot_3d_mesh_prediction(self, y_reconstructed, y_label, faces, vertices, normalizar=False):
        # Crear renderizadores para var_represent y var_represent_original
        renderer_var = EGMRenderer_EGM(faces, vertices, min_val=-0.6, max_val=0.6)
        try:
            renderer_label = EGMRenderer_EGM(faces, vertices,  min_val=-0.6, max_val=0.6)
        except:
            y_label=y_label[0][0]
            renderer_label = EGMRenderer_EGM(faces, vertices,  min_val=-0.6, max_val=0.6)

        if y_reconstructed.ndim>2:
            y_reconstructed = y_reconstructed.reshape(-1, y_reconstructed.shape[2]) 


            var_represent = y_reconstructed

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
            if self.tikhonov:
                output_video = os.path.join(self.output_directory, f"df_map_{view}_tik.gif")
            else:
                output_video = os.path.join(self.output_directory, f"df_map_{view}_dl_2.gif")

            fps = 10  # Frames por segundo

            array_frames=[]
            # Iterar sobre instantes y generar frames
            for instant in range(0, 1, 1):  # Cada 500 instantes
                # Actualizar los datos escalares de los dos renderizadores
                vtk_scalars_var = numpy_to_vtk(var_represent[ :], deep=True)
                vtk_scalars_label = numpy_to_vtk(var_represent_original[ :], deep=True)

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



    def plot_3d_mesh_df(self, y_reconstructed, y_label, vertices,faces,min_val_value,max_val_value, normalizar=True ):


        # Crear renderizadores para var_represent y var_represent_original
        renderer_var = EGMRenderer_EGM(faces, vertices, min_val=min_val_value, max_val=max_val_value)
        renderer_label = EGMRenderer_EGM(faces, vertices, min_val=min_val_value, max_val=max_val_value)

        if y_reconstructed.ndim>2:
            y_reconstructed = y_reconstructed.reshape(-1, y_reconstructed.shape[2]) 
        
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

            # Crear la paleta "cool to warm"
            custom_lut,cmap = self.create_custom_colormap_df(min_val_value,max_val_value)

            # Asignar la paleta al mapper de ambos renderizadores
            renderer_var.mapper.SetLookupTable(custom_lut)
            renderer_label.mapper.SetLookupTable(custom_lut)

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
            custom_lut.SetNumberOfTableValues(256)
            custom_lut.Build()

            for i in range(255):
                r, g, b, _ = cmap(i / 255.0)  # Obtener color del colormap
                custom_lut.SetTableValue(i, r, g, b, 1.0)  # Asignar color en la tabla de VTK

            # 3️⃣ Crear un mapper en VTK y asignar la tabla de colores
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetLookupTable(custom_lut)
            mapper.SetScalarRange(-math.pi, math.pi)  # Ajustar al rango de valores de tus datos

            # Scalar bars remain unchanged but without titles
            scalar_bar_var = vtk.vtkScalarBarActor()
            scalar_bar_var.SetLookupTable(renderer_var.mapper.GetLookupTable())
            scalar_bar_var.GetLabelTextProperty().SetColor(0, 0, 0)  # Set font color to black
            scalar_bar_var.GetLabelTextProperty().SetItalic(False)  # Ensure text is not italicized
            scalar_bar_var.GetLabelTextProperty().SetShadow(False) 
            scalar_bar_var.GetLabelTextProperty().SetFontFamilyToArial()  # Set font to Arial
            scalar_bar_var.GetLabelTextProperty().SetFontSize(15) 
            scalar_bar_var.SetMaximumNumberOfColors(256)  # Usa más colores en la barra
            scalar_bar_var.SetTitle("DF (Hz)")  # Título de la barra
            scalar_bar_var.SetPosition(0.8, 0.2)  # Ajustar la posición de la barra de color
            scalar_bar_var.SetWidth(0.05)  # Ajustar el ancho de la barra
            scalar_bar_var.SetHeight(0.6)  # Ajustar la altura de la barra
            # Configurar la orientación del título para que esté en vertical
            title_text_property = scalar_bar_var.GetTitleTextProperty()
            title_text_property.SetFontSize(5)  # Ajustar el tamaño de fuente
            title_text_property.SetColor(0, 0, 0)  # Color negro para el título
            scalar_bar_var.SetNumberOfLabels(256)  # Aumenta el número de etiquetas para una transición más suave
            scalar_bar_var.SetNumberOfLabels(3)

            scalar_bar_label = vtk.vtkScalarBarActor()
            scalar_bar_label.SetLookupTable(renderer_label.mapper.GetLookupTable())
            scalar_bar_label.GetLabelTextProperty().SetColor(0, 0, 0)  # Set font color to black
            scalar_bar_label.SetNumberOfLabels(3)
            scalar_bar_label.SetMaximumNumberOfColors(256) 
            scalar_bar_label.SetTitle("DF (Hz)")  # Título de la barra
            # Configurar la orientación del título para que esté en vertical
            title_text_property = scalar_bar_label.GetTitleTextProperty()
            title_text_property.SetFontSize(5)  # Ajustar el tamaño de fuente
            title_text_property.SetColor(0, 0, 0)  # Color negro para el título
            scalar_bar_label.SetPosition(0.9, 0.2)  # Ajustar la posición de la barra de color
            scalar_bar_label.SetWidth(0.05)  # Ajustar el ancho de la barra
            scalar_bar_label.SetHeight(0.6)  # Ajustar la altura de la barra

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
            if self.tikhonov:
                output_video = os.path.join(self.output_directory, f"df_map_{view}_tik.gif")
            else:
                output_video = os.path.join(self.output_directory, f"df_map_{view}_dl2.gif")

            instant=0
            array_frames=[]
            # Iterar sobre instantes y generar frames
            # Actualizar los datos escalares de los dos renderizadores
            vtk_scalars_var = numpy_to_vtk(var_represent, deep=True)
            vtk_scalars_label = numpy_to_vtk(var_represent_original, deep=True)

            renderer_var.mesh.GetPointData().SetScalars(vtk_scalars_var)
            renderer_label.mesh.GetPointData().SetScalars(vtk_scalars_label)

            renderer_var.mesh.Modified()
            renderer_label.mesh.Modified()

            # Renderizar la ventana con subplots
            render_window.Render()

            # Guardar la imagen como PNG
            window_to_image_filter.Modified()
            window_to_image_filter.Update()
            output_file = os.path.join(self.output_directory, f"frame{instant}.png")

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
            print(f"Frame guardado: {output_file}")
            os.remove(output_file)

            # Liberar el objeto VideoWriter
            imageio.mimsave(output_video, array_frames,format='GIF', fps=10)

            print(f"Video guardado en {output_video}") 

    def plot_3d_mesh_phase(self, y_reconstructed, y_label, vertices,faces,min_val_value,max_val_value, normalizar=True ):


        # Crear renderizadores para var_represent y var_represent_original
        renderer_var = EGMRenderer_EGM(faces, vertices, min_val=min_val_value, max_val=max_val_value)
        try:
            renderer_label = EGMRenderer_EGM(faces, vertices, min_val=min_val_value, max_val=max_val_value)
        except:
            y_label=y_label[0][0]
            renderer_label = EGMRenderer_EGM(faces, vertices, min_val=min_val_value, max_val=max_val_value)

        if y_reconstructed.ndim>2:
            y_reconstructed = y_reconstructed.reshape(-1, y_reconstructed.shape[2]) 
        
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

            # Crear la paleta "cool to warm"
            custom_lut,cmap = self.create_custom_colormap_phase(min_val_value,max_val_value)

            # Asignar la paleta al mapper de ambos renderizadores
            renderer_var.mapper.SetLookupTable(custom_lut)
            renderer_label.mapper.SetLookupTable(custom_lut)

            # Configurar fondo blanco
            renderer_var.renderer.SetBackground(1, 1, 1)  
            renderer_label.renderer.SetBackground(1, 1, 1)  

            # Create a text actor for var_represent
            title_var = vtk.vtkTextActor()
            title_var.SetInput(f"Reconstructed")
            title_varprop = title_var.GetTextProperty()
            title_varprop.SetFontFamilyToArial()
            title_varprop.SetFontSize(25)
            #title_varprop.BoldOn()
            title_varprop.SetColor(0, 0, 0)  # Black color
            title_var.SetPosition(300, 500)  # Adjust position manually as needed
            #title_var.SetFontFamilyToArial()
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
            custom_lut.SetNumberOfTableValues(256)
            custom_lut.Build()

            for i in range(255):
                r, g, b, _ = cmap(i / 255.0)  # Obtener color del colormap
                custom_lut.SetTableValue(i, r, g, b, 1.0)  # Asignar color en la tabla de VTK

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetLookupTable(custom_lut)
            mapper.SetScalarRange(min_val_value, max_val_value)  # Ajustar al rango de valores de tus datos


            # Scalar bars remain unchanged but without titles
            scalar_bar_var = vtk.vtkScalarBarActor()
            scalar_bar_var.SetLookupTable(renderer_var.mapper.GetLookupTable())
            scalar_bar_var.GetLabelTextProperty().SetColor(0, 0, 0)  # Set font color to black
            scalar_bar_var.GetLabelTextProperty().SetItalic(False)  # Ensure text is not italicized
            scalar_bar_var.GetLabelTextProperty().SetShadow(False) 
            scalar_bar_var.GetLabelTextProperty().SetFontFamilyToArial()  # Set font to Arial
            scalar_bar_var.GetLabelTextProperty().SetFontSize(5) 
            scalar_bar_var.GetLabelTextProperty().SetFontFamilyToArial()
            scalar_bar_var.SetMaximumNumberOfColors(700)  
            scalar_bar_var.SetNumberOfLabels(3)
            scalar_bar_var.SetTextPositionToPrecedeScalarBar()
            #scalar_bar_var.SetFontSize(5)  # Ajustar el tamaño de fuente
            #scalar_bar_var.SetFontFamilyToArial()  

            scalar_bar_label = vtk.vtkScalarBarActor()
            scalar_bar_label.SetLookupTable(renderer_label.mapper.GetLookupTable())
            scalar_bar_label.GetLabelTextProperty().SetColor(0, 0, 0)  # Set font color to black
            scalar_bar_label.SetNumberOfLabels(3)
            scalar_bar_label.SetMaximumNumberOfColors(700) 
            scalar_bar_label.GetLabelTextProperty().SetFontSize(5) 
            scalar_bar_label.GetLabelTextProperty().SetFontFamilyToArial()
            scalar_bar_label.SetTextPositionToPrecedeScalarBar()
            #scalar_bar_label.SetFontSize(5)  # Ajustar el tamaño de fuente
            #scalar_bar_label.SetFontFamilyToArial()  

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
            if self.tikhonov:
                output_video = os.path.join(self.output_directory, f"phase_map_{view}_tik.gif")
            else:
                output_video = os.path.join(self.output_directory, f"phase_map_{view}_dl.gif")

            array_frames=[]
            for instant in range(0, self.duration): 

                # Iterar sobre instantes y generar frames
                # Actualizar los datos escalares de los dos renderizadores
                vtk_scalars_var = numpy_to_vtk(var_represent[:, instant], deep=True)
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
                output_file = os.path.join(self.output_directory, f"frame{instant}.png")
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
                os.remove(output_file)

            # Liberar el objeto VideoWriter
            imageio.mimsave(output_video, array_frames,format='GIF', fps=10)

            print(f"Video guardado en {output_video}") 

    def create_custom_colormap_df(self,min_val_value, max_val_value):

        colors = [
            (0.0, "white"),   
            (0.5, "blue"), 
            (1.0, "red")    
            ]

        cmap = mcolors.LinearSegmentedColormap.from_list("custom_colormap", colors, N=255)

        lut = vtk.vtkLookupTable()
        return lut, cmap
    
    def create_custom_colormap_phase(self,min_val_value, max_val_value):

        # Definir los colores con rosa añadido
        colors = [
            (0.0, "white"),   # Mínimo: Blanco
            (0.15, "yellow"), # Amarillo
            (0.3, "red"),     # Rojo
            (0.5, "green"),  # Verde
            (0.6, "blue"),    # Azul
            (0.8, "purple"),  # Morado
            (1.0, "black")    # Máximo: Negro
        ]

        cmap = mcolors.LinearSegmentedColormap.from_list("custom_colormap", colors, N=700)

        # 2️⃣ Convertir colormap de Matplotlib a una tabla de colores en VTK
        lut = vtk.vtkLookupTable()
        return lut, cmap
    

    def __call__(self):
        y_reconstructed, y_label, faces, vertices=self.load_geometry_and_egm()      

        if self.labels_mode:
            self.plot_3d_mesh_label(y_reconstructed, y_label, faces, vertices)
        else:
            self.plot_3d_mesh_prediction(y_reconstructed, y_label, faces, vertices)

