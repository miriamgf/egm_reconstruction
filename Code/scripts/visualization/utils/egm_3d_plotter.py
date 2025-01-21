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

from scripts.visualization.utils.renderizer import EGMRenderer_EGM


class EGM_3D_PLOTTER:

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
    
    def plot_3d_mesh_prediction(self, y_reconstructed, y_label, faces, vertices, normalizar=True):
        # Crear renderizadores para var_represent y var_represent_original
        renderer_var = EGMRenderer_EGM(faces, vertices, min_val=-0.6, max_val=0.6)
        try:
            renderer_label = EGMRenderer_EGM(faces, vertices,  min_val=-0.6, max_val=0.6)
        except:
            y_label=y_label[0][0]
            renderer_label = EGMRenderer_EGM(faces, vertices,  min_val=-0.6, max_val=0.6)

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
            if self.tikhonov:
                output_video = os.path.join(self.output_directory, f"egm_rec_tikhonov_{view}.gif")
            else:
                output_video = os.path.join(self.output_directory, f"egm_rec_DL_{view}.gif")

            fps = 10  # Frames por segundo

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

    def __call__(self):
        y_reconstructed, y_label, faces, vertices=self.load_geometry_and_egm()      

        if self.labels_mode:
            self.plot_3d_mesh_label(y_reconstructed, y_label, faces, vertices)
        else:
            self.plot_3d_mesh_prediction(y_reconstructed, y_label, faces, vertices)

