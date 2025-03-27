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

from scripts.visualization.utils.renderizer import EGMRenderer_BSP


class BSP_3D_PLOTTER:
    """
    Created by Miriam Gutierrez on Jan 2025
    Last mofified: Jan 2025 (Miriam Gutierrez)

    
    A class for visualizing Body Surface Potential Mapping (BSPM) data in both 2D and 3D formats.

    Attributes:
        torso_num (int): The number identifying the specific torso.
        all_torsos_names (list): List of all torso names.
        y_list (list): A list containing BSPM signals.
        X_1channel (ndarray): BSPM in image format (Lx64x32).
        output_directory (str): Directory to save output videos and frames.
        model_path (str): Path to the model file.
        duration (int): Duration of the video or visualization in frames.

    Methods:
        load_geometry_and_bspm(normalizar_bsp_64=True):
            Loads the geometry and BSPM data, optionally normalizing the BSPM signals.
        
        plot_64_videos(bsp_64):
            Generates and saves a 2D visualization video of BSPM data.

        plot_3d_mesh(bspm_signal, faces, vertices, normalizar=True):
            Visualizes BSPM data on a 3D torso mesh and creates a video.

        call():
            Orchestrates the loading of data and visualization processes.
    """

    def __init__(self, torso_num, all_torsos_names, y_list, X_1channel, output_directory, model_path, time):
        """
        Initializes the BSP_3D_PLOTTER with required data.

        Args:
            torso_num (int): The torso number.
            all_torsos_names (list): Names of all torsos.
            y_list (list): List of BSPM signals.
            X_1channel (ndarray): Data for 2D visualization.
            output_directory (str): Directory to save outputs.
            model_path (str): Path to the model file.
            time (int): Duration for visualization.
        """
        self.torso_num = torso_num
        self.all_torsos_names = all_torsos_names
        self.y_list = y_list
        self.X_1channel = X_1channel
        self.output_directory = output_directory
        self.model_path = model_path
        self.duration = time

    def load_geometry_and_bspm(self, normalizar_bsp_64=True):
        """
        Loads the geometry and BSPM data from the model file, optionally normalizing the BSPM data.

        Args:
            normalizar_bsp_64 (bool, optional): Whether to normalize the BSPM data. Defaults to True.

        Returns:
            tuple: Normalized BSPM data, original BSPM signal, faces, and vertices for 3D plotting.
        """
        model = sio.loadmat(self.model_path)
        matrix_A = model["TransferMatrix"]
        torso = model["torso"]
        faces = torso["faces"][0, 0] - 1  # Convert to 0-based indexing
        vertices = torso["vertices"][0, 0]

        torso_name = f"Torso{self.torso_num}_mod.mat"
        torso_index = self.all_torsos_names.index(torso_name)
        bspm_signal = self.y_list[torso_index]['y']

        # Split and reshape BSPM data for visualization
        X_1channel_split = np.array_split(self.X_1channel, 10)
        bsp_64 = X_1channel_split[torso_index]
        bsp_64_reshaped = bsp_64.reshape(bsp_64.shape[0], -1)

        if normalizar_bsp_64:
            bsp_64_n = np.zeros_like(bsp_64_reshaped)
            for nodo in range(bsp_64_reshaped.shape[1]):  
                min_val = np.min(bsp_64_reshaped[:, nodo])
                max_val = np.max(bsp_64_reshaped[:, nodo])
                
                if max_val != min_val:  
                    bsp_64_n[:, nodo] = (bsp_64_reshaped[:, nodo] - min_val) / (max_val - min_val)
                else:
                    bsp_64_n[:, nodo] = 0  
        
        bsp_64 = bsp_64_n.reshape(bsp_64_n.shape[0], 12, 32)
        return bsp_64, bspm_signal, faces, vertices

    def plot_64_videos(self, bsp_64):
        """
        Generates a 2D visualization of BSPM data and saves it as a video.

        Args:
            bsp_64 (ndarray): Normalized BSPM data.
        """
        output_video = os.path.join(self.output_directory, f"video_bsp_64.gif")
        filenames = []
        for instant in range(0, self.duration, 1):
            path_frames = os.path.join(self.output_directory, f"frame_{instant:04d}.png")
            frame = bsp_64[instant, :, :]
            plt.figure()
            plt.imshow(frame)
            plt.colorbar(label='mV')
            plt.savefig(path_frames)
            filenames.append(path_frames)
            plt.close()

        with imageio.get_writer(output_video, mode='I', duration=0.1) as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)
        print('64 lead video saved in', output_video)

    def plot_3d_mesh(self, bspm_signal, faces, vertices, normalizar=True):
        """
        Visualizes BSPM data on a 3D torso mesh and creates a video.

        Args:
            bspm_signal (ndarray): BSPM signal data.
            faces (ndarray): Mesh faces data.
            vertices (ndarray): Mesh vertices data.
            normalizar (bool, optional): Whether to normalize the BSPM signals. Defaults to True.
        """
        if normalizar:
            y_label_n = np.zeros_like(bspm_signal)
            for nodo in range(bspm_signal.shape[1]):  
                min_val = np.min(bspm_signal[:, nodo])
                max_val = np.max(bspm_signal[:, nodo])
                
                if max_val != min_val:  
                    normalized = (bspm_signal[:, nodo] - min_val) / (max_val - min_val)
                    y_label_n[:, nodo] = 2 * normalized - 1
                else:
                    y_label_n[:, nodo] = 0

            var_represent = y_label_n
            var_represent_original = y_label_n
        else:
            var_represent = bspm_signal
            var_represent_original = bspm_signal

        # Set up renderers for visualization
        renderer_var = EGMRenderer_BSP(faces, vertices, min_val=-0.6, max_val=0.6, view="front")
        renderer_label = EGMRenderer_BSP(faces, vertices, min_val=-0.6, max_val=0.6, view="back")

        renderer_var.renderer.SetViewport(0.0, 0.0, 0.5, 1.0)
        renderer_label.renderer.SetViewport(0.5, 0.0, 1.0, 1.0)
        renderer_var.renderer.SetBackground(1, 1, 1)
        renderer_label.renderer.SetBackground(1, 1, 1)

        title_var = vtk.vtkTextActor()
        title_var.SetInput(f"Torso {self.torso_num} (Front)")
        title_varprop = title_var.GetTextProperty()
        title_varprop.SetFontFamilyToArial()
        title_varprop.SetFontSize(25)
        title_varprop.SetColor(0, 0, 0)
        title_var.SetPosition(300, 500)
        renderer_var.renderer.AddActor2D(title_var)

        title_label = vtk.vtkTextActor()
        title_label.SetInput(f"Torso {self.torso_num} (Back)")
        title_labelprop = title_label.GetTextProperty()
        title_labelprop.SetFontFamilyToArial()
        title_labelprop.SetFontSize(25)
        title_labelprop.SetColor(0, 0, 0)
        title_label.SetPosition(300, 500)
        renderer_label.renderer.AddActor2D(title_label)

        scalar_bar_var = vtk.vtkScalarBarActor()
        scalar_bar_var.SetLookupTable(renderer_var.mapper.GetLookupTable())
        scalar_bar_var.GetLabelTextProperty().SetColor(0, 0, 0)
        scalar_bar_var.GetLabelTextProperty().SetFontFamilyToArial()
        scalar_bar_var.GetLabelTextProperty().SetFontSize(15)
        scalar_bar_var.SetNumberOfLabels(5)

        scalar_bar_label = vtk.vtkScalarBarActor()
        scalar_bar_label.SetLookupTable(renderer_label.mapper.GetLookupTable())
        scalar_bar_label.GetLabelTextProperty().SetColor(0, 0, 0)
        scalar_bar_label.SetNumberOfLabels(5)

        renderer_var.renderer.AddActor2D(scalar_bar_var)
        renderer_label.renderer.AddActor2D(scalar_bar_label)

        render_window = vtk.vtkRenderWindow()
        render_window.SetSize(1600, 600)
        render_window.AddRenderer(renderer_var.renderer)
        render_window.AddRenderer(renderer_label.renderer)

        center = renderer_var.mesh.GetCenter()
        for renderer in [renderer_var, renderer_label]:
            renderer.camera.SetPosition(center[0], center[1], center[2] + 200)
            renderer.camera.SetFocalPoint(center[0], center[1], center[2])
            renderer.camera.SetViewUp(1, 0, 0)
            renderer.camera.Elevation(270)
            renderer.camera.Azimuth(0)
            renderer.renderer.ResetCameraClippingRange()

        window_to_image_filter = vtk.vtkWindowToImageFilter()
        window_to_image_filter.SetInput(render_window)

        fps = 10
        array_frames = []
        for instant in range(0, self.duration, 50):
            vtk_scalars_var = numpy_to_vtk(var_represent[:, instant], deep=True)
            vtk_scalars_label = numpy_to_vtk(var_represent_original[:, instant], deep=True)

            renderer_var.mesh.GetPointData().SetScalars(vtk_scalars_var)
            renderer_label.mesh.GetPointData().SetScalars(vtk_scalars_label)

            renderer_var.mesh.Modified()
            renderer_label.mesh.Modified()

            render_window.Render()
            window_to_image_filter.Modified()
            window_to_image_filter.Update()

            output_file = os.path.join(self.output_directory, f"frame_{instant:04d}.png")
            writer = vtk.vtkPNGWriter()
            writer.SetFileName(output_file)
            writer.SetInputData(window_to_image_filter.GetOutput())
            writer.Write()

            image_data = window_to_image_filter.GetOutput()
            dims = image_data.GetDimensions()
            vtk_array = vtk.util.numpy_support.vtk_to_numpy(image_data.GetPointData().GetScalars())
            frame = vtk_array.reshape((dims[1], dims[0], 3))[::-1]

            array_frames.append(frame)
            os.remove(output_file)


        output_gif = os.path.join(self.output_directory, f"video_torso.gif")
        imageio.mimsave(output_gif, array_frames, format='GIF', fps=fps)
        print(f"Video saved in {output_gif}")

    def __call__(self):
        """
        Main method to load data and generate visualizations.
        """
        bsp_64, bspm_signal, faces, vertices = self.load_geometry_and_bspm()
        self.plot_64_videos(bsp_64)
        self.plot_3d_mesh(bspm_signal, faces, vertices)
