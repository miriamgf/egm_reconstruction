import h5py  


class LoadExploreData:
    def __init__(self):
        self.path_to_data="/home/pdi/miriamgf/tesis/Autoencoders/Real_data/HEartLab/data_E18_F02_R02.mat"
    
    def load_and_select_fields(self):
        with h5py.File(self.path_to_data, "r") as f:
            # Listar las variables almacenadas
            print("Variables disponibles:", list(f.keys()))
            dataset_raw=f["data"]["raw"]["electric"]
            dataset_ecgi=f["data"]["ECGi"]
            tank_el_position=dataset_ecgi['geometries']['tank_el_position']

            #Raw Signals
            signal_MEA1_RA=f["data"]["raw"]["electric"]["MEA1"]
            signal_MEA3_LA=f["data"]["raw"]["electric"]["MEA3"]
            signal_tank=f["data"]["raw"]["electric"]["TANK"]

            print("MEA1 (RA) signal shape:", signal_MEA1_RA.shape)
            print("MEA3 (LA) signal shape:", signal_MEA3_LA.shape)
            print("Tank signal shape:", signal_tank.shape)

            #Geometry heart
            geometry_heart=f["data"]["ECGi"]["geometries"]["heart"]
            heart_faces=geometry_heart["faces"]
            heart_vertices=geometry_heart["vertices"]

            print("Heart geometry vertices shape:", heart_vertices.shape )
            print("Heart geometry faces shape:", heart_faces.shape )

            #Geometry tank
            geometry_tank=f["data"]["ECGi"]["geometries"]["tank"]
            tank_faces=geometry_tank["faces"]
            tank_vertices=geometry_tank["vertices"]
      
            print("Tank geometry vertices shape:", tank_faces.shape )
            print("Tank geometry faces shape:", tank_vertices.shape )

            #Matrix Signals (Interpolated)
            matrix_signal_MEA1_RA=f["data"]["matrix"]["electric"]["MEA1"]
            matrix_signal_MEA3_LA=f["data"]["matrix"]["electric"]["MEA3"]
            matrix_signal_tank=f["data"]["matrix"]["electric"]["TANK"]

            print("Matrix signal MEA1 shape:", matrix_signal_MEA1_RA)
            print("Matrix signal MEA3 shape:", matrix_signal_MEA3_LA)
            print("Matrix signal tank shape:", matrix_signal_tank)
    
    def select_fields_and_save_as_mat(self):
        '''
        This function selects the fields of interest and saves them as a .mat file
        with same format as simulation data
        '''

        with h5py.File(self.path_to_data, "r") as f:
            # Listar las variables almacenadas
            print("Variables disponibles:", list(f.keys()))
            dataset_raw=f["data"]["raw"]["electric"]
            dataset_ecgi=f["data"]["ECGi"]
            tank_el_position=dataset_ecgi['geometries']['tank_el_position']


            #Raw Signals
            signal_MEA1_RA=f["data"]["raw"]["electric"]["MEA1"]
            signal_MEA3_LA=f["data"]["raw"]["electric"]["MEA3"]
            signal_tank=f["data"]["raw"]["electric"]["TANK"]


            #Save as .mat file
            import scipy.io as sio

            path_to_save="/home/pdi/miriamgf/tesis/Autoencoders/Real_data/HEartLab/data_E18_F02_R02_selection.mat"

            sio.savemat(path_to_save, {'signal_MEA1_RA': signal_MEA1_RA,
                                                'signal_MEA3_LA': signal_MEA3_LA,
                                                'signal_tank': signal_tank, 
                                                "tank_el_position": tank_el_position})
            
            print("Saved as .mat file in:", path_to_save)

    def run(self):
        self.load_and_select_fields()
        self.select_fields_and_save_as_mat()



if __name__ == "__main__":
    loader = LoadExploreData()  
    loader.run()  


        
