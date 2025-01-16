**README**  

1. Data
- sinus_raw: raw data in .mat  
	- Rows: Electrodes.  
	- Columns: Samples.  
	- MEA1 (Right Atrium): 1–16 (16 electrodes).  
	- MEA2 (Ventricle): 17–32 (16 electrodes).  
	- MEA3 (Left Atrium): 65–80 (16 electrodes).  
	- Tank Electrodes: 129–174, 177–190 (60 electrodes).
	- Sample Rate: 4000 Hz.  
	- TTL: Used to synchronize with optical recordings.
- eletrodos_LR: vector that contains the correspondent node of each tank electrode in the tank 3D geometry.
- heart_geometry: 3D heart geometry (20002 nodes).
- MTransfer: transfer matrix obtained with the heart and tank geometries
- tank_geometry: 3D tank geometry
- structure.oebin: raw data.

2. Filtering codes
- main_extraction_filtering code extracts and filters from .oebin file.

3. ECGi codes
- 01 - correct_geometries: checks and correct the position of the heart geometry.
- 02 - transfer_matrix: transfer_matrix_main calculates transfer matrix using heart and tank geometries.
- 03 - estimation: estimation_main code calls the files and the functions to reconstruct the signals.
