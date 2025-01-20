#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 14:04:38 2025

Inverse problem on data from Angelica/Joao

@author: obarquero
"""


#%% Read data

#files names
signal_file = "../../01 - data/electric_data_Exx_Fxx_Rxx_filtered.mat"
electrodes_idx_file = '../../01 - data/eletrodos_LR.mat'
heart_geo_file = "../../01 - data/../../01 - data/heart_geometry_20000_exp14.mat"
tank_geo_file = "../../01 - data/LR_tank.mat"
mtransfer_file = "../../01 - data/MTransfer_exp14_LR_20000.mat"

#read data
