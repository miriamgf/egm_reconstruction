load ecg_fv
fs = 200; %Sampling frequency 200Hz

%Estimation of dominant(fundamental) frequency
[ff1,indff1,bwfromf01,z1,Pz1,fz1] = df_ECG(ecg_fv(1,:),200);
pause
close all
[ff2,indff2,bwfromf02,z2,Pz2,fz2] = df_ECG(ecg_fv(2,:),200);
pause
close all
[ff3,indff3,bwfromf03,z3,Pz3,fz3] = df_ECG(ecg_fv(3,:),200);
pause
close all

%% Complete spectral analysis


%Note, df_ECG is called within df_ECG_analysis, and output parameters from
%df_ECG are saved in the struc fvpar_xxx_x

fvpar_chan_1 = df_ECG_Analysis(ecg_fv(1,:),fs);
pause
close all
fvpar_chan_2 = df_ECG_Analysis(ecg_fv(2,:),fs);
pause
close all
fvpar_chan_3 = df_ECG_Analysis(ecg_fv(3,:),fs);
pause
close all
%Verifica que tanto en df_ECG_Analysis como en privado/df_ECG existen unas
%variables drawflag que están colocadas a 1 por defecto, no pasadas como
%parámetro.
%
%Con estas variables a 1 se representan ciertas cosas.
%Cuando hayas comprobado que entiendes la mayoría de las gráficas, para
%realizar los análisis sobre todos los pacientes, lo mejor es que coloques
%estas variables a 0, de forma que no te salgan todas las gráficas.