# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 10:28:18 2018

@author: Miguel Ángel
"""

import forward_inverse_problem as fip
import filtering
import data_load as dl
import precompute_matrix as pre_m
import metrics
import freq_phase_analysis as freq_pha
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
import numpy as np

# Initial parameters
fs = 2034.5
model = 'Valencia_Pat2_113014'
constrained_basket = 'basket'

# Load EGMs, ECGs, transfer matrix and geometrical model. 
x,y = dl.load_egms_real(model)
A = dl.load_transfer_real(model)
atrial_model, torso_model = dl.load_geometry_real(model)

# EGMs and ECGs filtering.
x,_ = filtering.detrendSpline(x,fs,l_w = 0.25)
y,_ = filtering.detrendSpline(y,fs,l_w = 0.25)
x_filtered = filtering.ECG_filtering_real(x, fs)
y_filtered = filtering.ECG_filtering_real(y, fs)

#  Computing of inverse problem required matrix (AA, L, LL, D).
AA_0,L_0,LL_0 = pre_m.precompute_matrix_real(A, atrial_model, model, 0)
AA_1,L_1,LL_1 = pre_m.precompute_matrix_real(A, atrial_model, model, 1)
AA_2,L_2,LL_2 = pre_m.precompute_matrix_real(A, atrial_model, model, 2)

""" Analysis for reference signals"""
# Generate D matrix and x sorted matrix with each signal in the correspondent node (row).
D,known_nodes, x_ref_cons_full = pre_m.D_nodes_constrained_real(model, x_filtered, A.shape[1])

# DF Analysis Ref signals
f, pxx_ref, df_ref, signals_ref_df = freq_pha.df_estimation_peak(x_ref_cons_full,fs)

""" Analysis for Tikh0"""
# Inverse problem (Tikh0)
#x_hat_tikh0, lambda_opt_tikh0 = fip.classical_tikhonov(A, AA_0, L_0, LL_0, y_filtered)

x_hat_tikh0 = np.matmul(np.matmul(np.linalg.inv(AA_0+3.1623e-5*LL_0),np.transpose(A)),y_filtered);

# DF Analysis Tikh0
f_tikh0, pxx_tikh0, df_tikh0, signals_tikh0_df = freq_pha.df_estimation_peak(x_hat_tikh0,fs)

# RMSE DF Tikh0 (All known nodes)
RMSE_DF_tikh0 = np.sqrt((df_tikh0[known_nodes]-df_ref[known_nodes])**2)
mean_RMSE_DF_tikh0 = np.mean(RMSE_DF_tikh0)
std_RMSE_DF_tikh0 = np.std(RMSE_DF_tikh0)

""" Analysis for Cons1 (All known nodes)"""
# Inverse problem (Cons1)
#x_hat_cons1,lambda_opt_cons1 = fip.constrained_tikhonov(A, AA_1, L_1, LL_1, D, y_filtered, x_ref_cons_full)
#x_hat_cons1,lambda_opt_cons1,_,_,_ = fip.constrained_tikhonov_noiter(A, AA_1, L_1, LL_1, D, y_filtered, x_ref_cons_full)

inv_term = np.linalg.inv(AA_1+1e-4*LL_1+1*np.matmul(np.transpose(D),D))
sec_term = np.matmul(np.transpose(A),y_filtered)+1*np.matmul(np.transpose(D),x_ref_cons_full)
x_hat_cons1 = np.matmul(inv_term,sec_term);

# DF Analysis Cons1
f_cons1, pxx_cons1, df_cons1, signals_cons1_df = freq_pha.df_estimation_peak(x_hat_cons1,fs)

# RMSE DF Cons1 (All known nodes)
RMSE_DF_cons1 = np.sqrt((df_cons1[known_nodes]-df_ref[known_nodes])**2)
mean_RMSE_DF_cons1 = np.mean(RMSE_DF_cons1)
std_RMSE_DF_cons1 = np.std(RMSE_DF_cons1)


""" Analysis for lambda analysis Cons1 """
# Lambda test values
lambda_test_1=np.logspace(-2,-5,4)
lambda_test_2=np.logspace(0,-3,4)

# For loop over those lambdas
for i in lambda_test_1:
    for j in lambda_test_2:
        # Inverse problem
        inv_term = np.linalg.inv(AA_1+i*LL_1+j*np.matmul(np.transpose(D),D))
        sec_term = np.matmul(np.transpose(A),y_filtered)+j*np.matmul(np.transpose(D),x_ref_cons_full)
        x_hat = np.matmul(inv_term,sec_term);
        # Plot nodes (normalized) 
        x_hat = x_hat[1544,100:-500]/np.max(x_hat[1544,100:-500]) #Normalize and remove start and end segments.
        x_gt = x_ref_cons_full[1544,100:-500]/np.max(x_ref_cons_full[1544,100:-500])
        t=np.linspace(0,x_gt.shape[0]/fs,x_gt.shape[0])
        plt.figure(),
        plt.plot(t,x_gt,'k'),plt.plot(t,x_hat,'r')
        plt.xlim((0,2)),plt.xlabel('Time (s)'),plt.ylabel('Amplitude (Normalized)')
        plt.legend(('GT','Cons1')),plt.title(r'Cons1, node 1534. $\lambda_1$=%.3e, $\lambda_2$=%.3e'%(i,j))


""" Analysis for Cons1 - Dropout mode
known_nodes_list=[]
x_hat_cons1_list=[]
lambda_opt_cons1_list=[]
dropped_nodes_list=[]
pxx_classical_cons1_list = []
df_classical_cons1_list=[]
z_egms_BS_cons1_list = []
P_z_BS_cons1_list = []
df_BS_cons1_list = []
RMSE_DF_cons1_list=[]
RMSE_DF_BS_cons1_list=[]
RMSE_DF_tikh0_list=[]
RMSE_DF_BS_tikh0_list=[]

for k in range(0,300):
    print('Constrained Tikhonov dropout mode. Iteration n. %d' % (k+1))

    # Compute known nodes and dropped nodes
    D,known_nodes,x_ref_cons, dropped_nodes = pre_m.D_nodes_constrained_real_dropout(model, x, A.shape[1])
    known_nodes_list.append(known_nodes)
    dropped_nodes_list.append(dropped_nodes)
    
    # Compute inverse problem
    x_hat_cons1,lambda_opt_list_cons1 = fip.constrained_tikhonov(A, AA_1, L_1, LL_1, D, y, x_ref_cons)
    x_hat_cons1_list.append(x_hat_cons1)
    lambda_opt_cons1_list.append(lambda_opt_list_cons1)

    # DF Analysis Cons1 (Classical)
    f_classical, pxx_classical_cons1, df_classical_cons1 = freq_pha.df_estimation_peak(x_hat_cons1,fs)
    pxx_classical_cons1_list.append(pxx_classical_cons1)
    df_classical_cons1_list.append(df_classical_cons1)
    
    # DF Analysis Cons1 (BS)
    z_egms_BS_cons1, P_z_BS_cons1, f_BS_cons1, df_BS_cons1 = freq_pha.botterom_smith_analysis(x_hat_cons1,fs,f1 = 10,f2=50)
    z_egms_BS_cons1_list.append(z_egms_BS_cons1)
    P_z_BS_cons1_list.append(P_z_BS_cons1)
    df_BS_cons1_list.append(df_BS_cons1)
    
    # Compute RMSE for DFs in x_ref nodes with Tikh0 (estimation via DF Peak).
    RMSE_DF_tikh0=[]
    for l in range(0,len(dropped_nodes)):
        RMSE_DF_tikh0.append(np.sqrt((df_classical_tikh0[dropped_nodes[l]]-df_ref_classical[dropped_nodes[l]])**2))
    RMSE_DF_tikh0=np.asarray(RMSE_DF_tikh0)
    RMSE_DF_tikh0_list.append(RMSE_DF_tikh0)

    # Compute RMSE for DFs in x_ref nodes with Tikh0 (BS).
    RMSE_DF_BS_tikh0=[]
    for m in range(0,len(dropped_nodes)):
        RMSE_DF_BS_tikh0.append(np.sqrt((df_BS_tikh0[dropped_nodes[m]]-df_ref_BS[dropped_nodes[m]])**2))
    RMSE_DF_BS_tikh0=np.asarray(RMSE_DF_BS_tikh0)
    RMSE_DF_BS_tikh0_list.append(RMSE_DF_BS_tikh0)
    
    # Compute RMSE for DFs in x_ref dropped nodes (estimation via DF Peak).
    RMSE_DF_cons1=[]
    for i in range(0,len(dropped_nodes)):
        RMSE_DF_cons1.append(np.sqrt((df_classical_cons1[dropped_nodes[i]]-df_ref_classical[dropped_nodes[i]])**2))
    RMSE_DF_cons1=np.asarray(RMSE_DF_cons1)
    RMSE_DF_cons1_list.append(RMSE_DF_cons1)
    
    # Compute RMSE for DFs in x_ref dropped nodes (BS).
    RMSE_DF_BS_cons1=[]
    for j in range(0,len(dropped_nodes)):
        RMSE_DF_BS_cons1.append(np.sqrt((df_BS_cons1[dropped_nodes[j]]-df_ref_BS[dropped_nodes[j]])**2))
    RMSE_DF_BS_cons1=np.asarray(RMSE_DF_BS_cons1)
    RMSE_DF_BS_cons1_list.append(RMSE_DF_BS_cons1)
    
for iteration in range(0,len(RMSE_DF_BS_cons1_list)):
    print('mRMSE Cons1 = %.3e , mRMSE Tikh0 = %.3e' % (np.mean(RMSE_DF_BS_cons1_list[iteration]),np.mean(RMSE_DF_BS_tikh0_list[iteration])))
    
"""