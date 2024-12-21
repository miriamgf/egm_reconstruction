# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 10:28:18 2018

@author: Miguel Ángel
"""

import forward_inverse_problem as fip
import filtering
import data_load as dl
import precompute_matrix as pre_m
#import metrics
import freq_phase_analysis as freq_pha
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D  
import numpy as np

# %% Initial parameters
fs = 2034.5
model = 'Valencia_Pat_110114'
constrained_basket = 'basket'

# %% Load EGMs, ECGs, transfer matrix and geometrical model. 

x,y = dl.load_egms_real(model)
A = dl.load_transfer_real(model)
atrial_model,_ = dl.load_geometry_real(model)

# %% EGMs and ECGs filtering.

x,_ = filtering.detrendSpline(x,fs,l_w = 0.25)
y,_ = filtering.detrendSpline(y,fs,l_w = 0.25)
x_filtered = filtering.ECG_filtering_real(x, fs)
y_filtered = filtering.ECG_filtering_real(y, fs)
del(x,y)

# %%  Computing of inverse problem required matrix (AA, L, LL, D).

AA,L_0,LL_0 = pre_m.precompute_matrix_real(A, atrial_model, model, 0)
_,L_1,LL_1 = pre_m.precompute_matrix_real(A, atrial_model, model, 1)
#AA_2,L_2,LL_2 = pre_m.precompute_matrix_real(A, atrial_model, model, 2)
del(atrial_model)

# %% Analysis for reference signals

# Generate D matrix and x sorted matrix with each signal in the correspondent node (row).
D,known_nodes_full, x_ref_cons_full = pre_m.D_nodes_constrained_real(model, x_filtered, A.shape[1])

# DF Analysis Ref signals
f_ref, pxx_ref, df_ref, _ = freq_pha.df_estimation_peak(x_ref_cons_full,fs)

# %% Analysis for Tikh0

# Inverse problem (Tikh0)
#x_hat_tikh0, lambda_opt_tikh0 = fip.classical_tikhonov(A, AA_0, L_0, LL_0, y_filtered)
#x_hat_tikh0, lambda_opt_tikh0, magnitude_term_tikh0, error_term_tikh0, maxcurve_index_tikh0 = fip.classical_tikhonov_noiter_global(A, AA_0, L_0, LL_0, y_filtered)

# Pat1: lambda_opt=0.00013125330147352273
lambda_opt_tikh0 = 0.00013125330147352273
x_hat_tikh0 = np.matmul(np.matmul(np.linalg.inv(AA+lambda_opt_tikh0*LL_0),np.transpose(A)),y_filtered);

# DF Analysis Tikh0
f_tikh0, pxx_tikh0, df_tikh0, _ = freq_pha.df_estimation_peak(x_hat_tikh0,fs)

# RMSE DF Tikh0 (All known nodes)
RMSE_DF_tikh0_full = np.sqrt((df_tikh0[known_nodes_full]-df_ref[known_nodes_full])**2)
mean_RMSE_DF_tikh0_full = np.mean(RMSE_DF_tikh0_full)
std_RMSE_DF_tikh0_full = np.std(RMSE_DF_tikh0_full)

"""
# %% Analysis for Cons1 (All known nodes)

# Inverse problem (Cons1)
#x_hat_cons1,lambda_opt_cons1 = fip.constrained_tikhonov(A, AA_1, L_1, LL_1, D, y_filtered, x_ref_cons_full)
#x_hat_cons1,lambda_opt_cons1,error_term_1_cons1,error_term_2_cons1,magnitude_term_cons1,maxcurve_index_cons1 = fip.constrained_tikhonov_noiter(A, AA_1, L_1, LL_1, D, y_filtered, x_ref_cons_full)

inv_term = np.linalg.inv(AA_1+1e-4*LL_1+1*np.matmul(np.transpose(D),D))
sec_term = np.matmul(np.transpose(A),y_filtered)+1*np.matmul(np.transpose(D),x_ref_cons_full)
x_hat_cons1 = np.matmul(inv_term,sec_term);

# DF Analysis Cons1
f_cons1, pxx_cons1, df_cons1, signals_cons1_df = freq_pha.df_estimation_peak(x_hat_cons1,fs)

# RMSE DF Cons1 (All known nodes)
RMSE_DF_cons1 = np.sqrt((df_cons1[known_nodes_full]-df_ref[known_nodes_full])**2)
mean_RMSE_DF_cons1 = np.mean(RMSE_DF_cons1)
std_RMSE_DF_cons1 = np.std(RMSE_DF_cons1)
"""

""" 
# %% Analysis for lambda analysis Cons1 
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
        x_hat = x_hat[1534,100:-500]/np.max(x_hat[1534,100:-500]) #Normalize and remove start and end segments.
        x_gt = x_ref_cons_full[1534,100:-500]/np.max(x_ref_cons_full[1534,100:-500])
        t=np.linspace(0,x_gt.shape[0]/fs,x_gt.shape[0])
        plt.figure(),
        plt.plot(t,x_gt,'k'),plt.plot(t,x_hat,'r')
        plt.xlim((0,2)),plt.xlabel('Time (s)'),plt.ylabel('Amplitude (Normalized)')
        plt.legend(('GT','Cons1')),plt.title(r'Cons1, node 1534. $\lambda_1$=%.3e, $\lambda_2$=%.3e'%(i,j))
"""

# %% Analysis for Cons1 - Dropout mode

known_nodes_dropout_list=[]
#x_hat_cons1_list=[]
#lambda_opt_cons1_list=[]
dropped_nodes_list=[]
pxx_cons1_dropout_list = []
df_cons1_dropout_list=[]
RMSE_DF_tikh0_dropout_list=[]
#mRMSE_DF_tikh0_dropout_list=[]
RMSE_DF_cons1_dropout_list = []
#mRMSE_DF_cons1_dropout_list = []

n_iterations = 600
for k in range(n_iterations):
    print('Constrained Tikhonov dropout mode. Iteration n. %d' % (k+1))
    
    # Compute known nodes and dropped nodes
    D,known_nodes,x_ref_cons, dropped_nodes = pre_m.D_nodes_constrained_real_dropout(model, x_filtered, A.shape[1],prob=1/62)
    known_nodes_dropout_list.append(known_nodes)
    dropped_nodes_list.append(dropped_nodes)
    
    # Compute inverse problem
    #x_hat_cons1,lambda_opt_list_cons1 = fip.constrained_tikhonov(A, AA_1, L_1, LL_1, D, y, x_ref_cons)
    #lambda_opt_cons1_list.append(lambda_opt_list_cons1)    
    inv_term = np.linalg.inv(AA+1e-4*LL_1+1*np.matmul(np.transpose(D),D))
    sec_term = np.matmul(np.transpose(A),y_filtered)+1*np.matmul(np.transpose(D),x_ref_cons)
    x_hat_cons1 = np.matmul(inv_term,sec_term);
    #x_hat_cons1_list.append(x_hat_cons1)

    # DF Analysis Cons1 (Classical)
    _, _, df_cons_dropout,_ = freq_pha.df_estimation_peak(x_hat_cons1,fs)
    #pxx_cons1_dropout_list.append(pxx_cons_dropout)
    df_cons1_dropout_list.append(df_cons_dropout)
    
    # Compute RMSE for DFs in x_ref dropped nodes (estimation via DF Peak).
    RMSE_DF_tikh0_dropout_list.append(np.sqrt((df_tikh0[dropped_nodes]-df_ref[dropped_nodes])**2))
    RMSE_DF_cons1_dropout_list.append(np.sqrt((df_cons_dropout[dropped_nodes]-df_ref[dropped_nodes])**2))
    
    #mRMSE_DF_tikh0_dropout_list.append(np.mean(np.sqrt((df_tikh0[dropped_nodes]-df_ref[dropped_nodes])**2)))
    #mRMSE_DF_cons1_dropout_list.append(np.mean(np.sqrt((df_cons_dropout[dropped_nodes]-df_ref[dropped_nodes])**2)))
    
    # Clean memory
    del(D,known_nodes,x_ref_cons,dropped_nodes,inv_term,sec_term,x_hat_cons1,df_cons_dropout)

# Clean memory
del(A,AA,LL_0,LL_1,L_0,L_1)


## Check number of nodes dropped on each iteration
number_drop_nodes = np.zeros(len(dropped_nodes_list))
for i in range(0,len(dropped_nodes_list)):
    number_drop_nodes[i]=dropped_nodes_list[i].shape[0]
plt.hist(number_drop_nodes)
plt.title('Number of nodes used in Cons-Tikh simulations')
plt.ylabel('Frequency')
plt.xlabel('Number of nodes')
# Dropped nodes rate: 0.3. We only take in account those iterations that dropped,
# at least, mean +- std nodes.
mean_drop_nodes = int(np.round(np.mean(number_drop_nodes)))
std_drop_nodes = int(np.round(np.std(number_drop_nodes)))
low_lim = mean_drop_nodes-std_drop_nodes
high_lim = mean_drop_nodes+std_drop_nodes
del(number_drop_nodes)

## Re-build RMSE DF data. Only consider iterations whose number of dropped_nodes is between low_lim and high_lim
RMSE_DF_tikh0_dropout_list_rebuilt=[]
mRMSE_DF_tikh0_dropout_list_rebuilt=[]
RMSE_DF_cons1_dropout_list_rebuilt = []
mRMSE_DF_cons1_dropout_list_rebuilt = []

diff_mRMSE_DF_Cons1_Tikh0_rebuilt = []
diff_mRMSE_DF_counter_rebuilt = 0
total_it_considered = 0

for iteration in range(0,len(RMSE_DF_cons1_dropout_list)):
    if len(dropped_nodes_list[iteration])<=high_lim and len(dropped_nodes_list[iteration])>=low_lim:
        total_it_considered+=1
        RMSE_DF_tikh0_dropout_list_rebuilt.append(RMSE_DF_tikh0_dropout_list[iteration])
        mean_RMSE_tikh0 = np.mean(RMSE_DF_tikh0_dropout_list[iteration])
        mRMSE_DF_tikh0_dropout_list_rebuilt.append(mean_RMSE_tikh0)
        
        RMSE_DF_cons1_dropout_list_rebuilt.append(RMSE_DF_cons1_dropout_list[iteration])
        mean_RMSE_cons1 = np.mean(RMSE_DF_cons1_dropout_list[iteration])
        mRMSE_DF_cons1_dropout_list_rebuilt.append(mean_RMSE_cons1)
        
        #print('mRMSE Cons1 = %.3f , mRMSE Tikh0 = %.3f' % (mRMSE_DF_cons1_dropout_list[iteration],mRMSE_DF_tikh0_dropout_list[iteration]))
        if mean_RMSE_cons1< mean_RMSE_tikh0:
            diff_mRMSE_DF_Cons1_Tikh0_rebuilt.append(True)
            diff_mRMSE_DF_counter_rebuilt+=1
        else:
            diff_mRMSE_DF_Cons1_Tikh0_rebuilt.append(False)

# Clean memory        
del(RMSE_DF_tikh0_dropout_list,RMSE_DF_cons1_dropout_list,mean_RMSE_tikh0,mean_RMSE_cons1)

# Print mean results
mRMSE_tikh0_dropout = np.mean(mRMSE_DF_tikh0_dropout_list_rebuilt)
mRMSE_cons1_dropout = np.mean(mRMSE_DF_cons1_dropout_list_rebuilt)

print('Total iterations considered: %d/%d' %(total_it_considered,n_iterations))
print('mRMSE Cons1 = %.3f Hz, mRMSE Tikh0 = %.3f Hz' % (mRMSE_cons1_dropout,mRMSE_tikh0_dropout))
print('Number of iterations where mRMSE(Cons1) < (mRMSE Tikh0): %d/%d' % (diff_mRMSE_DF_counter_rebuilt,total_it_considered))




"""
# Compute RMSE for each node in each iteration (both Tikh0 and Cons1)
RMSE_DF_cons1_nodebynode = {}

for full_nodes_index in range(0,known_nodes_full.shape[0]):
    node=known_nodes_full[full_nodes_index]
    RMSE_cons_1_list_node = []

    for iteration in range(0,len(known_nodes_dropout_list)):
        if dropped_nodes_list[iteration].shape[0] <= high_lim and dropped_nodes_list[iteration].shape[0] >= low_lim and node in dropped_nodes_list[iteration]:
            index_dropped_node = np.where(dropped_nodes_list[iteration] == node)[0][0]
            RMSE_cons_1_list_node.append(RMSE_DF_cons1_dropout_list[iteration][index_dropped_node])
    RMSE_DF_cons1_nodebynode[node] = RMSE_cons_1_list_node
 
# Analysis of number of times that each node is not used in Cons-Tikh
number_times = np.zeros(len(RMSE_DF_cons1_nodebynode))
for node in range(0,known_nodes_full.shape[0]):
    number_times[node]=len(RMSE_DF_cons1_nodebynode[known_nodes_full[node]])
    
plt.hist(number_times)
plt.title('Number of times that each node was not used in Cons-Tikh')
plt.ylabel('Frequency')
plt.xlabel('Number of times')
    
# Analysis of RMSE DF
RMSE_DF_cons1_analysis_nodebynode = {}
RMSE_DF_comparison_nodebynode = {} # If Cons1 gives a better mean result in a node, it will be marked as True. If not, it will be marked as "False".
true_counter=0
for full_nodes_index in range(0,known_nodes_full.shape[0]):
    node=known_nodes_full[full_nodes_index]
    mean_cons1_RMSE = np.mean(RMSE_DF_cons1_nodebynode[node])
    std_cons1_RMSE = np.std(RMSE_DF_cons1_nodebynode[node])
    RMSE_DF_cons1_analysis_nodebynode[node] = (mean_cons1_RMSE,std_cons1_RMSE)
    if mean_cons1_RMSE < RMSE_DF_tikh0[full_nodes_index]:
        RMSE_DF_comparison_nodebynode[node] = True
        true_counter+=1
    else:
        RMSE_DF_comparison_nodebynode[node] = False
"""  
