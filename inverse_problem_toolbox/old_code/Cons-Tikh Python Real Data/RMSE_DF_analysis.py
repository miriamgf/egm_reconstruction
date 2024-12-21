# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 11:45:36 2020

@author: Miguel Ángel
"""
import precompute_matrix as pre_m
import numpy as np
import matplotlib.pyplot as plt

_,known_nodes_full,_ = pre_m.D_nodes_constrained_real(model, x, A.shape[1])

## Check number of nodes dropped in each iteration
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

RMSE_DF_cons1_nodebynode = {}
RMSE_DF_tikh0_nodebynode = {}

# Compute RMSE for each node in each iteration (both Tikh0 and Cons1)
for full_nodes_index in range(0,known_nodes_full.shape[1]):
    node=known_nodes_full[0,full_nodes_index]
    RMSE_cons_1_list_node = []
    RMSE_tikh_0_list_node = []

    for iteration in range(0,len(known_nodes_list)):
        if dropped_nodes_list[iteration].shape[0] <= high_lim and dropped_nodes_list[iteration].shape[0] >= low_lim and node in dropped_nodes_list[iteration]:
            index_dropped_node = np.where(dropped_nodes_list[iteration] == node)[0][0]
            RMSE_cons_1_list_node.append(RMSE_DF_cons1_list[iteration][index_dropped_node])
            RMSE_tikh_0_list_node.append(RMSE_DF_tikh0_list[iteration][index_dropped_node])
    RMSE_DF_cons1_nodebynode[node] = RMSE_cons_1_list_node
    # Values for Tikh0 must be always the same
    RMSE_DF_tikh0_nodebynode[node] = np.unique(RMSE_tikh_0_list_node)[0]
 
# Analysis of number of times that each node is not used in Cons-Tikh
number_times = np.zeros(len(RMSE_DF_cons1_nodebynode))
for node in range(0,known_nodes_full.shape[1]):
    number_times[node]=len(RMSE_DF_cons1_nodebynode[known_nodes_full[0,node]])
    
plt.hist(number_times)
plt.title('Number of times that each node was not used in Cons-Tikh')
plt.ylabel('Frequency')
plt.xlabel('Number of times')
    
# Analysis of RMSE DF
RMSE_DF_cons1_analysis_nodebynode = {}
RMSE_DF_comparison_nodebynode = {} # If Cons1 gives a better mean result in a node, it will be marked as True. If not, it will be marked as "False".
true_counter=0
for full_nodes_index in range(0,known_nodes_full.shape[1]):
    node=known_nodes_full[0,full_nodes_index]
    mean_cons1_RMSE = np.mean(RMSE_DF_cons1_nodebynode[node])
    std_cons1_RMSE = np.std(RMSE_DF_cons1_nodebynode[node])
    RMSE_DF_cons1_analysis_nodebynode[node] = (mean_cons1_RMSE,std_cons1_RMSE)
    if mean_cons1_RMSE < RMSE_DF_tikh0_nodebynode[node]:
        RMSE_DF_comparison_nodebynode[node] = True
        true_counter+=1
    else:
        RMSE_DF_comparison_nodebynode[node] = False
        
        
# Graphical analysis of DF detection       
t=np.linspace(0,x.shape[1]/fs,x.shape[1])
for node in range(0,x.shape[0]):
    plt.figure(figsize=(13,10))
    
    # Plot signals (normalized)
    plt.subplot(211), 
    plt.title('Signals. Node: %d' % (known_nodes_full[0,node]))
    plt.plot(t,x_ref_cons_full[known_nodes_full[0,node],:]/np.max(x_ref_cons_full[known_nodes_full[0,node],:]),'k'), #Ground Truth
    plt.plot(t,x_hat_tikh0[known_nodes_full[0,node],:]/np.max(x_hat_tikh0[known_nodes_full[0,node],:]),'b'), #Tikh0
    plt.plot(t,x_hat_cons1[known_nodes_full[0,node],:]/np.max(x_hat_cons1[known_nodes_full[0,node],:]),'r'), #Cons1
    plt.legend(('GT','Tikh0','Cons1'))
    plt.xlim((0,2)),plt.xlabel('Time (s)'),plt.ylabel('Amplitude (Normalized)')
    
    # Plot spectrums
    plt.subplot(212), 
    plt.title('DF Analysis')
    plt.plot(f_classical, pxx_ref_classical[known_nodes_full[0,node]]/np.max(pxx_ref_classical[known_nodes_full[0,node]]),'k'), #Ground Truth
    plt.plot(f_classical, pxx_classical_tikh0[known_nodes_full[0,node]]/np.max(pxx_classical_tikh0[known_nodes_full[0,node]]),'b'), #Tikh0
    plt.plot(f_classical, pxx_classical_cons1[known_nodes_full[0,node]]/np.max(pxx_classical_cons1[known_nodes_full[0,node]]),'r'), #Cons1
    plt.xlim((0,25)), plt.xlabel('Frequency (Hz)'),plt.ylabel('Pxx (Normalized)')
    plt.legend(('GT, DF: %.3f Hz'% (df_ref_classical[known_nodes_full[0,node]]),
                'Tikh0, DF: %.3f Hz'% (df_classical_tikh0[known_nodes_full[0,node]]),
                'Cons1, DF: %.3f Hz'% (df_classical_cons1[known_nodes_full[0,node]])))
    plt.axvline(x=df_ref_classical[known_nodes_full[0,node]],color='k')
    plt.axvline(x=df_classical_tikh0[known_nodes_full[0,node]],color='b')
    plt.axvline(x=df_classical_cons1[known_nodes_full[0,node]],color='r')
    
    # Save figs
    plt.savefig('figs/Sig_DF_node_%d.png'% (known_nodes_full[0,node]))
    
plt.close('all')