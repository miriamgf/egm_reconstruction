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

# Initial parameters
fs = 500
model = 'RAA'
SNR = 20
order = 1
constrained_basket = 'basket'

# Model EGMs, transfer matrix and geometrical model loading. 
x = dl.load_egms(model)
A = dl.load_transfer()
atrial_model, torso_model = dl.load_geometry()

# EGMs filtering.
x = filtering.remove_mean(x)
x = filtering.ECG_filtering(x, fs, model=model) #rows: nodes, columns: time

# DF and phase of EGMs
DF_real, sig_k_real, phase_real = freq_pha.kuklik_DF_phase(x,fs)

# Forward problem, noise adition to BSPs and BSPs filtering.
y = fip.forward_problem(x,A)
y,Cn = filtering.addwhitenoise(y, SNR=SNR)
y = filtering.ECG_filtering(y, fs, model=model)

#  Computing of inverse problem required matrix (AA, L, LL, D).
AA,L,LL = pre_m.precompute_matrix(A, atrial_model, order)
D,known_nodes = pre_m.D_nodes_constrained(n_nodes=constrained_basket)

# Classical Tikhonov-based inverse problem approach.
x_hat_tikh, lambda_opt_tikh = fip.classical_tikhonov(A, AA, L, LL, y)
DF_tikh, sig_k_tikh, phase_tikh = freq_pha.kuklik_DF_phase(x_hat_tikh,fs)

# Classical Tikhonov-based inverse problem metrics
RDMSt_tikh, mRDMSt_tikh, stdRDMSt_tikh = metrics.RDMS_calc(x,x_hat_tikh)
CCt_tikh, mCCt_tikh, stdCCt_tikh = metrics.CC_calc(x,x_hat_tikh)

DDF_tikh, mDDF_tikh, stdDDF_tikh = metrics.DFmetrics_calc(DF_real, DF_tikh)
RDMSt_tikh_phase, mRDMSt_tikh_phase, stdRDMSt_tikh_phase = metrics.RDMS_calc(phase_real,phase_tikh)
CCt_tikh_phase, mCCt_tikh_phase, stdCCt_tikh_phase = metrics.CC_calc(phase_real,phase_tikh)

# Constrained Tikhonov-based inverse problem approach.
x_hat_cons, lambda_opt_cons = fip.constrained_tikhonov(A, AA, L, LL, D, y, x)
DF_cons, sig_k_cons, phase_cons = freq_pha.kuklik_DF_phase(x_hat_cons,fs)

# Constrained Tikhonov-based inverse problem metrics
RDMSt_cons, mRDMSt_cons, stdRDMSt_cons = metrics.RDMS_calc(x,x_hat_cons)
CCt_cons, mCCt_cons, stdCCt_cons = metrics.CC_calc(x,x_hat_cons)

DDF_cons, mDDF_cons, stdDDF_cons = metrics.DFmetrics_calc(DF_real, DF_cons)
RDMSt_cons_phase, mRDMSt_cons_phase, stdRDMSt_cons_phase = metrics.RDMS_calc(phase_real,phase_cons)
CCt_cons_phase, mCCt_cons_phase, stdCCt_cons_phase = metrics.CC_calc(phase_real,phase_cons)