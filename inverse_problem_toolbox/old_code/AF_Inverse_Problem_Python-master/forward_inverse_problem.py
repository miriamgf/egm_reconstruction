# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 14:17:08 2018

@author: Miguel Ángel
"""

import numpy as np
from scipy import signal as sigproc
import time

def forward_problem(EGMs,MTransfer):
    """
    Calculate ECGI forward problem from atrial EGMs
 
    Parameters:
        EGMs (array): atrial electrograms.
        MTransfer (matrix): transfer matrix.
    Returns:
        ECG (array): ECG reconstruction (torso).
    """
    ECG=np.matmul(MTransfer,EGMs)
    return ECG
        
def classical_tikhonov(A,AA,L,LL,y,n_iterations=50):
    """
    Tikhonov global method reconstruction.
    The analytical solution of the inverse problem in terms of Tikhonov regularization is:
    phi_x_hat = inv(A'*A + lambda*L'*L)*A'*phi_T    
 
    Parameters:
        A (matrix): transfer matrix.
        AA (matrix): transfer matrix.
        L (matrix): regularization matrix
        LL (matrix): L'*L
        y (matrix): Body surface potentials (BSP)
        n_iterations (int): number of iterations for computing optimization parameter
    Returns:
        x_hat (matrix): epicardial potentials reconstruction.
        lambda_opt: regularization parameter.
    """
    tic = time.time()
    lambda_test=np.logspace(-1,-9,4)
    
    for index in range(0, n_iterations-1):
        print('Classical Tikhonov method. Iteration %d/%d' % (index+1,n_iterations))
        
        magnitude_term=np.zeros(np.size(lambda_test))
        error_term=np.zeros(np.size(lambda_test))
        
        for i in range(0, np.size(lambda_test)):
            #inv_A = np.linalg.inv(AA+lambda_test[i]*LL)
            #inv_A_At= np.matmul(inv_A,np.transpose(A))
            #x_hat = np.matmul(inv_A_At,y);
            x_hat = np.matmul(np.matmul(np.linalg.inv(AA+lambda_test[i]*LL),np.transpose(A)),y);
            error_term[i]    = np.power(np.linalg.norm(np.matmul(A,x_hat)-y,'fro'),2);
            magnitude_term[i]= np.power(np.linalg.norm(np.matmul(L,x_hat),'fro'),2);
        
        x_term=np.log10(error_term)
        z_term=np.log10(magnitude_term)
        
        dx=np.gradient(x_term,2)
        dz=np.gradient(z_term,2)
        ddx=np.gradient(dx,2)
        ddz=np.gradient(dz,2)
        curve_term1=np.multiply(dx,ddz)-np.multiply(ddx,dz)
        curve_term2=np.power(np.power(dx,2)+np.power(dz,2),3/2)
        curve=np.divide(curve_term1,curve_term2)
        
        abscurve=np.abs(curve)
        maxcurve_index=np.argmax(abscurve)
        lambda_opt=lambda_test[maxcurve_index]
        
        I=maxcurve_index
        if I==0:
            lambda_inf=np.log10(lambda_test[I]);
            lambda_sup=np.log10(lambda_test[I+1]);
        elif I+1>=np.size(lambda_test):
            lambda_inf=np.log10(lambda_test[I-1]);
            lambda_sup=np.log10(lambda_test[I]);
        else:
            lambda_inf=np.log10(lambda_test[I-1]);
            lambda_sup=np.log10(lambda_test[I+1]);
        
        lambda_test=np.logspace(lambda_inf,lambda_sup,4)
    
        if lambda_test[0]-lambda_test[1]<=1e-12:
            break;
            
    #inv_A = np.linalg.inv(AA+lambda_opt*LL)
    #inv_A_At= np.matmul(inv_A,np.transpose(A))
    #x_hat = np.matmul(inv_A_At,y);
    x_hat = np.matmul(np.matmul(np.linalg.inv(AA+lambda_opt*LL),np.transpose(A)),y);
    
    toc = time.time()
    print('Classical Tikhonov. Elapsed time is',round(toc-tic,2), 'seconds')
    
    return x_hat,lambda_opt

def constrained_tikhonov(A,AA,L,LL,D,y,x_ref,n_iterations=50):
    """
    Constrained Tikhonov global method reconstruction.
    The analytical solution of the inverse problem in terms of Tikhonov regularization is:
    phi_x_hat = inv(A'*A + lambda*L'*L)*A'*phi_T    
 
    Parameters:
        A (matrix): transfer matrix.
        AA (matrix): transfer matrix.
        L (matrix): regularization matrix
        LL (matrix): L'*L
        D (matrix): diagonal matrix indicating known nodes
        y (matrix): Body surface potentials (BSP)
        x_ref (matrix): intracavitary signals
        n_iterations (int): number of iterations for computing optimization parameter
    Returns:
        x_hat (matrix): epicardial potentials reconstruction.
        lambda_opt: regularization parameters.
    """
    tic = time.time()

    lambda_test_1=np.logspace(-1,-9,4)
    lambda_test_2=np.logspace(-1,-9,4)
    lambda_opt=np.zeros(2)

    for index in range(0, n_iterations-1):
        print('Constrained Tikhonov method. Iteration %d/%d' % (index+1,n_iterations))
        
        magnitude_term=np.zeros((np.size(lambda_test_1),np.size(lambda_test_1)))
        error_term_1=np.zeros((np.size(lambda_test_1),np.size(lambda_test_1)))
        error_term_2=np.zeros((np.size(lambda_test_2),np.size(lambda_test_2)))

        for i in range(0, np.size(lambda_test_1)):
            for j in range(0, np.size(lambda_test_2)):
                inv_term = np.linalg.inv(AA+lambda_test_1[i]*LL+lambda_test_2[j]*np.matmul(np.transpose(D),D))
                sec_term = np.matmul(np.transpose(A),y)+lambda_test_2[j]*np.matmul(np.transpose(D),x_ref)
                x_hat = np.matmul(inv_term,sec_term);
                error_term_1[i,j] = np.power(np.linalg.norm(np.matmul(A,x_hat)-y,'fro'),2);
                error_term_2[i,j] = np.power(np.linalg.norm(np.matmul(D,x_hat)-x_ref,'fro'),2);
                magnitude_term[i,j]= np.power(np.linalg.norm(np.matmul(L,x_hat),'fro'),2);
        
        x_term=np.log10(error_term_1)
        y_term=np.log10(error_term_2)
        z_term=np.log10(magnitude_term)
        
        dxx,dxy=np.gradient(x_term,2)
        dx=np.sqrt(np.power(dxx,2)+np.power(dxy,2))
        dyx,dyy=np.gradient(y_term,2)
        dy=np.sqrt(np.power(dyx,2)+np.power(dyy,2))
        dzx,dzy=np.gradient(z_term,2)
        dz=np.sqrt(np.power(dzx,2)+np.power(dzy,2))
        
        ddxx,ddxy=np.gradient(dx,2)
        ddx=np.sqrt(np.power(ddxx,2)+np.power(ddxy,2))
        ddyx,ddyy=np.gradient(dy,2)
        ddy=np.sqrt(np.power(ddyx,2)+np.power(ddyy,2))
        ddzx,ddzy=np.gradient(dz,2)
        ddz=np.sqrt(np.power(ddzx,2)+np.power(ddzy,2))
        
        curve_term1=np.sqrt(np.power(np.multiply(ddz,dy)-np.multiply(ddy,dz),2)+np.power(np.multiply(ddx,dz)-np.multiply(ddz,dx),2)+np.power(np.multiply(ddy,dx)-np.multiply(ddx,dy),2))
        curve_term2=np.power(np.power(dx,2)+np.power(dy,2)+np.power(dz,2),3/2)
        curve=np.divide(curve_term1,curve_term2)
        
        abscurve=np.abs(curve)
        maxcurve_index=np.unravel_index(np.argmax(abscurve),abscurve.shape)
        lambda_opt[0]=lambda_test_1[maxcurve_index[0]]
        lambda_opt[1]=lambda_test_2[maxcurve_index[1]]

        I=maxcurve_index[0]
        if I==0:
            lambda1_inf=np.log10(lambda_test_1[I]);
            lambda1_sup=np.log10(lambda_test_1[I+1]);
        elif I+1>=np.size(lambda_test_1):
            lambda1_inf=np.log10(lambda_test_1[I-1]);
            lambda1_sup=np.log10(lambda_test_1[I]);
        else:
            lambda1_inf=np.log10(lambda_test_1[I-1]);
            lambda1_sup=np.log10(lambda_test_1[I+1]);
        
        J=maxcurve_index[1]
        if J==0:
            lambda2_inf=np.log10(lambda_test_2[J]);
            lambda2_sup=np.log10(lambda_test_2[J+1]);
        elif J+1>=np.size(lambda_test_2):
            lambda2_inf=np.log10(lambda_test_2[J-1]);
            lambda2_sup=np.log10(lambda_test_2[J]);
        else:
            lambda2_inf=np.log10(lambda_test_2[J-1]);
            lambda2_sup=np.log10(lambda_test_2[J+1]);
            
        lambda_test_1=np.logspace(lambda1_inf,lambda1_sup,4)
        lambda_test_2=np.logspace(lambda2_inf,lambda2_sup,4)

        if lambda_test_1[0]-lambda_test_1[1]<=1e-12 and lambda_test_2[0]-lambda_test_2[1]<=1e-12:
            break;
            
    inv_term = np.linalg.inv(AA+lambda_opt[0]*LL+lambda_opt[1]*np.matmul(np.transpose(D),D))
    sec_term = np.matmul(np.transpose(A),y)+lambda_opt[1]*np.matmul(np.transpose(D),x_ref)
    x_hat = np.matmul(inv_term,sec_term);
    
    toc = time.time()
    print('Constrained Tikhonov. Elapsed time is',round(toc-tic,2), 'seconds')
    
    return x_hat,lambda_opt