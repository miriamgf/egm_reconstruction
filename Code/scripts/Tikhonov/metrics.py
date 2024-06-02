# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 10:59:02 2018

@author: Miguel Ángel
"""

import numpy as np
from scipy import signal as sigproc
from scipy import stats


def RDMS_calc(reference_sig, estimation_sig):
    """
    Computes the Relative Difference Measurement Star (RDMS)

    Parameters:
        reference_sig (array): Ground Truth
        estimation_sig (array): Estimated signal
    Returns:
        RDMSt (array): RDMS of the signals.
        mRDMSt (float): mean RDMS.
        stdRDMSt (float): standard deviation of RDMSt
    """

    # Norms
    norm_reference = np.linalg.norm(reference_sig, 2, axis=1)
    norm_estimation = np.linalg.norm(estimation_sig, 2, axis=1)

    # Normalize signals
    reference_normalized = np.zeros(np.shape(reference_sig))
    estimation_normalized = np.zeros(np.shape(estimation_sig))

    for i in range(0, np.shape(reference_sig)[0]):
        reference_normalized[i, :] = np.divide(reference_sig[i, :], norm_reference[i])
        estimation_normalized[i, :] = np.divide(
            estimation_sig[i, :], norm_estimation[i]
        )

    # Calculate the norm of the difference of signals
    difference = reference_normalized - estimation_normalized
    RDMSt = np.linalg.norm(difference, 2, axis=1)

    # Mean and std of RDMS
    mRDMSt = np.mean(RDMSt)
    stdRDMSt = np.std(RDMSt)

    return RDMSt, mRDMSt, stdRDMSt


def CC_calc(reference_sig, estimation_sig):
    """
    Computes Pearson's Correlation Coefficient (CC)

    Parameters:
        reference_sig (array): Ground Truth
        estimation_sig (array): Estimated signal
    Returns:
        CCt (array): CC of the signals.
        mCCt (float): mean CC.
        stdCCt (float): standard deviation of CCt
    """

    # Norms
    norm_reference = np.linalg.norm(reference_sig, 2, axis=1)
    norm_estimation = np.linalg.norm(estimation_sig, 2, axis=1)

    # Normalize signals
    reference_normalized = np.zeros(np.shape(reference_sig))
    estimation_normalized = np.zeros(np.shape(estimation_sig))

    CCt = np.zeros(np.shape(reference_sig)[0])
    CCt_p_value = np.zeros(np.shape(reference_sig)[0])

    for i in range(0, np.shape(reference_sig)[0]):
        reference_normalized[i, :] = np.divide(reference_sig[i, :], norm_reference[i])
        estimation_normalized[i, :] = np.divide(
            estimation_sig[i, :], norm_estimation[i]
        )
        CCt[i], CCt_p_value[i] = stats.pearsonr(
            reference_normalized[i, :], estimation_normalized[i, :]
        )

    # Mean and std of CCt
    mCCt = np.mean(CCt)
    stdCCt = np.std(CCt)

    return CCt, mCCt, stdCCt


def DFmetrics_calc(reference_DF, estimated_DF):
    """
    Computes DF difference (DDF)

    Parameters:
        reference_DDF (array): Ground Truth
        estimated_DDF (array): Estimated DF
    Returns:
        DDF (array): difference between estimated DF and Ground Truth.
        mDDF (float): mean DDF.
        stdDDF (float): standard deviation of DDF
    """
    DDF = np.abs(reference_DF - estimated_DF)
    mDDF = np.mean(DDF)
    stdDDF = np.std(DDF)

    return DDF, mDDF, stdDDF
