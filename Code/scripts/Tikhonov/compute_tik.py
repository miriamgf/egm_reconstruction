import sys
sys.path.append("../Code")
import argparse
import math
import os
import sys
import time

import scripts.Tikhonov.forward_inverse_problem as fip
import numpy as np
import scripts.Tikhonov.precompute_matrix as pre_m

class TikhonovReconstruction:

    def __init__(self,bsp,transfer_matrix, order):

        self.transfer_matrix=transfer_matrix
        self.bsp=bsp
        self.order=order

    def __call__(self):
        
        A = np.array(self.transfer_matrix)  # Only one torso
        AA, L, LL = pre_m.precompute_matrix(A, self.order)
        print('Computing Tikhonov')
        x_hat, lambda_opt, magnitude_term, error_term, maxcurve_index = (
        fip.classical_tikhonov_noiter_global(A, AA, L, LL, self.bsp))
        return x_hat