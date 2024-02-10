import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import ML_lib.svm as svm
import ML_lib.utils as utils
import ML_lib.preprocessing as pre

D_BASE = np.load('data_npy\\DTR_language.npy')
L = np.load('data_npy\\LTR_language.npy')

N_CLASS = 2
M_BASE, NTR = D_BASE.shape

utils.cross_validation(D_BASE, L, 10, svm.SupportVectorMachine, [
    #[K, C, None] for K in [0., 1e-6, 1e-4, 0.01, 1., 100.] for C in [1e-4, 0.01, 1.]
    [K, C, None, [1-prior, prior]] for K in [0., 1e-6, 1e-4, 0.01, 1., 100.] for C in [1e-4, 0.01, 1.] for prior in [0.2, 0.5, 0.8]
    ], progress= True, effective= [0.2, 0.5, 0.8], print_err= True, save= True, filename= 'results\\cross_val_lin_svm.txt', prepro= [
    [(pre.NoTransform, [])],
    [(pre.Standardizer, [])],

    [(pre.Standardizer, []), (pre.PCA, [5])],
    [(pre.Standardizer, []), (pre.PCA, [4])],
    [(pre.Standardizer, []), (pre.PCA, [3])],
    ])