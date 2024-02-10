import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import ML_lib.gaussian as gau
import ML_lib.utils as utils
import ML_lib.preprocessing as pre

D_BASE = np.load('data_npy\\DTR_language.npy')
L = np.load('data_npy\\LTR_language.npy')

N_CLASS = 2
M_BASE, NTR = D_BASE.shape

utils.cross_validation(D_BASE, L, 100, gau.GaussianClassifier, [
    [],
    ['naive'],
    ['tied'],
    ['naive-tied']
    ], progress= True, effective= [0.2, 0.5, 0.8], save= True, filename= 'results\\cross_val_gau.txt', prepro= [

    [(pre.NoTransform, [])],
    [(pre.Standardizer, [])],
    [(pre.Gaussianizer, [])],

    [(pre.Standardizer, []), (pre.PCA, [5])],
    [(pre.Standardizer, []), (pre.PCA, [4])],
    [(pre.Standardizer, []), (pre.PCA, [3])],

    [(pre.Gaussianizer, []), (pre.PCA, [5])],
    [(pre.Gaussianizer, []), (pre.PCA, [4])],
    [(pre.Gaussianizer, []), (pre.PCA, [3])],
    [(pre.LDA, [1])]
    ], print_err= True)