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

model_params_init = [[K, C, lam] for K in [0., 1e-6, 1e-4, 0.01, 1., 100.] for C in [1e-4, 0.01, 1.] for lam in [0., 1e-6, 1e-4, 0.01, 1., 100.]]
model_params = [[K, C, svm.get_kernel_RBF(lam, K**2)] for K, C, lam in model_params_init]

utils.cross_validation(D_BASE, L, 10, svm.SupportVectorMachine, model_params, progress= True, print_act= False, save= True, filename= 'results\\cross_val_rbf_svm.txt', model_params_print= model_params_init, prepro= [
    [(pre.NoTransform, [])],
    
    [(pre.Standardizer, [])],
    [(pre.Standardizer, []), (pre.PCA, [5])],
    [(pre.Standardizer, []), (pre.PCA, [4])],
    ])