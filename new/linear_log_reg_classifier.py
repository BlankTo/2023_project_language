import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import ML_lib.log_reg as lg
import ML_lib.utils as utils
import ML_lib.preprocessing as pre

D_BASE = np.load('data_npy\\DTR_language.npy')
L = np.load('data_npy\\LTR_language.npy')


print(D_BASE.shape)
print(L.shape)

N_CLASS = 2
M_BASE, NTR = D_BASE.shape

prior = (L == 1).sum() / NTR

model_params = [0] + [10 ** lam_exp for lam_exp in range(-6, 3)]

model_params = []
for prior in np.arange(0.1, 0.5, 0.1):
    model_params += [[0, prior]]
    for lam_exp in range(-6, 3):
        model_params += [[10 ** lam_exp, prior]]

utils.cross_validation(D_BASE, L, 10, lg.LinearRegressionClassifier, model_params, progress= True, print_act= False, save= False, filename= 'results\\cross_val_lin_log_reg.txt', prepro= [
    [(pre.NoTransform, [])],
    [(pre.Standardizer, [])],

    [(pre.Standardizer, []), (pre.PCA, [5])],
    [(pre.Standardizer, []), (pre.PCA, [4])],
    [(pre.Standardizer, []), (pre.PCA, [3])],
    ])