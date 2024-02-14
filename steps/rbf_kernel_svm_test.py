import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import ML_lib.svm as svm
import ML_lib.utils as utils
import ML_lib.preprocessing as pre

DTR_base = np.load('data_npy\\DTR_language.npy')
DTE_base = np.load('data_npy\\DTE_language.npy')
LTR = np.load('data_npy\\LTR_language.npy')
LTE = np.load('data_npy\\LTE_language.npy')


N_CLASS = 2
M_BASE, NTR = DTR_base.shape

#DTR_base, LTR, DTE_base, LTE = utils.shuffle_and_divide(D_BASE, L, 2/3)

K = 0
prior = None
exp_range = range(-3, 2)
lambda_range = [10 ** C_exp for C_exp in exp_range]

plt.figure()

for lam in lambda_range:
    
    res_05 = []

    exp_range = range(-4, 3)
    C_range = [10 ** C_exp for C_exp in exp_range]

    for C in C_range:

        print("Raw")
        print((lam, C))

        scores = svm.SupportVectorMachine(DTR_base, LTR, K, C, svm.get_kernel_RBF(lam, K**2), None if prior is None else [1 - prior, prior]).getScores(DTE_base)
        _, _, minDCF, _, _ = utils.get_metrics(scores, LTE, 0.2, 1, 1, ret_all= True)
        print(f'minDCF(0.2): {minDCF}')
        _, _, minDCF, _, _ = utils.get_metrics(scores, LTE, 0.5, 1, 1, ret_all= True)
        print(f'minDCF(0.5): {minDCF}')
        res_05.append(minDCF)
        _, _, minDCF, _, _ = utils.get_metrics(scores, LTE, 0.8, 1, 1, ret_all= True)
        print(f'minDCF(0.8): {minDCF}')
        
    plt.plot(C_range, res_05, label= f'lam: {lam}')

plt.title(f'Raw - prior: {prior}')
plt.xscale('log')
plt.xticks(C_range, [f'10^{ex}' for ex in exp_range])
plt.legend()
plt.grid()
plt.show()

######
    
gauss = pre.Gaussianizer(DTR_base, LTR)
DTR = gauss.transform(DTR_base)
DTE = gauss.transform(DTE_base)

pca = pre.PCA(DTR)
DTR = pca.transform(DTR, 5)
DTE = pca.transform(DTE, 5)

plt.figure()

for lam in lambda_range:
    
    res_02 = []
    res_05 = []
    res_08 = []

    exp_range = range(-4, 3)
    C_range = [10 ** C_exp for C_exp in exp_range]

    for C in C_range:

        print((lam, C))

        scores = svm.SupportVectorMachine(DTR, LTR, K, C, svm.get_kernel_RBF(lam, K**2), None if prior is None else [1 - prior, prior]).getScores(DTE)
        _, _, minDCF, _, _ = utils.get_metrics(scores, LTE, 0.5, 1, 1, ret_all= True)
        res_05.append(minDCF)
        
    plt.plot(C_range, res_05, label= f'lam: {lam}')

plt.title(f'Gaussianized PCA 5 - prior: {prior}')
plt.xscale('log')
plt.xticks(C_range, [f'10^{ex}' for ex in exp_range])
plt.legend()
plt.grid()
plt.show()

######