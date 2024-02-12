import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import ML_lib.svm as svm
import ML_lib.utils as utils
import ML_lib.preprocessing as pre

D_BASE = np.load('data_npy\\DTR_language.npy')
L = np.load('data_npy\\LTR_language.npy')

N_CLASS = 2
M_BASE, NTR = D_BASE.shape

DTR_base, LTR, DTE_base, LTE = utils.shuffle_and_divide(D_BASE, L, 2/3)

K = 0
prior = 0.5
exp_range = range(-3, 2)
lambda_range = [10 ** C_exp for C_exp in exp_range]

plt.figure()

for lam in lambda_range:
    
    res_02 = []
    res_05 = []
    res_08 = []

    exp_range = range(-5, 3)
    C_range = [10 ** C_exp for C_exp in exp_range]

    for C in C_range:

        print((lam, C))

        scores = svm.SupportVectorMachine(DTR_base, LTR, K, C, svm.get_kernel_RBF(lam, K**2), None if prior is None else [1 - prior, prior]).getScores(DTE_base)
        _, _, minDCF, _, _ = utils.get_metrics(scores, LTE, 0.5, 1, 1, ret_all= True)
        res_05.append(minDCF)

    plt.plot(C_range, res_05, label= f'lam: {lam}')

plt.title(f'No Prepro - prior: {prior}')
plt.xscale('log')
plt.xticks(C_range, [f'10^{ex}' for ex in exp_range])
plt.legend()
plt.grid()
#plt.show()

######
    
pca = pre.PCA(DTR_base)
DTR = pca.transform(DTR_base, 5)
DTE = pca.transform(DTE_base, 5)

plt.figure()

for lam in lambda_range:
    
    res_02 = []
    res_05 = []
    res_08 = []

    exp_range = range(-5, 3)
    C_range = [10 ** C_exp for C_exp in exp_range]

    for C in C_range:

        print((lam, C))

        scores = svm.SupportVectorMachine(DTR, LTR, K, C, svm.get_kernel_RBF(lam, K**2), None if prior is None else [1 - prior, prior]).getScores(DTE)
        _, _, minDCF, _, _ = utils.get_metrics(scores, LTE, 0.5, 1, 1, ret_all= True)
        res_05.append(minDCF)

    plt.plot(C_range, res_05, label= f'lam: {lam}')

plt.title(f'PCA 5 - prior: {prior}')
plt.xscale('log')
plt.xticks(C_range, [f'10^{ex}' for ex in exp_range])
plt.legend()
plt.grid()
#plt.show()

######
    
gauss = pre.Gaussianizer(DTR_base, LTR)
DTR = gauss.transform(DTR_base)
DTE = gauss.transform(DTE_base)

plt.figure()

for lam in lambda_range:
    
    res_02 = []
    res_05 = []
    res_08 = []

    exp_range = range(-5, 3)
    C_range = [10 ** C_exp for C_exp in exp_range]

    for C in C_range:

        print((lam, C))

        scores = svm.SupportVectorMachine(DTR, LTR, K, C, svm.get_kernel_RBF(lam, K**2), None if prior is None else [1 - prior, prior]).getScores(DTE)
        _, _, minDCF, _, _ = utils.get_metrics(scores, LTE, 0.5, 1, 1, ret_all= True)
        res_05.append(minDCF)
        
    plt.plot(C_range, res_05, label= f'lam: {lam}')

plt.title(f'Gaussianized - prior: {prior}')
plt.xscale('log')
plt.xticks(C_range, [f'10^{ex}' for ex in exp_range])
plt.legend()
plt.grid()
#plt.show()

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

    exp_range = range(-5, 3)
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

#######
    # Cross val
#######

plt.figure()
    
for lam in lambda_range:
    
    res_02 = []
    res_05 = []
    res_08 = []

    exp_range = range(-5, 3)
    C_range = [10 ** C_exp for C_exp in exp_range]

    for C in C_range:

        print((lam, C))

        LTE_cross, _, scores = utils.cross_validation_base(D_BASE, L, svm.SupportVectorMachine, 10, None, None, 0, [K, C, svm.get_kernel_RBF(lam, K**2), None] if prior is None else [K, C, svm.get_kernel_RBF(lam, K**2), [1 - prior, prior]], True)

        _, _, minDCF, _, _ = utils.get_metrics(scores, LTE_cross, 0.5, 1, 1, ret_all= True)
        res_05.append(minDCF)

    plt.plot(C_range, res_05, label= f'lam: {lam}')
    
plt.title(f'Cross Val 10 No Prepro - prior: {prior}')
plt.xscale('log')
plt.xticks(C_range, [f'10^{ex}' for ex in exp_range])
plt.legend()
plt.grid()
#plt.show()

#######

plt.figure()
    
for lam in lambda_range:
    
    res_02 = []
    res_05 = []
    res_08 = []

    exp_range = range(-5, 3)
    C_range = [10 ** C_exp for C_exp in exp_range]

    for C in C_range:

        print((lam, C))

        LTE_cross, _, scores = utils.cross_validation_base(D_BASE, L, svm.SupportVectorMachine, 10, pre.Gaussianizer, [pre.PCA, 5], 0, [K, C, svm.get_kernel_RBF(lam, K**2), None] if prior is None else [K, C, svm.get_kernel_RBF(lam, K**2), [1 - prior, prior]], True)

        _, _, minDCF, _, _ = utils.get_metrics(scores, LTE_cross, 0.5, 1, 1, ret_all= True)
        res_05.append(minDCF)
        
    plt.plot(C_range, res_05, label= f'lam: {lam}')

plt.title(f'Cross Val 10 Gaussianized PCA 5 - prior: {prior}')
plt.xscale('log')
plt.xticks(C_range, [f'10^{ex}' for ex in exp_range])
plt.legend()
plt.grid()
#plt.show()

#######

plt.figure()
    
for lam in lambda_range:
    
    res_02 = []
    res_05 = []
    res_08 = []

    exp_range = range(-5, 3)
    C_range = [10 ** C_exp for C_exp in exp_range]

    for C in C_range:

        print((lam, C))

        LTE_cross, _, scores = utils.cross_validation_base(D_BASE, L, svm.SupportVectorMachine, 10, pre.Gaussianizer, None, 0, [K, C, svm.get_kernel_RBF(lam, K**2), None] if prior is None else [K, C, svm.get_kernel_RBF(lam, K**2), [1 - prior, prior]], True)

        _, _, minDCF, _, _ = utils.get_metrics(scores, LTE_cross, 0.5, 1, 1, ret_all= True)
        res_05.append(minDCF)
        
    plt.plot(C_range, res_05, label= f'lam: {lam}')

plt.title(f'Cross Val 10 Gaussianized - prior: {prior}')
plt.xscale('log')
plt.xticks(C_range, [f'10^{ex}' for ex in exp_range])
plt.legend()
plt.grid()
#plt.show()

#######

plt.figure()
    
for lam in lambda_range:
    
    res_02 = []
    res_05 = []
    res_08 = []

    exp_range = range(-5, 3)
    C_range = [10 ** C_exp for C_exp in exp_range]

    for C in C_range:

        print((lam, C))

        LTE_cross, _, scores = utils.cross_validation_base(D_BASE, L, svm.SupportVectorMachine, 10, None, [pre.PCA, 5], 0, [K, C, svm.get_kernel_RBF(lam, K**2), None] if prior is None else [K, C, svm.get_kernel_RBF(lam, K**2), [1 - prior, prior]], True)

        _, _, minDCF, _, _ = utils.get_metrics(scores, LTE_cross, 0.5, 1, 1, ret_all= True)
        res_05.append(minDCF)
        
    plt.plot(C_range, res_05, label= f'lam: {lam}')

plt.title(f'Cross Val 10 PCA 5 - prior: {prior}')
plt.xscale('log')
plt.xticks(C_range, [f'10^{ex}' for ex in exp_range])
plt.legend()
plt.grid()
plt.show()

exit(0)

prior = (L == 1).sum() / NTR

#model_params_init = [[K, C, c, d] for K in [0., 1e-6, 1e-4, 0.01, 1., 100.] for C in [1e-4, 0.01, 1.] for c in [0., 1.] for d in [2]]

model_params_init = [[K, C, c, d] for K in [0., 1e-6, 1e-4, 0.01, 1., 100.] for C in [1e-4, 0.01, 1.] for c in [0., 1.] for d in [3]]

model_params = [[K, C, svm.get_kernel_poly(c, d, K**2)] for K, C, c, d in model_params_init]

utils.cross_validation(D_BASE, L, 10, svm.SupportVectorMachine, model_params, progress= True, print_err= True, save= True, filename= 'results\\cross_val_poly3_svm.txt', model_params_print= model_params_init, prepro= [
    [(pre.NoTransform, [])],
    
    [(pre.Standardizer, [])],
    [(pre.Standardizer, []), (pre.PCA, [5])],
    [(pre.Standardizer, []), (pre.PCA, [4])],
    ])