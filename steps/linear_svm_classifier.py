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

for prior in [0.5, None]: #[0.2, 0.5, 0.8]:
    
    res_02 = []
    res_05 = []
    res_08 = []

    exp_range = range(-4, -3)
    C_range = [10 ** C_exp for C_exp in exp_range]

    for C in C_range:

        print('Base')

        print((prior, C))

        scores = svm.SupportVectorMachine(DTR_base, LTR, K, C, None, None if prior is None else [1 - prior, prior]).getScores(DTE_base)
        _, _, minDCF, _, _ = utils.get_metrics(scores, LTE, 0.2, 1, 1, ret_all= True)
        print(f'minDCF(0.2): {minDCF}')
        res_02.append(minDCF)
        _, _, minDCF, _, _ = utils.get_metrics(scores, LTE, 0.5, 1, 1, ret_all= True)
        print(f'minDCF(0.5): {minDCF}')
        res_05.append(minDCF)
        _, _, minDCF, _, _ = utils.get_metrics(scores, LTE, 0.8, 1, 1, ret_all= True)
        print(f'minDCF(0.8): {minDCF}')
        res_08.append(minDCF)

#    plt.figure()
#    plt.plot(C_range, res_02, label= 'eff: 0.2')
#    plt.plot(C_range, res_05, label= 'eff: 0.5')
#    plt.plot(C_range, res_08, label= 'eff: 0.8')
#    plt.title(f'No Prepro - prior: {prior}')
#    plt.xscale('log')
#    plt.xticks(C_range, [f'10^{ex}' for ex in exp_range])
#    plt.legend()
#    plt.grid()
#    #plt.show()

######
    
pca = pre.PCA(DTR_base)
DTR = pca.transform(DTR_base, 5)
DTE = pca.transform(DTE_base, 5)

for prior in [0.5, None]: #[0.2, 0.5, 0.8]:
    
    res_02 = []
    res_05 = []
    res_08 = []

    exp_range = range(-4, -3)
    C_range = [10 ** C_exp for C_exp in exp_range]

    for C in C_range:

        print('PCA 5')

        print((prior, C))

        scores = svm.SupportVectorMachine(DTR, LTR, K, C, None, None if prior is None else [1 - prior, prior]).getScores(DTE)
        _, _, minDCF, _, _ = utils.get_metrics(scores, LTE, 0.2, 1, 1, ret_all= True)
        print(f'minDCF(0.2): {minDCF}')
        res_02.append(minDCF)
        _, _, minDCF, _, _ = utils.get_metrics(scores, LTE, 0.5, 1, 1, ret_all= True)
        print(f'minDCF(0.5): {minDCF}')
        res_05.append(minDCF)
        _, _, minDCF, _, _ = utils.get_metrics(scores, LTE, 0.8, 1, 1, ret_all= True)
        print(f'minDCF(0.8): {minDCF}')
        res_08.append(minDCF)

#    plt.figure()
#    plt.plot(C_range, res_02, label= 'eff: 0.2')
#    plt.plot(C_range, res_05, label= 'eff: 0.5')
#    plt.plot(C_range, res_08, label= 'eff: 0.8')
#    plt.title(f'PCA 5 - prior: {prior}')
#    plt.xscale('log')
#    plt.xticks(C_range, [f'10^{ex}' for ex in exp_range])
#    plt.legend()
#    plt.grid()
#    #plt.show()

######
    
gauss = pre.Gaussianizer(DTR_base, LTR)
DTR = gauss.transform(DTR_base)
DTE = gauss.transform(DTE_base)

for prior in [0.5, None]: #[0.2, 0.5, 0.8]:
    
    res_02 = []
    res_05 = []
    res_08 = []

    exp_range = range(-4, -3)
    C_range = [10 ** C_exp for C_exp in exp_range]

    for C in C_range:

        print('Gaussianized')

        print((prior, C))

        scores = svm.SupportVectorMachine(DTR, LTR, K, C, None, None if prior is None else [1 - prior, prior]).getScores(DTE)
        _, _, minDCF, _, _ = utils.get_metrics(scores, LTE, 0.2, 1, 1, ret_all= True)
        print(f'minDCF(0.2): {minDCF}')
        res_02.append(minDCF)
        _, _, minDCF, _, _ = utils.get_metrics(scores, LTE, 0.5, 1, 1, ret_all= True)
        print(f'minDCF(0.5): {minDCF}')
        res_05.append(minDCF)
        _, _, minDCF, _, _ = utils.get_metrics(scores, LTE, 0.8, 1, 1, ret_all= True)
        print(f'minDCF(0.8): {minDCF}')
        res_08.append(minDCF)

#    plt.figure()
#    plt.plot(C_range, res_02, label= 'eff: 0.2')
#    plt.plot(C_range, res_05, label= 'eff: 0.5')
#    plt.plot(C_range, res_08, label= 'eff: 0.8')
#    plt.title(f'Gaussianized - prior: {prior}')
#    plt.xscale('log')
#    plt.xticks(C_range, [f'10^{ex}' for ex in exp_range])
#    plt.legend()
#    plt.grid()
#    #plt.show()

######
    
gauss = pre.Gaussianizer(DTR_base, LTR)
DTR = gauss.transform(DTR_base)
DTE = gauss.transform(DTE_base)

pca = pre.PCA(DTR)
DTR = pca.transform(DTR, 5)
DTE = pca.transform(DTE, 5)

for prior in [0.5, None]: #[0.2, 0.5, 0.8]:
    
    res_02 = []
    res_05 = []
    res_08 = []

    exp_range = range(-4, -3)
    C_range = [10 ** C_exp for C_exp in exp_range]

    for C in C_range:

        print('Gaussianized PCA 5')

        print((prior, C))

        scores = svm.SupportVectorMachine(DTR, LTR, K, C, None, None if prior is None else [1 - prior, prior]).getScores(DTE)
        _, _, minDCF, _, _ = utils.get_metrics(scores, LTE, 0.2, 1, 1, ret_all= True)
        print(f'minDCF(0.2): {minDCF}')
        res_02.append(minDCF)
        _, _, minDCF, _, _ = utils.get_metrics(scores, LTE, 0.5, 1, 1, ret_all= True)
        print(f'minDCF(0.5): {minDCF}')
        res_05.append(minDCF)
        _, _, minDCF, _, _ = utils.get_metrics(scores, LTE, 0.8, 1, 1, ret_all= True)
        print(f'minDCF(0.8): {minDCF}')
        res_08.append(minDCF)

#    plt.figure()
#    plt.plot(C_range, res_02, label= 'eff: 0.2')
#    plt.plot(C_range, res_05, label= 'eff: 0.5')
#    plt.plot(C_range, res_08, label= 'eff: 0.8')
#    plt.title(f'Gaussianized PCA 5 - prior: {prior}')
#    plt.xscale('log')
#    plt.xticks(C_range, [f'10^{ex}' for ex in exp_range])
#    plt.legend()
#    plt.grid()
#    plt.show()

#######
    # Cross val
#######
    
for prior in [0.5, None]: #[0.2, 0.5, 0.8]:
    
    res_02 = []
    res_05 = []
    res_08 = []

    exp_range = range(-4, -3)
    C_range = [10 ** C_exp for C_exp in exp_range]

    for C in C_range:

        print('Cross Val Base')

        print((prior, C))

        LTE_cross, _, scores = utils.cross_validation_base(D_BASE, L, svm.SupportVectorMachine, 10, None, None, 0, [K, C, None, None] if prior is None else [K, C, None, [1 - prior, prior]], True)
        _, _, minDCF, _, _ = utils.get_metrics(scores, LTE_cross, 0.2, 1, 1, ret_all= True)
        print(f'minDCF(0.2): {minDCF}')
        res_02.append(minDCF)
        _, _, minDCF, _, _ = utils.get_metrics(scores, LTE_cross, 0.5, 1, 1, ret_all= True)
        print(f'minDCF(0.5): {minDCF}')
        res_05.append(minDCF)
        _, _, minDCF, _, _ = utils.get_metrics(scores, LTE_cross, 0.8, 1, 1, ret_all= True)
        print(f'minDCF(0.8): {minDCF}')
        res_08.append(minDCF)

#    plt.figure()
#    plt.plot(C_range, res_02, label= 'eff: 0.2')
#    plt.plot(C_range, res_05, label= 'eff: 0.5')
#    plt.plot(C_range, res_08, label= 'eff: 0.8')
#    plt.title(f'Cross Val 10 No Prepro - prior: {prior}')
#    plt.xscale('log')
#    plt.xticks(C_range, [f'10^{ex}' for ex in exp_range])
#    plt.legend()
#    plt.grid()
#    #plt.show()

#######

#######
    
for prior in [0.5, None]: #[0.2, 0.5, 0.8]:
    
    res_02 = []
    res_05 = []
    res_08 = []

    exp_range = range(-4, -3)
    C_range = [10 ** C_exp for C_exp in exp_range]

    for C in C_range:

        print('Cross Val Gaussianized')

        print((prior, C))

        LTE_cross, _, scores = utils.cross_validation_base(D_BASE, L, svm.SupportVectorMachine, 10, pre.Gaussianizer, None, 0, [K, C, None, None] if prior is None else [K, C, None, [1 - prior, prior]], True)
        _, _, minDCF, _, _ = utils.get_metrics(scores, LTE_cross, 0.2, 1, 1, ret_all= True)
        print(f'minDCF(0.2): {minDCF}')
        res_02.append(minDCF)
        _, _, minDCF, _, _ = utils.get_metrics(scores, LTE_cross, 0.5, 1, 1, ret_all= True)
        print(f'minDCF(0.5): {minDCF}')
        res_05.append(minDCF)
        _, _, minDCF, _, _ = utils.get_metrics(scores, LTE_cross, 0.8, 1, 1, ret_all= True)
        print(f'minDCF(0.8): {minDCF}')
        res_08.append(minDCF)

#    plt.figure()
#    plt.plot(C_range, res_02, label= 'eff: 0.2')
#    plt.plot(C_range, res_05, label= 'eff: 0.5')
#    plt.plot(C_range, res_08, label= 'eff: 0.8')
#    plt.title(f'Cross Val 10 Gaussianized - prior: {prior}')
#    plt.xscale('log')
#    plt.xticks(C_range, [f'10^{ex}' for ex in exp_range])
#    plt.legend()
#    plt.grid()
#    #plt.show()

#######
    
for prior in [0.5, None]: #[0.2, 0.5, 0.8]:
    
    res_02 = []
    res_05 = []
    res_08 = []

    exp_range = range(-4, -3)
    C_range = [10 ** C_exp for C_exp in exp_range]

    for C in C_range:

        print('Cross Val PCA 5')

        print((prior, C))

        LTE_cross, _, scores = utils.cross_validation_base(D_BASE, L, svm.SupportVectorMachine, 10, None, [pre.PCA, 5], 0, [K, C, None, None] if prior is None else [K, C, None, [1 - prior, prior]], True)
        _, _, minDCF, _, _ = utils.get_metrics(scores, LTE_cross, 0.2, 1, 1, ret_all= True)
        print(f'minDCF(0.2): {minDCF}')
        res_02.append(minDCF)
        _, _, minDCF, _, _ = utils.get_metrics(scores, LTE_cross, 0.5, 1, 1, ret_all= True)
        print(f'minDCF(0.5): {minDCF}')
        res_05.append(minDCF)
        _, _, minDCF, _, _ = utils.get_metrics(scores, LTE_cross, 0.8, 1, 1, ret_all= True)
        print(f'minDCF(0.8): {minDCF}')
        res_08.append(minDCF)

#    plt.figure()
#    plt.plot(C_range, res_02, label= 'eff: 0.2')
#    plt.plot(C_range, res_05, label= 'eff: 0.5')
#    plt.plot(C_range, res_08, label= 'eff: 0.8')
#    plt.title(f'Cross Val 10 PCA 5 - prior: {prior}')
#    plt.xscale('log')
#    plt.xticks(C_range, [f'10^{ex}' for ex in exp_range])
#    plt.legend()
#    plt.grid()
#    plt.show()

############
    
for prior in [0.5, None]: #[0.2, 0.5, 0.8]:
    
    res_02 = []
    res_05 = []
    res_08 = []

    exp_range = range(-4, -3)
    C_range = [10 ** C_exp for C_exp in exp_range]

    for C in C_range:

        print('Cross Val Gaussianized PCA 5')

        print((prior, C))

        LTE_cross, _, scores = utils.cross_validation_base(D_BASE, L, svm.SupportVectorMachine, 10, pre.Gaussianizer, [pre.PCA, 5], 0, [K, C, None, None] if prior is None else [K, C, None, [1 - prior, prior]], True)
        _, _, minDCF, _, _ = utils.get_metrics(scores, LTE_cross, 0.2, 1, 1, ret_all= True)
        print(f'minDCF(0.2): {minDCF}')
        res_02.append(minDCF)
        _, _, minDCF, _, _ = utils.get_metrics(scores, LTE_cross, 0.5, 1, 1, ret_all= True)
        print(f'minDCF(0.5): {minDCF}')
        res_05.append(minDCF)
        _, _, minDCF, _, _ = utils.get_metrics(scores, LTE_cross, 0.8, 1, 1, ret_all= True)
        print(f'minDCF(0.8): {minDCF}')
        res_08.append(minDCF)

#    plt.figure()
#    plt.plot(C_range, res_02, label= 'eff: 0.2')
#    plt.plot(C_range, res_05, label= 'eff: 0.5')
#    plt.plot(C_range, res_08, label= 'eff: 0.8')
#    plt.title(f'Cross Val 10 Gaussianized PCA 5 - prior: {prior}')
#    plt.xscale('log')
#    plt.xticks(C_range, [f'10^{ex}' for ex in exp_range])
#    plt.legend()
#    plt.grid()
#    #plt.show()


exit(0)

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