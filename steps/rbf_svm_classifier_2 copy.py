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

for prior in [None, 0.2, 0.5, 0.8]:

    for C, lam in [(1, 0.01)]:

        print('Base')

        print((prior, C, lam))

        scores = svm.SupportVectorMachine(DTR_base, LTR, K, C, svm.get_kernel_RBF(lam, K**2), None if prior is None else [1 - prior, prior]).getScores(DTE_base)
        _, _, minDCF, _, _ = utils.get_metrics(scores, LTE, 0.2, 1, 1, ret_all= True)
        print(f'minDCF(0.2): {minDCF}')
        _, _, minDCF, _, _ = utils.get_metrics(scores, LTE, 0.5, 1, 1, ret_all= True)
        print(f'minDCF(0.5): {minDCF}')
        _, _, minDCF, _, _ = utils.get_metrics(scores, LTE, 0.8, 1, 1, ret_all= True)
        print(f'minDCF(0.8): {minDCF}')

######
    
#pca = pre.PCA(DTR_base)
#DTR = pca.transform(DTR_base, 5)
#DTE = pca.transform(DTE_base, 5)
#
#for prior in [None, 0.5]: #[0.2, 0.5, 0.8]:
#
#    for C, lam in [(1, 0.01), (10, 0.001)]:
#
#        print('PCA 5')
#
#        print((prior, C, lam))
#
#        scores = svm.SupportVectorMachine(DTR, LTR, K, C, svm.get_kernel_RBF(lam, K**2), None if prior is None else [1 - prior, prior]).getScores(DTE)
#        _, _, minDCF, _, _ = utils.get_metrics(scores, LTE, 0.2, 1, 1, ret_all= True)
#        print(f'minDCF(0.2): {minDCF}')
#        _, _, minDCF, _, _ = utils.get_metrics(scores, LTE, 0.5, 1, 1, ret_all= True)
#        print(f'minDCF(0.5): {minDCF}')
#        _, _, minDCF, _, _ = utils.get_metrics(scores, LTE, 0.8, 1, 1, ret_all= True)
#        print(f'minDCF(0.8): {minDCF}')

######
    
#gauss = pre.Gaussianizer(DTR_base, LTR)
#DTR = gauss.transform(DTR_base)
#DTE = gauss.transform(DTE_base)
#
#for prior in [None, 0.5]: #[0.2, 0.5, 0.8]:
#
#    for C, lam in [(1, 1), (10, 0.1)]:
#
#        print('Gaussianized')
#
#        print((prior, C, lam))
#
#        scores = svm.SupportVectorMachine(DTR, LTR, K, C, svm.get_kernel_RBF(lam, K**2), None if prior is None else [1 - prior, prior]).getScores(DTE)
#        _, _, minDCF, _, _ = utils.get_metrics(scores, LTE, 0.2, 1, 1, ret_all= True)
#        print(f'minDCF(0.2): {minDCF}')
#        _, _, minDCF, _, _ = utils.get_metrics(scores, LTE, 0.5, 1, 1, ret_all= True)
#        print(f'minDCF(0.5): {minDCF}')
#        _, _, minDCF, _, _ = utils.get_metrics(scores, LTE, 0.8, 1, 1, ret_all= True)
#        print(f'minDCF(0.8): {minDCF}')

######
    
gauss = pre.Gaussianizer(DTR_base, LTR)
DTR = gauss.transform(DTR_base)
DTE = gauss.transform(DTE_base)

pca = pre.PCA(DTR)
DTR = pca.transform(DTR, 5)
DTE = pca.transform(DTE, 5)

for prior in [None, 0.2, 0.5, 0.8]:

    for C, lam in [(1, 1)]:

        print('Gaussianized PCA 5')

        print((prior, C, lam))

        scores = svm.SupportVectorMachine(DTR, LTR, K, C, svm.get_kernel_RBF(lam, K**2), None if prior is None else [1 - prior, prior]).getScores(DTE)
        _, _, minDCF, _, _ = utils.get_metrics(scores, LTE, 0.2, 1, 1, ret_all= True)
        print(f'minDCF(0.2): {minDCF}')
        _, _, minDCF, _, _ = utils.get_metrics(scores, LTE, 0.5, 1, 1, ret_all= True)
        print(f'minDCF(0.5): {minDCF}')
        _, _, minDCF, _, _ = utils.get_metrics(scores, LTE, 0.8, 1, 1, ret_all= True)
        print(f'minDCF(0.8): {minDCF}')

#######
    # Cross val
#######
    
#for prior in [None, 0.2, 0.5, 0.8]:
#
#    for C, lam in [(1, 0.01)]:
#
#        print('Base')
#
#        print((prior, C, lam))
#
#        LTE_cross, _, scores = utils.cross_validation_base(D_BASE, L, svm.SupportVectorMachine, 10, None, None, 0, [K, C, svm.get_kernel_RBF(lam, K**2), None] if prior is None else [K, C, svm.get_kernel_RBF(lam, K**2), [1 - prior, prior]], True)
#        _, _, minDCF, _, _ = utils.get_metrics(scores, LTE_cross, 0.2, 1, 1, ret_all= True)
#        print(f'minDCF(0.2): {minDCF}')
#        _, _, minDCF, _, _ = utils.get_metrics(scores, LTE_cross, 0.5, 1, 1, ret_all= True)
#        print(f'minDCF(0.5): {minDCF}')
#        _, _, minDCF, _, _ = utils.get_metrics(scores, LTE_cross, 0.8, 1, 1, ret_all= True)
#        print(f'minDCF(0.8): {minDCF}')
#
########
#
########
#    
##for prior in [None, 0.5]: #[0.2, 0.5, 0.8]:
##
##    for C, lam in [(1, 1), (10, 0.1)]:
##
##        print('Gaussianized')
##
##        print((prior, C, lam))
##        
##        LTE_cross, _, scores = utils.cross_validation_base(D_BASE, L, svm.SupportVectorMachine, 10, pre.Gaussianizer, None, 0, [K, C, svm.get_kernel_RBF(lam, K**2), None] if prior is None else [K, C, svm.get_kernel_RBF(lam, K**2), [1 - prior, prior]], True)
##        _, _, minDCF, _, _ = utils.get_metrics(scores, LTE_cross, 0.2, 1, 1, ret_all= True)
##        print(f'minDCF(0.2): {minDCF}')
##        _, _, minDCF, _, _ = utils.get_metrics(scores, LTE_cross, 0.5, 1, 1, ret_all= True)
##        print(f'minDCF(0.5): {minDCF}')
##        _, _, minDCF, _, _ = utils.get_metrics(scores, LTE_cross, 0.8, 1, 1, ret_all= True)
##        print(f'minDCF(0.8): {minDCF}')
#
########
#    
##for prior in [None, 0.5]: #[0.2, 0.5, 0.8]:
##
##    for C, lam in [(1, 0.01), (10, 0.001)]:
##
##        print('PCA 5')
##
##        print((prior, C, lam))
##
##        LTE_cross, _, scores = utils.cross_validation_base(D_BASE, L, svm.SupportVectorMachine, 10, None, [pre.PCA, 5], 0, [K, C, svm.get_kernel_RBF(lam, K**2), None] if prior is None else [K, C, svm.get_kernel_RBF(lam, K**2), [1 - prior, prior]], True)
##        _, _, minDCF, _, _ = utils.get_metrics(scores, LTE_cross, 0.2, 1, 1, ret_all= True)
##        print(f'minDCF(0.2): {minDCF}')
##        _, _, minDCF, _, _ = utils.get_metrics(scores, LTE_cross, 0.5, 1, 1, ret_all= True)
##        print(f'minDCF(0.5): {minDCF}')
##        _, _, minDCF, _, _ = utils.get_metrics(scores, LTE_cross, 0.8, 1, 1, ret_all= True)
##        print(f'minDCF(0.8): {minDCF}')
#
#############
#    
#for prior in [None, 0.2, 0.5, 0.8]:
#
#    for C, lam in [(1, 1)]:
#
#        print('Gaussianized PCA 5')
#
#        print((prior, C, lam))
#
#        LTE_cross, _, scores = utils.cross_validation_base(D_BASE, L, svm.SupportVectorMachine, 10, pre.Gaussianizer, [pre.PCA, 5], 0, [K, C, svm.get_kernel_RBF(lam, K**2), None] if prior is None else [K, C, svm.get_kernel_RBF(lam, K**2), [1 - prior, prior]], True)
#        _, _, minDCF, _, _ = utils.get_metrics(scores, LTE_cross, 0.2, 1, 1, ret_all= True)
#        print(f'minDCF(0.2): {minDCF}')
#        _, _, minDCF, _, _ = utils.get_metrics(scores, LTE_cross, 0.5, 1, 1, ret_all= True)
#        print(f'minDCF(0.5): {minDCF}')
#        _, _, minDCF, _, _ = utils.get_metrics(scores, LTE_cross, 0.8, 1, 1, ret_all= True)
#        print(f'minDCF(0.8): {minDCF}')