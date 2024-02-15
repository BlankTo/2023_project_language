import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import ML_lib.log_reg as lg
import ML_lib.utils as utils
import ML_lib.preprocessing as pre

DTR_base = np.load('data_npy\\DTR_language.npy')
DTE_base = np.load('data_npy\\DTE_language.npy')
LTR = np.load('data_npy\\LTR_language.npy')
LTE = np.load('data_npy\\LTE_language.npy')


N_CLASS = 2
M_BASE, NTR = DTR_base.shape

#DTR_base, LTR, DTE_base, LTE = utils.shuffle_and_divide(D_BASE, L, 2/3)

for prior in [False, 0.2, 0.5, 0.8]:
    
    res_02 = []
    res_05 = []
    res_08 = []

    exp_range = range(-6, 3)
    lam_range = [0] #[10 ** lam_exp for lam_exp in exp_range] #[0]# + [10 ** lam_exp for lam_exp in exp_range]

    for lam in lam_range:

        print('Base')

        print((prior, lam))

        scores = lg.QuadraticRegressionClassifier(DTR_base, LTR, hyp= lam, priors= prior).getScores(DTE_base)
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
#    plt.plot(lam_range, res_02, label= 'eff: 0.2')
#    plt.plot(lam_range, res_05, label= 'eff: 0.5')
#    plt.plot(lam_range, res_08, label= 'eff: 0.8')
#    plt.title(f'No Prepro - prior: {prior}')
#    plt.xscale('log')
#    plt.xticks(lam_range, [f'10^{ex}' for ex in exp_range])
#    plt.legend()
#    plt.grid()
#    plt.show()

######
    
pca = pre.PCA(DTR_base)
DTR = pca.transform(DTR_base, 5)
DTE = pca.transform(DTE_base, 5)

for prior in [False, 0.2, 0.5, 0.8]:
    
    res_02 = []
    res_05 = []
    res_08 = []

    exp_range = range(-6, 3)
    lam_range = [0] #[10 ** lam_exp for lam_exp in exp_range] #[0]# + [10 ** lam_exp for lam_exp in exp_range]

    for lam in lam_range:

        print('PCA 5')

        print((prior, lam))

        scores = lg.QuadraticRegressionClassifier(DTR, LTR, hyp= lam, priors= prior).getScores(DTE)
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
#    plt.plot(lam_range, res_02, label= 'eff: 0.2')
#    plt.plot(lam_range, res_05, label= 'eff: 0.5')
#    plt.plot(lam_range, res_08, label= 'eff: 0.8')
#    plt.title(f'PCA 5 - prior: {prior}')
#    plt.xscale('log')
#    plt.xticks(lam_range, [f'10^{ex}' for ex in exp_range])
#    plt.legend()
#    plt.grid()
#    plt.show()

######