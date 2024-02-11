import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import ML_lib.log_reg as lg
import ML_lib.utils as utils
import ML_lib.preprocessing as pre

D_BASE = np.load('data_npy\\DTR_language.npy')
L = np.load('data_npy\\LTR_language.npy')


print(D_BASE.shape)
print(L.shape)

DTR_base, LTR, DTE_base, LTE = utils.shuffle_and_divide(D_BASE, L, 2/3)

for prior in [0.5]: #[0.2, 0.5, 0.8]:
    
    res_02 = []
    res_05 = []
    res_08 = []

    exp_range = range(-6, 4)
    lam_range = [10 ** lam_exp for lam_exp in exp_range]

    for lam in lam_range:

        scores = lg.LinearRegressionClassifier(DTR_base, LTR, lam= lam, priors= prior).getScores(DTE_base)
        _, _, minDCF, _, _ = utils.get_metrics(scores, LTE, 0.2, 1, 1, ret_all= True)
        res_02.append(minDCF)
        _, _, minDCF, _, _ = utils.get_metrics(scores, LTE, 0.5, 1, 1, ret_all= True)
        res_05.append(minDCF)
        _, _, minDCF, _, _ = utils.get_metrics(scores, LTE, 0.8, 1, 1, ret_all= True)
        res_08.append(minDCF)

    plt.figure()
    plt.plot(lam_range, res_02, label= 'eff: 0.2')
    plt.plot(lam_range, res_05, label= 'eff: 0.5')
    plt.plot(lam_range, res_08, label= 'eff: 0.8')
    plt.title(f'No Prepro - prior: {prior}')
    plt.xscale('log')
    plt.xticks(lam_range, [f'10^{ex}' for ex in exp_range])
    plt.legend()
    plt.grid()
    #plt.show()

######
    
pca = pre.PCA(DTR_base)
DTR = pca.transform(DTR_base, 5)
DTE = pca.transform(DTE_base, 5)

for prior in [0.5]: #[0.2, 0.5, 0.8]:
    
    res_02 = []
    res_05 = []
    res_08 = []

    exp_range = range(-6, 4)
    lam_range = [10 ** lam_exp for lam_exp in exp_range]

    for lam in lam_range:

        scores = lg.LinearRegressionClassifier(DTR, LTR, lam= lam, priors= prior).getScores(DTE)
        _, _, minDCF, _, _ = utils.get_metrics(scores, LTE, 0.2, 1, 1, ret_all= True)
        res_02.append(minDCF)
        _, _, minDCF, _, _ = utils.get_metrics(scores, LTE, 0.5, 1, 1, ret_all= True)
        res_05.append(minDCF)
        _, _, minDCF, _, _ = utils.get_metrics(scores, LTE, 0.8, 1, 1, ret_all= True)
        res_08.append(minDCF)

    plt.figure()
    plt.plot(lam_range, res_02, label= 'eff: 0.2')
    plt.plot(lam_range, res_05, label= 'eff: 0.5')
    plt.plot(lam_range, res_08, label= 'eff: 0.8')
    plt.title(f'PCA 5 - prior: {prior}')
    plt.xscale('log')
    plt.xticks(lam_range, [f'10^{ex}' for ex in exp_range])
    plt.legend()
    plt.grid()
    #plt.show()

######
    
gauss = pre.Gaussianizer(DTR_base, LTR)
DTR = gauss.transform(DTR_base)
DTE = gauss.transform(DTE_base)

for prior in [0.5]: #[0.2, 0.5, 0.8]:
    
    res_02 = []
    res_05 = []
    res_08 = []

    exp_range = range(-6, 4)
    lam_range = [10 ** lam_exp for lam_exp in exp_range]

    for lam in lam_range:

        scores = lg.LinearRegressionClassifier(DTR, LTR, lam= lam, priors= prior).getScores(DTE)
        _, _, minDCF, _, _ = utils.get_metrics(scores, LTE, 0.2, 1, 1, ret_all= True)
        res_02.append(minDCF)
        _, _, minDCF, _, _ = utils.get_metrics(scores, LTE, 0.5, 1, 1, ret_all= True)
        res_05.append(minDCF)
        _, _, minDCF, _, _ = utils.get_metrics(scores, LTE, 0.8, 1, 1, ret_all= True)
        res_08.append(minDCF)

    plt.figure()
    plt.plot(lam_range, res_02, label= 'eff: 0.2')
    plt.plot(lam_range, res_05, label= 'eff: 0.5')
    plt.plot(lam_range, res_08, label= 'eff: 0.8')
    plt.title(f'Gaussianized - prior: {prior}')
    plt.xscale('log')
    plt.xticks(lam_range, [f'10^{ex}' for ex in exp_range])
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

for prior in [0.5]: #[0.2, 0.5, 0.8]:
    
    res_02 = []
    res_05 = []
    res_08 = []

    exp_range = range(-6, 4)
    lam_range = [10 ** lam_exp for lam_exp in exp_range]

    for lam in lam_range:

        scores = lg.LinearRegressionClassifier(DTR, LTR, lam= lam, priors= prior).getScores(DTE)
        _, _, minDCF, _, _ = utils.get_metrics(scores, LTE, 0.2, 1, 1, ret_all= True)
        res_02.append(minDCF)
        _, _, minDCF, _, _ = utils.get_metrics(scores, LTE, 0.5, 1, 1, ret_all= True)
        res_05.append(minDCF)
        _, _, minDCF, _, _ = utils.get_metrics(scores, LTE, 0.8, 1, 1, ret_all= True)
        res_08.append(minDCF)

    plt.figure()
    plt.plot(lam_range, res_02, label= 'eff: 0.2')
    plt.plot(lam_range, res_05, label= 'eff: 0.5')
    plt.plot(lam_range, res_08, label= 'eff: 0.8')
    plt.title(f'Gaussianized PCA 5 - prior: {prior}')
    plt.xscale('log')
    plt.xticks(lam_range, [f'10^{ex}' for ex in exp_range])
    plt.legend()
    plt.grid()
    plt.show()

#######
    # Cross val
#######
    
for prior in [0.5]: #[0.2, 0.5, 0.8]:
    
    res_02 = []
    res_05 = []
    res_08 = []

    exp_range = range(-6, 4)
    lam_range = [10 ** lam_exp for lam_exp in exp_range]

    for lam in lam_range:

        LTE_cross, _, scores = utils.cross_validation_base(D_BASE, L, lg.LinearRegressionClassifier, 10, None, None, 0, [lam, prior], True)

        _, _, minDCF, _, _ = utils.get_metrics(scores, LTE_cross, 0.2, 1, 1, ret_all= True)
        res_02.append(minDCF)
        _, _, minDCF, _, _ = utils.get_metrics(scores, LTE_cross, 0.5, 1, 1, ret_all= True)
        res_05.append(minDCF)
        _, _, minDCF, _, _ = utils.get_metrics(scores, LTE_cross, 0.8, 1, 1, ret_all= True)
        res_08.append(minDCF)

    plt.figure()
    plt.plot(lam_range, res_02, label= 'eff: 0.2')
    plt.plot(lam_range, res_05, label= 'eff: 0.5')
    plt.plot(lam_range, res_08, label= 'eff: 0.8')
    plt.title(f'Cross Val 10 No Prepro - prior: {prior}')
    plt.xscale('log')
    plt.xticks(lam_range, [f'10^{ex}' for ex in exp_range])
    plt.legend()
    plt.grid()
    #plt.show()

#######
    
for prior in [0.5]: #[0.2, 0.5, 0.8]:
    
    res_02 = []
    res_05 = []
    res_08 = []

    exp_range = range(-6, 4)
    lam_range = [10 ** lam_exp for lam_exp in exp_range]

    for lam in lam_range:

        LTE_cross, _, scores = utils.cross_validation_base(D_BASE, L, lg.LinearRegressionClassifier, 10, pre.Gaussianizer, [pre.PCA, 5], 0, [lam, prior], True)

        _, _, minDCF, _, _ = utils.get_metrics(scores, LTE_cross, 0.2, 1, 1, ret_all= True)
        res_02.append(minDCF)
        _, _, minDCF, _, _ = utils.get_metrics(scores, LTE_cross, 0.5, 1, 1, ret_all= True)
        res_05.append(minDCF)
        _, _, minDCF, _, _ = utils.get_metrics(scores, LTE_cross, 0.8, 1, 1, ret_all= True)
        res_08.append(minDCF)

    plt.figure()
    plt.plot(lam_range, res_02, label= 'eff: 0.2')
    plt.plot(lam_range, res_05, label= 'eff: 0.5')
    plt.plot(lam_range, res_08, label= 'eff: 0.8')
    plt.title(f'Cross Val 10 Gaussianized PCA 5 - prior: {prior}')
    plt.xscale('log')
    plt.xticks(lam_range, [f'10^{ex}' for ex in exp_range])
    plt.legend()
    plt.grid()
    #plt.show()

#######
    
for prior in [0.5]: #[0.2, 0.5, 0.8]:
    
    res_02 = []
    res_05 = []
    res_08 = []

    exp_range = range(-6, 4)
    lam_range = [10 ** lam_exp for lam_exp in exp_range]

    for lam in lam_range:

        LTE_cross, _, scores = utils.cross_validation_base(D_BASE, L, lg.LinearRegressionClassifier, 10, pre.Gaussianizer, None, 0, [lam, prior], True)

        _, _, minDCF, _, _ = utils.get_metrics(scores, LTE_cross, 0.2, 1, 1, ret_all= True)
        res_02.append(minDCF)
        _, _, minDCF, _, _ = utils.get_metrics(scores, LTE_cross, 0.5, 1, 1, ret_all= True)
        res_05.append(minDCF)
        _, _, minDCF, _, _ = utils.get_metrics(scores, LTE_cross, 0.8, 1, 1, ret_all= True)
        res_08.append(minDCF)

    plt.figure()
    plt.plot(lam_range, res_02, label= 'eff: 0.2')
    plt.plot(lam_range, res_05, label= 'eff: 0.5')
    plt.plot(lam_range, res_08, label= 'eff: 0.8')
    plt.title(f'Cross Val 10 Gaussianized - prior: {prior}')
    plt.xscale('log')
    plt.xticks(lam_range, [f'10^{ex}' for ex in exp_range])
    plt.legend()
    plt.grid()
    #plt.show()

#######
    
for prior in [0.5]: #[0.2, 0.5, 0.8]:
    
    res_02 = []
    res_05 = []
    res_08 = []

    exp_range = range(-6, 4)
    lam_range = [10 ** lam_exp for lam_exp in exp_range]

    for lam in lam_range:

        LTE_cross, _, scores = utils.cross_validation_base(D_BASE, L, lg.LinearRegressionClassifier, 10, None, [pre.PCA, 5], 0, [lam, prior], True)

        _, _, minDCF, _, _ = utils.get_metrics(scores, LTE_cross, 0.2, 1, 1, ret_all= True)
        res_02.append(minDCF)
        _, _, minDCF, _, _ = utils.get_metrics(scores, LTE_cross, 0.5, 1, 1, ret_all= True)
        res_05.append(minDCF)
        _, _, minDCF, _, _ = utils.get_metrics(scores, LTE_cross, 0.8, 1, 1, ret_all= True)
        res_08.append(minDCF)

    plt.figure()
    plt.plot(lam_range, res_02, label= 'eff: 0.2')
    plt.plot(lam_range, res_05, label= 'eff: 0.5')
    plt.plot(lam_range, res_08, label= 'eff: 0.8')
    plt.title(f'Cross Val 10 PCA 5 - prior: {prior}')
    plt.xscale('log')
    plt.xticks(lam_range, [f'10^{ex}' for ex in exp_range])
    plt.legend()
    plt.grid()
    plt.show()

exit(0)

N_CLASS = 2
M_BASE, NTR = D_BASE.shape

prior = (L == 1).sum() / NTR

model_params = []
for prior in [0.5]: #[0.2, 0.5, 0.8]:#np.arange(0.1, 0.9, 0.2):
    model_params += [[0, prior]]
    for lam_exp in range(-6, 3):
        model_params += [[10 ** lam_exp, prior]]

utils.cross_validation(D_BASE, L, 10, lg.LinearRegressionClassifier, model_params, effective= [0.2, 0.5, 0.8], progress= True, print_err= True, save= True , filename= 'results\\cross_val_lin_log_reg.txt', prepro= [
    [(pre.NoTransform, [])],
    [(pre.Standardizer, [])],

    [(pre.Standardizer, []), (pre.PCA, [5])],
    [(pre.Standardizer, []), (pre.PCA, [4])],
    [(pre.Standardizer, []), (pre.PCA, [3])],

    [(pre.LDA, [1])],
    ])