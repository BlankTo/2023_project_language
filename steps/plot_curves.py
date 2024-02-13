import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import ML_lib.svm as svm
import ML_lib.utils as utils
import ML_lib.preprocessing as pre
import ML_lib.plot_lib as pltl

D_BASE = np.load('data_npy\\DTR_language.npy')
L = np.load('data_npy\\LTR_language.npy')

N_CLASS = 2
M_BASE, NTR = D_BASE.shape

DTR_base, LTR, DTE_base, LTE = utils.shuffle_and_divide(D_BASE, L, 2/3)

K = 0

for prior in [None]: #[0.2, 0.5, 0.8]:
    
    res_02 = []
    res_05 = []
    res_08 = []

    exp_range = range(-4, -3)
    C_range = [10 ** C_exp for C_exp in exp_range]

    for C in C_range:

        print('Base')

        print((prior, C))

        scores = svm.SupportVectorMachine(DTR_base, LTR, K, C, None, None if prior is None else [1 - prior, prior]).getScores(DTE_base)
        #pltl.ROCcurve(scores, LTE, 0.5, 1, 1)
        #pltl.DETcurve(scores, LTE, 0.5, 1, 1)

        pltl.BayesErrorPlot(scores, LTE, np.linspace(-5, 5, 21))