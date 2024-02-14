import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import ML_lib.log_reg as lg
import ML_lib.svm as svm
import ML_lib.gmm as gmm
import ML_lib.utils as utils
import ML_lib.preprocessing as pre
import ML_lib.plot_lib as pltl

D_BASE = np.load('data_npy\\DTR_language.npy')
L = np.load('data_npy\\LTR_language.npy')

N_CLASS = 2
M_BASE, NTR = D_BASE.shape

DTR_base, LTR, DTE_base, LTE = utils.shuffle_and_divide(D_BASE, L, 2/3)

prior = 0.5
lam = 0

LTE_cross_quad_log_reg, _, scores_quad_log_reg = utils.cross_validation_base(D_BASE, L, lg.QuadraticRegressionClassifier, 10, None, [pre.PCA, 5], 0, [lam, prior], True)

prior = None
C = 0.0001
K = 0
c = 100

LTE_cross_poly2, _, scores_poly2 = utils.cross_validation_base(D_BASE, L, svm.SupportVectorMachine, 10, None, [pre.PCA, 5], 0, [K, C, svm.get_kernel_poly(c, 2, K), None] if prior is None else [0, C, svm.get_kernel_poly(0, 2, 0), [1 - prior, prior]], True)

prior = 0.5
C = 1
K = 0
lam = 1

LTE_cross_rbf, _, scores_rbf = utils.cross_validation_base(D_BASE, L, svm.SupportVectorMachine, 10, pre.Gaussianizer, [pre.PCA, 5], 0, [K, C, svm.get_kernel_RBF(lam, K**2), None] if prior is None else [0, C, svm.get_kernel_RBF(lam, K**2), [1 - prior, prior]], True)

nG = 8

LTE_cross_gmm, _, scores_gmm = utils.cross_validation_base(D_BASE, L, gmm.GMMClassifier, 10, pre.Gaussianizer, [pre.PCA, 5], 0, [0.1, [nG], 0.01, ''], True)

#### DET

to_plot = [(scores_quad_log_reg, LTE_cross_quad_log_reg, 'Quad LogReg'), (scores_poly2, LTE_cross_poly2, 'Poly 2 SVM'), (scores_rbf, LTE_cross_rbf, 'RBF SVM'), (scores_gmm, LTE_cross_gmm, 'GMM')]

for i in range(len(to_plot)):
    for j in range(i+1, len(to_plot)):

        plt.figure()

        pltl.DETcurve(to_plot[i][0], to_plot[i][1], legend_text= to_plot[i][2], stack= True)
        pltl.DETcurve(to_plot[j][0], to_plot[j][1], legend_text= to_plot[j][2], stack= True)

        plt.xticks(np.arange(0., 1.1, 0.1))
        plt.yticks(np.arange(0., 1.1, 0.1))
        plt.grid()
        plt.title('DET - p1: ' + str(0.5) + ' Cfp: ' + str(1) + ' Cfn: ' + str(1))
        plt.legend()
        plt.show()

        plt.figure()

        pltl.BayesErrorPlot(to_plot[i][0], to_plot[i][1], np.linspace(-5, 5, 21), to_plot[i][2], stack= True)
        pltl.BayesErrorPlot(to_plot[j][0], to_plot[j][1], np.linspace(-5, 5, 21), to_plot[j][2], stack= True)

        plt.xticks(np.arange(0., 1.1, 0.1))
        plt.yticks(np.arange(0., 1.1, 0.1))
        plt.grid()
        plt.title('BEP - p1: ' + str(0.5) + ' Cfp: ' + str(1) + ' Cfn: ' + str(1))
        plt.legend()
        plt.show()

exit(0)

plt.figure()

pltl.DETcurve(scores_quad_log_reg, LTE_cross_quad_log_reg, legend_text= 'Quad log reg', stack= True)
pltl.DETcurve(scores_poly2, LTE_cross_poly2, legend_text= 'Poly 2 SVM', stack= True)
pltl.DETcurve(scores_rbf, LTE_cross_rbf, legend_text= 'RBF SVM', stack= True)
pltl.DETcurve(scores_gmm, LTE_cross_gmm, legend_text= 'GMM', stack= True)

plt.xticks(np.arange(0., 1.1, 0.1))
plt.yticks(np.arange(0., 1.1, 0.1))
plt.grid()
plt.title('DET - p1: ' + str(0.5) + ' Cfp: ' + str(1) + ' Cfn: ' + str(1))
plt.legend()
plt.show()

#### BEP

plt.figure()

pltl.BayesErrorPlot(scores_quad_log_reg, LTE_cross_quad_log_reg, np.linspace(-5, 5, 21), 'Quad log reg', stack= True)
pltl.BayesErrorPlot(scores_poly2, LTE_cross_poly2, np.linspace(-5, 5, 21), 'Poly 2 SVM', stack= True)
pltl.BayesErrorPlot(scores_rbf, LTE_cross_rbf, np.linspace(-5, 5, 21), 'RBF SVM', stack= True)
pltl.BayesErrorPlot(scores_gmm, LTE_cross_gmm, np.linspace(-5, 5, 21), 'GMM', stack= True)

plt.xticks(np.arange(0., 1.1, 0.1))
plt.yticks(np.arange(0., 1.1, 0.1))
plt.grid()
plt.title('BEP - p1: ' + str(0.5) + ' Cfp: ' + str(1) + ' Cfn: ' + str(1))
plt.legend()
plt.show()