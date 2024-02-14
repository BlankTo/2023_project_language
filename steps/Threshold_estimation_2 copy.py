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

D_BASE = np.load('data_npy\\DTR_language.npy')
L = np.load('data_npy\\LTR_language.npy')

N_CLASS = 2
M_BASE, NTR = D_BASE.shape

DTR_base, LTR, DTE_base, LTE = utils.shuffle_and_divide(D_BASE, L, 2/3)

#

print('Quad Log Reg PCA 5 lam 0 prior 05')

prior = 0.5
lam = 0

LTE_cross_quad_log_reg_1, _, scores_quad_log_reg_1 = utils.cross_validation_base(D_BASE, L, lg.QuadraticRegressionClassifier, 10, None, [pre.PCA, 5], 0, [lam, prior], True)

print(f'Quad LogReg nDCF(0.2): {utils.get_metrics(scores_quad_log_reg_1, LTE_cross_quad_log_reg_1, 0.2, 1, 1, ret_all= True)[1]}')
print(f'Quad LogReg nDCF(0.5): {utils.get_metrics(scores_quad_log_reg_1, LTE_cross_quad_log_reg_1, 0.5, 1, 1, ret_all= True)[1]}')
print(f'Quad LogReg nDCF(0.8): {utils.get_metrics(scores_quad_log_reg_1, LTE_cross_quad_log_reg_1, 0.8, 1, 1, ret_all= True)[1]}')

minDCF, best_threshold_quad_log_reg_1_02 = utils.getMinNormDCF(scores_quad_log_reg_1, LTE_cross_quad_log_reg_1, 0.2, 1, 1, retThreshold= True)
print(f'Quad LogReg best_threshold(0.2): {best_threshold_quad_log_reg_1_02} - minDCF(0.2): {minDCF}')
utils.get_metrics_threshold(scores_quad_log_reg_1, LTE_cross_quad_log_reg_1, best_threshold_quad_log_reg_1_02, 0.2, 1, 1)

minDCF, best_threshold_quad_log_reg_1_05 = utils.getMinNormDCF(scores_quad_log_reg_1, LTE_cross_quad_log_reg_1, 0.5, 1, 1, retThreshold= True)
print(f'Quad LogReg best_threshold(0.5): {best_threshold_quad_log_reg_1_05} - minDCF(0.5): {minDCF}')
utils.get_metrics_threshold(scores_quad_log_reg_1, LTE_cross_quad_log_reg_1, best_threshold_quad_log_reg_1_05, 0.5, 1, 1)

minDCF, best_threshold_quad_log_reg_1_08 = utils.getMinNormDCF(scores_quad_log_reg_1, LTE_cross_quad_log_reg_1, 0.8, 1, 1, retThreshold= True)
print(f'Quad LogReg best_threshold(0.8): {best_threshold_quad_log_reg_1_08} - minDCF(0.8): {minDCF}')
utils.get_metrics_threshold(scores_quad_log_reg_1, LTE_cross_quad_log_reg_1, best_threshold_quad_log_reg_1_08, 0.8, 1, 1)

#

print('Quad Log Reg PCA 5 lam 1 prior 05')

prior = 0.5
lam = 1

LTE_cross_quad_log_reg_2, _, scores_quad_log_reg_2 = utils.cross_validation_base(D_BASE, L, lg.QuadraticRegressionClassifier, 10, None, [pre.PCA, 5], 0, [lam, prior], True)

print(f'Quad LogReg nDCF(0.2): {utils.get_metrics(scores_quad_log_reg_2, LTE_cross_quad_log_reg_2, 0.2, 1, 1, ret_all= True)[1]}')
print(f'Quad LogReg nDCF(0.5): {utils.get_metrics(scores_quad_log_reg_2, LTE_cross_quad_log_reg_2, 0.5, 1, 1, ret_all= True)[1]}')
print(f'Quad LogReg nDCF(0.8): {utils.get_metrics(scores_quad_log_reg_2, LTE_cross_quad_log_reg_2, 0.8, 1, 1, ret_all= True)[1]}')

minDCF, best_threshold_quad_log_reg_2_02 = utils.getMinNormDCF(scores_quad_log_reg_2, LTE_cross_quad_log_reg_2, 0.2, 1, 1, retThreshold= True)
print(f'Quad LogReg best_threshold(0.2): {best_threshold_quad_log_reg_2_02} - minDCF(0.2): {minDCF}')
utils.get_metrics_threshold(scores_quad_log_reg_2, LTE_cross_quad_log_reg_2, best_threshold_quad_log_reg_2_02, 0.2, 1, 1)

minDCF, best_threshold_quad_log_reg_2_05 = utils.getMinNormDCF(scores_quad_log_reg_2, LTE_cross_quad_log_reg_2, 0.5, 1, 1, retThreshold= True)
print(f'Quad LogReg best_threshold(0.5): {best_threshold_quad_log_reg_2_05} - minDCF(0.5): {minDCF}')
utils.get_metrics_threshold(scores_quad_log_reg_2, LTE_cross_quad_log_reg_2, best_threshold_quad_log_reg_2_05, 0.5, 1, 1)

minDCF, best_threshold_quad_log_reg_2_08 = utils.getMinNormDCF(scores_quad_log_reg_2, LTE_cross_quad_log_reg_2, 0.8, 1, 1, retThreshold= True)
print(f'Quad LogReg best_threshold(0.8): {best_threshold_quad_log_reg_2_08} - minDCF(0.8): {minDCF}')
utils.get_metrics_threshold(scores_quad_log_reg_2, LTE_cross_quad_log_reg_2, best_threshold_quad_log_reg_1_08, 0.8, 1, 1)

#

print('Poly 2 SVM PCA 5 C 0.0001 K 0 c 100 prior None')

prior = None
C = 0.0001
K = 0
c = 100

LTE_cross_poly2, _, scores_poly2 = utils.cross_validation_base(D_BASE, L, svm.SupportVectorMachine, 10, None, [pre.PCA, 5], 0, [K, C, svm.get_kernel_poly(c, 2, K), None] if prior is None else [0, C, svm.get_kernel_poly(0, 2, 0), [1 - prior, prior]], True)

print(f'POLY 2 SVM nDCF(0.2): {utils.get_metrics(scores_poly2, LTE_cross_poly2, 0.2, 1, 1, ret_all= True)[1]}')
print(f'POLY 2 SVM nDCF(0.5): {utils.get_metrics(scores_poly2, LTE_cross_poly2, 0.5, 1, 1, ret_all= True)[1]}')
print(f'POLY 2 SVM nDCF(0.8): {utils.get_metrics(scores_poly2, LTE_cross_poly2, 0.8, 1, 1, ret_all= True)[1]}')

minDCF, best_threshold_poly2_02 = utils.getMinNormDCF(scores_poly2, LTE_cross_poly2, 0.2, 1, 1, retThreshold= True)
print(f'Poly 2 SVM best_threshold(0.2): {best_threshold_poly2_02} - minDCF(0.2): {minDCF}')
utils.get_metrics_threshold(scores_poly2, LTE_cross_poly2, best_threshold_poly2_02, 0.2, 1, 1)

minDCF, best_threshold_poly2_05 = utils.getMinNormDCF(scores_poly2, LTE_cross_poly2, 0.5, 1, 1, retThreshold= True)
print(f'Poly 2 SVM best_threshold(0.5): {best_threshold_poly2_05} - minDCF(0.5): {minDCF}')
utils.get_metrics_threshold(scores_poly2, LTE_cross_poly2, best_threshold_poly2_05, 0.5, 1, 1)

minDCF, best_threshold_poly2_08 = utils.getMinNormDCF(scores_poly2, LTE_cross_poly2, 0.8, 1, 1, retThreshold= True)
print(f'Poly 2 SVM best_threshold(0.8): {best_threshold_poly2_08} - minDCF(0.8): {minDCF}')
utils.get_metrics_threshold(scores_poly2, LTE_cross_poly2, best_threshold_poly2_08, 0.8, 1, 1)

#

print('RBF SVM Raw C 1 lam 0.01 prior 05')

prior = 0.5
C = 1
K = 0
lam = 0.01

LTE_cross_rbf_1, _, scores_rbf_1 = utils.cross_validation_base(D_BASE, L, svm.SupportVectorMachine, 10, None, None, 0, [K, C, svm.get_kernel_RBF(lam, K**2), None if prior is None else [1 - prior, prior]], True)

print(f'RBF SVM nDCF(0.2): {utils.get_metrics(scores_rbf_1, LTE_cross_rbf_1, 0.2, 1, 1, ret_all= True)[1]}')
print(f'RBF SVM nDCF(0.5): {utils.get_metrics(scores_rbf_1, LTE_cross_rbf_1, 0.5, 1, 1, ret_all= True)[1]}')
print(f'RBF SVM nDCF(0.8): {utils.get_metrics(scores_rbf_1, LTE_cross_rbf_1, 0.8, 1, 1, ret_all= True)[1]}')

minDCF, best_threshold_rbf_1_02 = utils.getMinNormDCF(scores_rbf_1, LTE_cross_rbf_1, 0.2, 1, 1, retThreshold= True)
print(f'RBF SVM best_threshold(0.2): {best_threshold_rbf_1_02} - minDCF(0.2): {minDCF}')
utils.get_metrics_threshold(scores_rbf_1, LTE_cross_rbf_1, best_threshold_rbf_1_02, 0.2, 1, 1)

minDCF, best_threshold_rbf_1_05 = utils.getMinNormDCF(scores_rbf_1, LTE_cross_rbf_1, 0.5, 1, 1, retThreshold= True)
print(f'RBF SVM best_threshold(0.5): {best_threshold_rbf_1_05} - minDCF(0.5): {minDCF}')
utils.get_metrics_threshold(scores_rbf_1, LTE_cross_rbf_1, best_threshold_rbf_1_05, 0.5, 1, 1)

minDCF, best_threshold_rbf_1_08 = utils.getMinNormDCF(scores_rbf_1, LTE_cross_rbf_1, 0.8, 1, 1, retThreshold= True)
print(f'RBF SVM best_threshold(0.8): {best_threshold_rbf_1_08} - minDCF(0.8): {minDCF}')
utils.get_metrics_threshold(scores_rbf_1, LTE_cross_rbf_1, best_threshold_rbf_1_08, 0.8, 1, 1)

#

print('RBF SVM Gauss PCA 5 C 1 lam 1 prior 05')

prior = 0.5
C = 1
K = 0
lam = 1

LTE_cross_rbf_2, _, scores_rbf_2 = utils.cross_validation_base(D_BASE, L, svm.SupportVectorMachine, 10, pre.Gaussianizer, [pre.PCA, 5], 0, [K, C, svm.get_kernel_RBF(lam, K**2), None if prior is None else [1 - prior, prior]], True)

print(f'RBF SVM nDCF(0.2): {utils.get_metrics(scores_rbf_2, LTE_cross_rbf_2, 0.2, 1, 1, ret_all= True)[1]}')
print(f'RBF SVM nDCF(0.5): {utils.get_metrics(scores_rbf_2, LTE_cross_rbf_2, 0.5, 1, 1, ret_all= True)[1]}')
print(f'RBF SVM nDCF(0.8): {utils.get_metrics(scores_rbf_2, LTE_cross_rbf_2, 0.8, 1, 1, ret_all= True)[1]}')

minDCF, best_threshold_rbf_2_02 = utils.getMinNormDCF(scores_rbf_2, LTE_cross_rbf_2, 0.2, 1, 1, retThreshold= True)
print(f'RBF SVM best_threshold(0.2): {best_threshold_rbf_2_02} - minDCF(0.2): {minDCF}')
utils.get_metrics_threshold(scores_rbf_2, LTE_cross_rbf_2, best_threshold_rbf_2_02, 0.2, 1, 1)

minDCF, best_threshold_rbf_2_05 = utils.getMinNormDCF(scores_rbf_2, LTE_cross_rbf_2, 0.5, 1, 1, retThreshold= True)
print(f'RBF SVM best_threshold(0.5): {best_threshold_rbf_2_05} - minDCF(0.5): {minDCF}')
utils.get_metrics_threshold(scores_rbf_2, LTE_cross_rbf_2, best_threshold_rbf_2_05, 0.5, 1, 1)

minDCF, best_threshold_rbf_2_08 = utils.getMinNormDCF(scores_rbf_2, LTE_cross_rbf_2, 0.8, 1, 1, retThreshold= True)
print(f'RBF SVM best_threshold(0.8): {best_threshold_rbf_2_08} - minDCF(0.8): {minDCF}')
utils.get_metrics_threshold(scores_rbf_2, LTE_cross_rbf_2, best_threshold_rbf_2_08, 0.8, 1, 1)

#

print('GMM Gauss PCA 5 nG 8')

nG = 8

LTE_cross_gmm, _, scores_gmm = utils.cross_validation_base(D_BASE, L, gmm.GMMClassifier, 10, pre.Gaussianizer, [pre.PCA, 5], 0, [0.1, [nG], 0.01, ''], True)

print(f'GMM nDCF(0.2): {utils.get_metrics(scores_gmm, LTE_cross_gmm, 0.2, 1, 1, ret_all= True)[1]}')
print(f'GMM nDCF(0.5): {utils.get_metrics(scores_gmm, LTE_cross_gmm, 0.5, 1, 1, ret_all= True)[1]}')
print(f'GMM nDCF(0.8): {utils.get_metrics(scores_gmm, LTE_cross_gmm, 0.8, 1, 1, ret_all= True)[1]}')

minDCF, best_threshold_gmm_02 = utils.getMinNormDCF(scores_gmm, LTE_cross_gmm, 0.2, 1, 1, retThreshold= True)
print(f'GMM best_threshold(0.2): {best_threshold_gmm_02} - minDCF(0.2): {minDCF}')
utils.get_metrics_threshold(scores_gmm, LTE_cross_gmm, best_threshold_gmm_02, 0.2, 1, 1)

minDCF, best_threshold_gmm_05 = utils.getMinNormDCF(scores_gmm, LTE_cross_gmm, 0.5, 1, 1, retThreshold= True)
print(f'GMM best_threshold(0.5): {best_threshold_gmm_05} - minDCF(0.5): {minDCF}')
utils.get_metrics_threshold(scores_gmm, LTE_cross_gmm, best_threshold_gmm_05, 0.5, 1, 1)

minDCF, best_threshold_gmm_08 = utils.getMinNormDCF(scores_gmm, LTE_cross_gmm, 0.8, 1, 1, retThreshold= True)
print(f'GMM best_threshold(0.8): {best_threshold_gmm_08} - minDCF(0.8): {minDCF}')
utils.get_metrics_threshold(scores_gmm, LTE_cross_gmm, best_threshold_gmm_08, 0.8, 1, 1)

#

print('TIED GMM PCA 5 nG 128')

nG = 128

LTE_cross_gmm_tied, _, scores_gmm_tied = utils.cross_validation_base(D_BASE, L, gmm.GMMClassifier, 10, None, [pre.PCA, 5], 0, [0.1, [nG], 0.01, 'tied'], True)

print(f'GMM nDCF(0.2): {utils.get_metrics(scores_gmm_tied, LTE_cross_gmm_tied, 0.2, 1, 1, ret_all= True)[1]}')
print(f'GMM nDCF(0.5): {utils.get_metrics(scores_gmm_tied, LTE_cross_gmm_tied, 0.5, 1, 1, ret_all= True)[1]}')
print(f'GMM nDCF(0.8): {utils.get_metrics(scores_gmm_tied, LTE_cross_gmm_tied, 0.8, 1, 1, ret_all= True)[1]}')

minDCF, best_threshold_gmm_tied_02 = utils.getMinNormDCF(scores_gmm_tied, LTE_cross_gmm_tied, 0.2, 1, 1, retThreshold= True)
print(f'GMM best_threshold(0.2): {best_threshold_gmm_tied_02} - minDCF(0.2): {minDCF}')
utils.get_metrics_threshold(scores_gmm_tied, LTE_cross_gmm_tied, best_threshold_gmm_tied_02, 0.2, 1, 1)

minDCF, best_threshold_gmm_tied_05 = utils.getMinNormDCF(scores_gmm_tied, LTE_cross_gmm_tied, 0.5, 1, 1, retThreshold= True)
print(f'GMM best_threshold(0.5): {best_threshold_gmm_tied_05} - minDCF(0.5): {minDCF}')
utils.get_metrics_threshold(scores_gmm_tied, LTE_cross_gmm_tied, best_threshold_gmm_tied_05, 0.5, 1, 1)

minDCF, best_threshold_gmm_tied_08 = utils.getMinNormDCF(scores_gmm_tied, LTE_cross_gmm_tied, 0.8, 1, 1, retThreshold= True)
print(f'GMM best_threshold(0.8): {best_threshold_gmm_tied_08} - minDCF(0.8): {minDCF}')
utils.get_metrics_threshold(scores_gmm_tied, LTE_cross_gmm_tied, best_threshold_gmm_tied_08, 0.8, 1, 1)

#####

DTR = D_BASE
LTR = L
DTE = np.load('data_npy\\DTE_language.npy')
LTE = np.load('data_npy\\LTE_language.npy')

gauss = pre.Gaussianizer(DTR, LTR)
DTR_gauss = gauss.transform(DTR)
DTE_gauss = gauss.transform(DTE)

pca = pre.PCA(DTR)
DTR_pca = pca.transform(DTR, 5)
DTE_pca = pca.transform(DTE, 5)

pca_gauss = pre.PCA(DTR_gauss)
DTR_gauss_pca = pca.transform(DTR_gauss, 5)
DTE_gauss_pca = pca.transform(DTE_gauss, 5)

#

print('--------------------------------------------')

print('Quad Log Reg PCA 5 lam 0 prior 05')

prior = 0.5
lam = 0

scores = lg.QuadraticRegressionClassifier(DTR_pca, LTR, hyp= lam, priors = prior).getScores(DTE_pca)

print(f'bayes threshold: {utils.getBayesThreshold(0.2, 1, 1)} - best_threshold: {best_threshold_quad_log_reg_1_02}')
utils.get_metrics(scores, LTE, 0.2, 1, 1)
print(f'bayes threshold: {utils.getBayesThreshold(0.5, 1, 1)} - best_threshold: {best_threshold_quad_log_reg_1_05}')
utils.get_metrics(scores, LTE, 0.5, 1, 1)
print(f'bayes threshold: {utils.getBayesThreshold(0.8, 1, 1)} - best_threshold: {best_threshold_quad_log_reg_1_08}')
utils.get_metrics(scores, LTE, 0.8, 1, 1)

utils.get_metrics_threshold(scores, LTE, best_threshold_quad_log_reg_1_02, 0.2, 1, 1)
utils.get_metrics_threshold(scores, LTE, best_threshold_quad_log_reg_1_05, 0.5, 1, 1)
utils.get_metrics_threshold(scores, LTE, best_threshold_quad_log_reg_1_08, 0.8, 1, 1)

#
#

print('--------------------------------------------')

print('Quad Log Reg PCA 5 lam 1 prior 05')

prior = 0.5
lam = 1

scores = lg.QuadraticRegressionClassifier(DTR_pca, LTR, hyp= lam, priors = prior).getScores(DTE_pca)

print(f'bayes threshold: {utils.getBayesThreshold(0.2, 1, 1)} - best_threshold: {best_threshold_quad_log_reg_2_02}')
utils.get_metrics(scores, LTE, 0.2, 1, 1)
print(f'bayes threshold: {utils.getBayesThreshold(0.5, 1, 1)} - best_threshold: {best_threshold_quad_log_reg_2_05}')
utils.get_metrics(scores, LTE, 0.5, 1, 1)
print(f'bayes threshold: {utils.getBayesThreshold(0.8, 1, 1)} - best_threshold: {best_threshold_quad_log_reg_2_08}')
utils.get_metrics(scores, LTE, 0.8, 1, 1)

utils.get_metrics_threshold(scores, LTE, best_threshold_quad_log_reg_2_02, 0.2, 1, 1)
utils.get_metrics_threshold(scores, LTE, best_threshold_quad_log_reg_2_05, 0.5, 1, 1)
utils.get_metrics_threshold(scores, LTE, best_threshold_quad_log_reg_2_08, 0.8, 1, 1)

#
#

print('--------------------------------------------')

print('Poly 2 SVM PCA 5 C 0.0001 K 0 c 100 prior None')

prior = None
C = 0.0001
K = 0
c = 100

scores = svm.SupportVectorMachine(DTR_pca, LTR, K, C, svm.get_kernel_poly(c, 2, K**2), None if prior is None else [1 - prior, prior]).getScores(DTE_pca)

print(f'bayes threshold: {utils.getBayesThreshold(0.2, 1, 1)} - best_threshold: {best_threshold_poly2_02}')
utils.get_metrics(scores, LTE, 0.2, 1, 1)
print(f'bayes threshold: {utils.getBayesThreshold(0.5, 1, 1)} - best_threshold: {best_threshold_poly2_05}')
utils.get_metrics(scores, LTE, 0.5, 1, 1)
print(f'bayes threshold: {utils.getBayesThreshold(0.8, 1, 1)} - best_threshold: {best_threshold_poly2_08}')
utils.get_metrics(scores, LTE, 0.8, 1, 1)

utils.get_metrics_threshold(scores, LTE, best_threshold_poly2_02, 0.2, 1, 1)
utils.get_metrics_threshold(scores, LTE, best_threshold_poly2_05, 0.5, 1, 1)
utils.get_metrics_threshold(scores, LTE, best_threshold_poly2_08, 0.8, 1, 1)

#
#

print('--------------------------------------------')

print('RBF SVM Raw C 1 lam 0.01 prior 05')

prior = 0.5
C = 1
K = 0
lam = 0.01

scores = svm.SupportVectorMachine(DTR, LTR, K, C, svm.get_kernel_RBF(lam, K**2), None if prior is None else [1 - prior, prior]).getScores(DTE_gauss_pca)

print(f'bayes threshold: {utils.getBayesThreshold(0.2, 1, 1)} - best_threshold: {best_threshold_rbf_1_02}')
utils.get_metrics(scores, LTE, 0.2, 1, 1)
print(f'bayes threshold: {utils.getBayesThreshold(0.5, 1, 1)} - best_threshold: {best_threshold_rbf_1_05}')
utils.get_metrics(scores, LTE, 0.5, 1, 1)
print(f'bayes threshold: {utils.getBayesThreshold(0.8, 1, 1)} - best_threshold: {best_threshold_rbf_1_08}')
utils.get_metrics(scores, LTE, 0.8, 1, 1)

utils.get_metrics_threshold(scores, LTE, best_threshold_rbf_1_02, 0.2, 1, 1)
utils.get_metrics_threshold(scores, LTE, best_threshold_rbf_1_05, 0.5, 1, 1)
utils.get_metrics_threshold(scores, LTE, best_threshold_rbf_1_08, 0.8, 1, 1)

#
#

print('--------------------------------------------')

print('RBF SVM Gauss PCA 5 C 1 lam 1 prior 05')

prior = 0.5
C = 1
K = 0
lam = 1

scores = svm.SupportVectorMachine(DTR_gauss_pca, LTR, K, C, svm.get_kernel_RBF(lam, K**2), None if prior is None else [1 - prior, prior]).getScores(DTE_gauss_pca)

print(f'bayes threshold: {utils.getBayesThreshold(0.2, 1, 1)} - best_threshold: {best_threshold_rbf_2_02}')
utils.get_metrics(scores, LTE, 0.2, 1, 1)
print(f'bayes threshold: {utils.getBayesThreshold(0.5, 1, 1)} - best_threshold: {best_threshold_rbf_2_05}')
utils.get_metrics(scores, LTE, 0.5, 1, 1)
print(f'bayes threshold: {utils.getBayesThreshold(0.8, 1, 1)} - best_threshold: {best_threshold_rbf_2_08}')
utils.get_metrics(scores, LTE, 0.8, 1, 1)

utils.get_metrics_threshold(scores, LTE, best_threshold_rbf_2_02, 0.2, 1, 1)
utils.get_metrics_threshold(scores, LTE, best_threshold_rbf_2_05, 0.5, 1, 1)
utils.get_metrics_threshold(scores, LTE, best_threshold_rbf_2_08, 0.8, 1, 1)

#
#

print('--------------------------------------------')
print('GMM')

nG = 8

scores = gmm.GMMClassifier(DTR_gauss_pca, LTR, 0.1, [nG], 0.01, '').getScores(DTE_gauss_pca)

print(f'bayes threshold: {utils.getBayesThreshold(0.2, 1, 1)} - best_threshold: {best_threshold_gmm_02}')
utils.get_metrics(scores, LTE, 0.2, 1, 1)
print(f'bayes threshold: {utils.getBayesThreshold(0.5, 1, 1)} - best_threshold: {best_threshold_gmm_05}')
utils.get_metrics(scores, LTE, 0.5, 1, 1)
print(f'bayes threshold: {utils.getBayesThreshold(0.8, 1, 1)} - best_threshold: {best_threshold_gmm_08}')
utils.get_metrics(scores, LTE, 0.8, 1, 1)

utils.get_metrics_threshold(scores, LTE, best_threshold_gmm_02, 0.2, 1, 1)
utils.get_metrics_threshold(scores, LTE, best_threshold_gmm_05, 0.5, 1, 1)
utils.get_metrics_threshold(scores, LTE, best_threshold_gmm_08, 0.8, 1, 1)

#
#

print('--------------------------------------------')
print('GMM')

nG = 128

scores = gmm.GMMClassifier(DTR_pca, LTR, 0.1, [nG], 0.01, 'tied').getScores(DTE_pca)

print(f'bayes threshold: {utils.getBayesThreshold(0.2, 1, 1)} - best_threshold: {best_threshold_gmm_tied_02}')
utils.get_metrics(scores, LTE, 0.2, 1, 1)
print(f'bayes threshold: {utils.getBayesThreshold(0.5, 1, 1)} - best_threshold: {best_threshold_gmm_tied_05}')
utils.get_metrics(scores, LTE, 0.5, 1, 1)
print(f'bayes threshold: {utils.getBayesThreshold(0.8, 1, 1)} - best_threshold: {best_threshold_gmm_tied_08}')
utils.get_metrics(scores, LTE, 0.8, 1, 1)

utils.get_metrics_threshold(scores, LTE, best_threshold_gmm_tied_02, 0.2, 1, 1)
utils.get_metrics_threshold(scores, LTE, best_threshold_gmm_tied_05, 0.5, 1, 1)
utils.get_metrics_threshold(scores, LTE, best_threshold_gmm_tied_08, 0.8, 1, 1)

#