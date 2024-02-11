import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import ML_lib.gaussian as gau
import ML_lib.utils as utils
import ML_lib.preprocessing as pre

D_BASE = np.load('data_npy\\DTR_language.npy')
L = np.load('data_npy\\LTR_language.npy')

N_CLASS = 2
M_BASE, NTR = D_BASE.shape

DTR_base, LTR, DTE_base, LTE = utils.shuffle_and_divide(D_BASE, L, 2/3)

print('No Prepro')
print('Base')
scores = gau.GaussianClassifier(DTR_base, LTR, version= 'base').getScores(DTE_base)
print('0.2')
utils.get_metrics(scores, LTE, 0.2, 1, 1, print_err= True)
print('0.5')
utils.get_metrics(scores, LTE, 0.5, 1, 1, print_err= True)
print('0.8')
utils.get_metrics(scores, LTE, 0.8, 1, 1, print_err= True)

print('Naive')
scores = gau.GaussianClassifier(DTR_base, LTR, version= 'naive').getScores(DTE_base)
print('0.2')
utils.get_metrics(scores, LTE, 0.2, 1, 1, print_err= True)
print('0.5')
utils.get_metrics(scores, LTE, 0.5, 1, 1, print_err= True)
print('0.8')
utils.get_metrics(scores, LTE, 0.8, 1, 1, print_err= True)

print('Tied')
scores = gau.GaussianClassifier(DTR_base, LTR, version= 'tied').getScores(DTE_base)
print('0.2')
utils.get_metrics(scores, LTE, 0.2, 1, 1, print_err= True)
print('0.5')
utils.get_metrics(scores, LTE, 0.5, 1, 1, print_err= True)
print('0.8')
utils.get_metrics(scores, LTE, 0.8, 1, 1, print_err= True)

print('Naive-Tied')
scores = gau.GaussianClassifier(DTR_base, LTR, version= 'naive-tied').getScores(DTE_base)
print('0.2')
utils.get_metrics(scores, LTE, 0.2, 1, 1, print_err= True)
print('0.5')
utils.get_metrics(scores, LTE, 0.5, 1, 1, print_err= True)
print('0.8')
utils.get_metrics(scores, LTE, 0.8, 1, 1, print_err= True)

print('\nPCA 5')
pca = pre.PCA(DTR_base)
DTR = pca.transform(DTR_base, 5)
DTE = pca.transform(DTE_base, 5)
print('Base')
scores = gau.GaussianClassifier(DTR, LTR, version= 'base').getScores(DTE)
print('0.2')
utils.get_metrics(scores, LTE, 0.2, 1, 1, print_err= True)
print('0.5')
utils.get_metrics(scores, LTE, 0.5, 1, 1, print_err= True)
print('0.8')
utils.get_metrics(scores, LTE, 0.8, 1, 1, print_err= True)

print('Naive')
scores = gau.GaussianClassifier(DTR, LTR, version= 'naive').getScores(DTE)
print('0.2')
utils.get_metrics(scores, LTE, 0.2, 1, 1, print_err= True)
print('0.5')
utils.get_metrics(scores, LTE, 0.5, 1, 1, print_err= True)
print('0.8')
utils.get_metrics(scores, LTE, 0.8, 1, 1, print_err= True)

print('Tied')
scores = gau.GaussianClassifier(DTR, LTR, version= 'tied').getScores(DTE)
print('0.2')
utils.get_metrics(scores, LTE, 0.2, 1, 1, print_err= True)
print('0.5')
utils.get_metrics(scores, LTE, 0.5, 1, 1, print_err= True)
print('0.8')
utils.get_metrics(scores, LTE, 0.8, 1, 1, print_err= True)

print('Naive-Tied')
scores = gau.GaussianClassifier(DTR, LTR, version= 'naive-tied').getScores(DTE)
print('0.2')
utils.get_metrics(scores, LTE, 0.2, 1, 1, print_err= True)
print('0.5')
utils.get_metrics(scores, LTE, 0.5, 1, 1, print_err= True)
print('0.8')
utils.get_metrics(scores, LTE, 0.8, 1, 1, print_err= True)

print('\nGaussian')
gauss = pre.Gaussianizer(DTR_base, LTR)
DTR = gauss.transform(DTR_base)
DTE = gauss.transform(DTE_base)
print('Base')
scores = gau.GaussianClassifier(DTR, LTR, version= 'base').getScores(DTE)
print('0.2')
utils.get_metrics(scores, LTE, 0.2, 1, 1, print_err= True)
print('0.5')
utils.get_metrics(scores, LTE, 0.5, 1, 1, print_err= True)
print('0.8')
utils.get_metrics(scores, LTE, 0.8, 1, 1, print_err= True)

print('Naive')
scores = gau.GaussianClassifier(DTR, LTR, version= 'naive').getScores(DTE)
print('0.2')
utils.get_metrics(scores, LTE, 0.2, 1, 1, print_err= True)
print('0.5')
utils.get_metrics(scores, LTE, 0.5, 1, 1, print_err= True)
print('0.8')
utils.get_metrics(scores, LTE, 0.8, 1, 1, print_err= True)

print('Tied')
scores = gau.GaussianClassifier(DTR, LTR, version= 'tied').getScores(DTE)
print('0.2')
utils.get_metrics(scores, LTE, 0.2, 1, 1, print_err= True)
print('0.5')
utils.get_metrics(scores, LTE, 0.5, 1, 1, print_err= True)
print('0.8')
utils.get_metrics(scores, LTE, 0.8, 1, 1, print_err= True)

print('Naive-Tied')
scores = gau.GaussianClassifier(DTR, LTR, version= 'naive-tied').getScores(DTE)
print('0.2')
utils.get_metrics(scores, LTE, 0.2, 1, 1, print_err= True)
print('0.5')
utils.get_metrics(scores, LTE, 0.5, 1, 1, print_err= True)
print('0.8')
utils.get_metrics(scores, LTE, 0.8, 1, 1, print_err= True)

print('\nGaussian PCA 5')
gauss = pre.Gaussianizer(DTR_base, LTR)
DTR = gauss.transform(DTR_base)
DTE = gauss.transform(DTE_base)
pca = pre.PCA(DTR)
DTR = pca.transform(DTR, 5)
DTE = pca.transform(DTE, 5)
print('Base')
scores = gau.GaussianClassifier(DTR, LTR, version= 'base').getScores(DTE)
print('0.2')
utils.get_metrics(scores, LTE, 0.2, 1, 1, print_err= True)
print('0.5')
utils.get_metrics(scores, LTE, 0.5, 1, 1, print_err= True)
print('0.8')
utils.get_metrics(scores, LTE, 0.8, 1, 1, print_err= True)

print('Naive')
scores = gau.GaussianClassifier(DTR, LTR, version= 'naive').getScores(DTE)
print('0.2')
utils.get_metrics(scores, LTE, 0.2, 1, 1, print_err= True)
print('0.5')
utils.get_metrics(scores, LTE, 0.5, 1, 1, print_err= True)
print('0.8')
utils.get_metrics(scores, LTE, 0.8, 1, 1, print_err= True)

print('Tied')
scores = gau.GaussianClassifier(DTR, LTR, version= 'tied').getScores(DTE)
print('0.2')
utils.get_metrics(scores, LTE, 0.2, 1, 1, print_err= True)
print('0.5')
utils.get_metrics(scores, LTE, 0.5, 1, 1, print_err= True)
print('0.8')
utils.get_metrics(scores, LTE, 0.8, 1, 1, print_err= True)

print('Naive-Tied')
scores = gau.GaussianClassifier(DTR, LTR, version= 'naive-tied').getScores(DTE)
print('0.2')
utils.get_metrics(scores, LTE, 0.2, 1, 1, print_err= True)
print('0.5')
utils.get_metrics(scores, LTE, 0.5, 1, 1, print_err= True)
print('0.8')
utils.get_metrics(scores, LTE, 0.8, 1, 1, print_err= True)

utils.cross_validation(D_BASE, L, 10, gau.GaussianClassifier, [
    [],
    ['naive'],
    ['tied'],
    ['naive-tied']
    ], progress= True, effective= [0.2, 0.5, 0.8], save= True, filename= 'results\\cross_val_gau.txt', prepro= [

    [(pre.NoTransform, [])],
    [(pre.Standardizer, [])],
    [(pre.Gaussianizer, [])],

    [(pre.Standardizer, []), (pre.PCA, [5])],
    [(pre.Standardizer, []), (pre.PCA, [4])],
    [(pre.Standardizer, []), (pre.PCA, [3])],

    [(pre.Gaussianizer, []), (pre.PCA, [5])],
    [(pre.Gaussianizer, []), (pre.PCA, [4])],
    [(pre.Gaussianizer, []), (pre.PCA, [3])],

    [(pre.LDA, [1])],

    ], print_err= True)

