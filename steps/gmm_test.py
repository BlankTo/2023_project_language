import os
import sys
import numpy as np
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import ML_lib.gmm as gmm
import ML_lib.utils as utils
import ML_lib.preprocessing as pre

DTR_base = np.load('data_npy\\DTR_language.npy')
DTE_base = np.load('data_npy\\DTE_language.npy')
LTR = np.load('data_npy\\LTR_language.npy')
LTE = np.load('data_npy\\LTE_language.npy')


N_CLASS = 2
M_BASE, NTR = DTR_base.shape

#DTR_base, LTR, DTE_base, LTE = utils.shuffle_and_divide(D_BASE, L, 2/3)

scores_all = gmm.GMMClassifier(DTR_base, LTR, 0.1, [16], retAll= True, bound= 0.01, variant= '').getScores_all(DTE_base)
print(f'Raw')

for scores, nG in scores_all:

   print(f'nG: {nG}')

   utils.get_metrics(scores, LTE, 0.2, 1, 1, print_err= True)
   utils.get_metrics(scores, LTE, 0.5, 1, 1, print_err= True)
   utils.get_metrics(scores, LTE, 0.8, 1, 1, print_err= True)

##

# pca = pre.PCA(DTR_base)
# DTR = pca.transform(DTR_base, 5)
# DTE = pca.transform(DTE_base, 5)

# scores_all = gmm.GMMClassifier(DTR, LTR, 0.1, [64], retAll= True, bound= 0.01, variant= '').getScores_all(DTE)

# for scores, nG in scores_all:

#    print(f'nG: {nG}')

#    utils.get_metrics(scores, LTE, 0.2, 1, 1, print_err= True)
#    utils.get_metrics(scores, LTE, 0.5, 1, 1, print_err= True)
#    utils.get_metrics(scores, LTE, 0.8, 1, 1, print_err= True)

# ##

# gauss = pre.Gaussianizer(DTR_base, LTR)
# DTR = gauss.transform(DTR_base)
# DTE = gauss.transform(DTE_base)

# scores_all = gmm.GMMClassifier(DTR, LTR, 0.1, [64], retAll= True, bound= 0.01, variant= '').getScores_all(DTE)

# for scores, nG in scores_all:

#    print(f'nG: {nG}')

#    utils.get_metrics(scores, LTE, 0.2, 1, 1, print_err= True)
#    utils.get_metrics(scores, LTE, 0.5, 1, 1, print_err= True)
#    utils.get_metrics(scores, LTE, 0.8, 1, 1, print_err= True)

# ##

gauss = pre.Gaussianizer(DTR_base, LTR)
DTR = gauss.transform(DTR_base)
DTE = gauss.transform(DTE_base)

pca = pre.PCA(DTR)
DTR = pca.transform(DTR, 5)
DTE = pca.transform(DTE, 5)

scores_all = gmm.GMMClassifier(DTR, LTR, 0.1, [16], retAll= True, bound= 0.01, variant= '').getScores_all(DTE)

for scores, nG in scores_all:

   print(f'Raw {nG}')
   print(f'nG: {nG}')

   utils.get_metrics(scores, LTE, 0.2, 1, 1, print_err= True)
   utils.get_metrics(scores, LTE, 0.5, 1, 1, print_err= True)
   utils.get_metrics(scores, LTE, 0.8, 1, 1, print_err= True)

######
    # cross val
######


# print('Cross val no prepro')
# LTE_cross, scores_all = gmm.cross_validation_gmm(D_BASE, L, 10, None, None, 0, 0.1, 64, 0.01, '')

# for scores, nG in scores_all:

#     print(f'nG: {nG}')
#     utils.get_metrics(scores, LTE_cross, 0.2, 1, 1, print_err= True)
#     utils.get_metrics(scores, LTE_cross, 0.5, 1, 1, print_err= True)
#     utils.get_metrics(scores, LTE_cross, 0.8, 1, 1, print_err= True)

# ##
    
# print('Cross val pca 5')
# LTE_cross, scores_all = gmm.cross_validation_gmm(D_BASE, L, 10, None, [pre.PCA, 5], 0, 0.1, 64, 0.01, '')

# for scores, nG in scores_all:

#     print(f'nG: {nG}')
#     utils.get_metrics(scores, LTE_cross, 0.2, 1, 1, print_err= True)
#     utils.get_metrics(scores, LTE_cross, 0.5, 1, 1, print_err= True)
#     utils.get_metrics(scores, LTE_cross, 0.8, 1, 1, print_err= True)

# ##
    
# print('Cross val gaussianized')
# LTE_cross, scores_all = gmm.cross_validation_gmm(D_BASE, L, 10, pre.Gaussianizer, None, 0, 0.1, 64, 0.01, '')

# for scores, nG in scores_all:

#     print(f'nG: {nG}')
#     utils.get_metrics(scores, LTE_cross, 0.2, 1, 1, print_err= True)
#     utils.get_metrics(scores, LTE_cross, 0.5, 1, 1, print_err= True)
#     utils.get_metrics(scores, LTE_cross, 0.8, 1, 1, print_err= True)

# ##
    
# print('Cross val gaussianized pca 5')
# LTE_cross, scores_all = gmm.cross_validation_gmm(D_BASE, L, 10, pre.Gaussianizer, [pre.PCA, 5], 0, 0.1, 64, 0.01, '')

# for scores, nG in scores_all:

#     print(f'nG: {nG}')
#     utils.get_metrics(scores, LTE_cross, 0.2, 1, 1, print_err= True)
#     utils.get_metrics(scores, LTE_cross, 0.5, 1, 1, print_err= True)
#     utils.get_metrics(scores, LTE_cross, 0.8, 1, 1, print_err= True)

# ##
