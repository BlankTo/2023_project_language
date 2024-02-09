import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

import ML_lib.utils as utils
import ML_lib.plot_lib as pltl
import ML_lib.preprocessing as pre


D, L, label_dict = utils.csv_to_npy('data_raw/iris.csv')
print(D.shape)
print(L.shape)
print(label_dict)

m_pca = 2
D_centered = pre.Normalizer(D).transform(D)


PCA = pre.PCA(D_centered)
print('check PCA against prof')
print(abs(PCA.PCA_mat - np.load('labs/lab03/IRIS_PCA_matrix_m4.npy')))
print(abs(PCA.PCA_mat - np.load('labs/lab03/IRIS_PCA_matrix_m4.npy')).sum())
D_pca = PCA.transform(D, m_pca)
pltl.plotSparse(D_pca, L, label_dict= label_dict, title= 'PCA ' + str(m_pca))

old_PCA = PCA
PCA = pre.PCA(D_centered)
print('chack PCA svd against norm')
print(abs(PCA.PCA_mat - old_PCA.PCA_mat))
print(abs(PCA.PCA_mat - old_PCA.PCA_mat).sum())
D_pca = PCA.transform(D, m_pca, svd= True)
pltl.plotSparse(D_pca, L, label_dict= label_dict, title= 'PCA ' + str(m_pca))


LDA = pre.LDA(D, L)
D_lda = LDA.transform(D)
print('check LDA against prof')
print(abs(LDA.W - np.load('labs/lab03/IRIS_LDA_matrix_m2.npy')))
print(abs(LDA.W - np.load('labs/lab03/IRIS_LDA_matrix_m2.npy')).sum())
pltl.plotSparse(D_lda, L, label_dict= label_dict, title= 'LDA')
