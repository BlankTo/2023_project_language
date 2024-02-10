import os
import sys
import numpy as np
from sklearn.cluster import KMeans

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import ML_lib.preprocessing as pre
import ML_lib.plot_lib as pltl
import ML_lib.utils as utils
from sklearn.model_selection import train_test_split

D_BASE = np.load('data_npy\\DTR_language.npy')
L = np.load('data_npy\\LTR_language.npy')

N_CLASS = 2
M_BASE, NTR = D_BASE.shape

import numpy.linalg as np_lng

class EuclideanClassifier:
    def __init__(self, D_train, L_train):
        self.n_class = len(set(L_train))
        self.class_means = np.array([D_train[:, L_train==c].mean(axis= 1) for c in range(self.n_class)])

    def predict(self, D_test):
        assigned_L = []
        for xt in D_test.T:
            distances = np.array([(np_lng.norm(xt-c_mean))**2 for c_mean in self.class_means])
            assigned_L.append(distances.argsort()[0])
        return np.array(assigned_L)
    
    def getScores(self, D_test):
        assigned_L = []
        scores_L = []
        for xt in D_test.T:
            distances = np.array([(np_lng.norm(xt-c_mean))**2 for c_mean in self.class_means])
            assigned_L.append(distances.argsort()[0])
            scores_L.append(distances[0] - distances[1])
        return scores_L


def euclidean_classifier(D_train, L_train, D_test, retScores= False):
    n_class = len(set(L_train))
    assigned_L = []
    if retScores and n_class == 2: scores_L = []
    class_means = np.array([D_train[:, L_train==c].mean(axis= 1) for c in range(n_class)])
    for xt in D_test.T:
        distances = np.array([(np_lng.norm(xt-c_mean))**2 for c_mean in class_means])
        assigned_L.append(distances.argsort()[0])
        if retScores and n_class == 2: scores_L.append(distances[0] - distances[1])
    if retScores and n_class == 2: return scores_L
    return np.array(assigned_L)

print('Euclidean')
utils.cross_validation(D_BASE, L, 100, EuclideanClassifier, [[]], print_err= True, progress= False, prepro= [
    [(pre.NoTransform, [])],
    [(pre.Standardizer, [])],
    [(pre.Standardizer, []), (pre.PCA, [5])],
    [(pre.Standardizer, []), (pre.PCA, [4])],
    [(pre.Standardizer, []), (pre.PCA, [3])],
    [(pre.LDA, [1])],
    #[(pre.Standardizer, []), (pre.LDA, [])],
    #[(pre.Standardizer, []), (pre.PCA, [5]), (pre.LDA, [])],
    #[(pre.Standardizer, []), (pre.PCA, [4]), (pre.LDA, [])],
    #[(pre.Standardizer, []), (pre.PCA, [3]), (pre.LDA, [])],
    ])


#class MahalanobisClassifier:
#    def __init__(self, DTR, LTR):
#        self.DTR = DTR
#        self.LTR = LTR
#        self.K = len(set(LTR))
#    
#    def predict(self, DTE):
#        M, NTE = DTE.shape
#        MD = np.zeros((self.K, NTE))
#        for c in range(self.K):
#            DTR_c = self.DTR[:, self.LTR==c]
#            DTE_mu_c = DTE - DTR_c.mean(axis=1).reshape(M, 1)
#            for i in range(NTE): MD[c, i] += np.sqrt(DTE_mu_c[:, i].T @ np_lng.inv(utils.get_covMatrix(DTR_c)) @ DTE_mu_c[:, i])
#        assigned = np.argmin(MD, axis=0)
#        return assigned
#    
#    def getScores(self, DTE):
#        M, NTE = DTE.shape
#        MD = np.zeros((self.K, NTE))
#        for c in range(self.K):
#            DTR_c = self.DTR[:, self.LTR==c]
#            DTE_mu_c = DTE - DTR_c.mean(axis=1).reshape(M, 1)
#            for i in range(NTE): MD[c, i] += np.sqrt(DTE_mu_c[:, i].T @ np_lng.inv(utils.get_covMatrix(DTR_c)) @ DTE_mu_c[:, i])
#        return MD[0, :] - MD[1, :]
#        
#
#def mahalanobis_classifier(DTR, LTR, DTE, retScores= False):
#    M, NTE = DTE.shape
#    K = len(set(LTR))
#    MD = np.zeros((K, NTE))
#    for c in range(K):
#        DTR_c = DTR[:, LTR==c]
#        DTE_mu_c = DTE - DTR_c.mean(axis=1).reshape(M, 1)
#        for i in range(NTE): MD[c, i] += np.sqrt(DTE_mu_c[:, i].T @ np_lng.inv(utils.get_covMatrix(DTR_c)) @ DTE_mu_c[:, i])
#    if retScores and K == 2: return MD[0, :] - MD[1, :]
#    assigned = np.argmin(MD, axis=0)
#    return assigned
#
#print('Mahalanobis')
#utils.cross_validation(D_BASE, L, 100, MahalanobisClassifier, [[]], print_err= True, progress= False, prepro= [
#    #[(pre.NoTransform, [])],
#    #[(pre.Standardizer, [])],
#    #[(pre.Standardizer, []), (pre.PCA, [5])],
#    #[(pre.Standardizer, []), (pre.PCA, [4])],
#    #[(pre.Standardizer, []), (pre.PCA, [3])],
#    [(pre.LDA, [1])],
#    #[(pre.Standardizer, []), (pre.LDA, [])],
#    #[(pre.Standardizer, []), (pre.PCA, [5]), (pre.LDA, [])],
#    #[(pre.Standardizer, []), (pre.PCA, [4]), (pre.LDA, [])],
#    #[(pre.Standardizer, []), (pre.PCA, [3]), (pre.LDA, [])],
#    ])
#

###########################################################################
###########################################################################
###########################################################################


#class KmeansClassifier:
#    def __init__(self, D_train, L_train= None, n_clust= 10, n_init= 'auto', alpha_threshold= 0.):
#        km = KMeans(n_clusters= n_clust, n_init= n_init)
#
#        D_train_class_0 = (D_train[:, L_train == 0]).T
#
#        self.model = km.fit(D_train_class_0)
#        class_0_dist = np.min(self.model.transform(D_train_class_0), axis= 1)
#
#        min_loss_c0 = np.min(class_0_dist)
#        max_loss_c0 = np.max(class_0_dist)
#        self.threshold = max_loss_c0 + alpha_threshold * (max_loss_c0 - min_loss_c0)
#
#    def getScores(self, D_test): return np.min(self.model.transform(D_test.T), axis= 1)
#
#    def predict(self, D_test): return (self.getScores(D_test) > self.threshold).astype(int)
#
#
#
#model_params = [[n_c, n_i, a_t] for n_c in [10] for n_i in ['auto'] for a_t in [1.]]
#
#print('k-means 1')
#utils.cross_validation(D_BASE, L, 100, KmeansClassifier, model_params, progress= False, prepro= [
#    #[(pre.NoTransform, [])],
#    #[(pre.Standardizer, [])],
#    #[(pre.Standardizer, []), (pre.PCA, [5])],
#    #[(pre.Standardizer, []), (pre.PCA, [4])],
#    #[(pre.Standardizer, []), (pre.PCA, [3])],
#    [(pre.LDA, [1])],
#    #[(pre.Standardizer, []), (pre.LDA, [])],
#    #[(pre.Standardizer, []), (pre.PCA, [5]), (pre.LDA, [])],
#    #[(pre.Standardizer, []), (pre.PCA, [4]), (pre.LDA, [])],
#    #[(pre.Standardizer, []), (pre.PCA, [3]), (pre.LDA, [])],
#    ], print_err= True)
#
#class KmeansClassifier_2:
#    def __init__(self, D_train, L_train= None, n_clust= 10, n_init= 'auto'):
#        km_c0 = KMeans(n_clusters= n_clust, n_init= n_init)
#        km_c1 = KMeans(n_clusters= n_clust, n_init= n_init)
#
#        D_train_class_0 = (D_train[:, L_train == 0]).T
#        D_train_class_1 = (D_train[:, L_train == 1]).T
#
#        self.model_c0 = km_c0.fit(D_train_class_0)
#        self.model_c1 = km_c1.fit(D_train_class_1)
#
#    def getScores(self, D_test):
#        D_t = D_test.T
#        distances = np.vstack([np.min(self.model_c0.transform(D_t), axis= 1), np.min(self.model_c1.transform(D_t), axis= 1)])
#        return (distances[0] - distances[1]).reshape(D_t.shape[0])
#
#    def predict(self, D_test): return (self.getScores(D_test) > 0).astype(int)
#
#model_params = [[n_c, n_i] for n_c in [10, 100] for n_i in ['auto'] for a_t in [0.5]]
#
#print('k-means 2')
#utils.cross_validation(D_BASE, L, 100, KmeansClassifier_2, model_params, progress= False, prepro= [
#    #[(pre.NoTransform, [])],
#    #[(pre.Standardizer, [])],
#    #[(pre.Standardizer, []), (pre.PCA, [5])],
#    #[(pre.Standardizer, []), (pre.PCA, [4])],
#    #[(pre.Standardizer, []), (pre.PCA, [3])],
#    [(pre.LDA, [1])],
#    #[(pre.Standardizer, []), (pre.LDA, [])],
#    #[(pre.Standardizer, []), (pre.PCA, [5]), (pre.LDA, [])],
#    #[(pre.Standardizer, []), (pre.PCA, [4]), (pre.LDA, [])],
#    #[(pre.Standardizer, []), (pre.PCA, [3]), (pre.LDA, [])],
#    ], print_err= True)