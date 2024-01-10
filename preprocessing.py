import numpy as np
import scipy.linalg as lng
#import numpy.linalg as np_lng
import scipy.stats as stat

import common as com

def normalize(D):
    mu = com.to_col(D.mean(axis= 1))
    return D - mu, mu


def standardize(D):
    mu = com.to_col(D.mean(axis= 1))
    std = com.to_col(np.sqrt(np.array([com.get_var(D[i]) for i in range(D.shape[0])])))
    return (D - mu) / std, mu, std


def gaussianize(X, DTR):
    M, N = X.shape
    NTR = DTR.shape[1]
    X_new = np.zeros(X.shape)
    for m in range(M):
        for i in range(N):
            X_new[m, i] += ((DTR[m, :] < X[m, i]).sum() + 1) / (NTR + 2)
    X_new = stat.norm.ppf(X_new)
    return X_new


def get_PCA_matrix(D, m, retVal= False):
    cov = (D@D.T) / D.shape[1]
    eig_val, eig_vect = np.linalg.eigh(cov)
    if retVal: return np.copy(eig_vect[:, ::-1][:, 0:m]), np.copy(eig_val[::-1][0:m])
    return np.copy(eig_vect[:, ::-1][:, 0:m])


def get_LDA_matrix(D, L, N_class, m, whitening= True, basis= False):
    M, N = D.shape
    mu = np.array(D.mean(axis= 1)).reshape(M, 1)
    Sb = np.zeros((M, M), dtype= np.float32)
    Sw = np.zeros((M, M), dtype= np.float32)
    for c in range(N_class):
        Dc = D[:, L==c]
        mc = Dc.mean(axis= 1).reshape(M, 1)
        mcmu = mc - mu
        Sb += Dc.shape[1] * mcmu @ mcmu.T
        mid = Dc - mc
        Sw += mid @ mid.T
    Sb = Sb / N
    Sw = Sw / N

    if not whitening:
        try:
            s, U = lng.eigh(Sb, Sw)
            W = U[:, ::-1][:, 0:m]
        except:
            print('whiten anyway')
            whitening = True

    if whitening:
        U, s, _ = np.linalg.svd(Sw)
        P1 = U @ np.diag(1.0/(s**0.5)) @ U.T
        Sbt = P1 @ Sb @ P1.T
        eig_val, eig_vect = lng.eigh(Sbt)
        P2 = eig_vect[:, ::-1][:, 0:m]
        W = -(P1.T @ P2)

    if not basis: return np.copy(W)
    else:
        UW, _, _ = np.linalg.svd(W)
        return np.copy(UW[:, 0:m])