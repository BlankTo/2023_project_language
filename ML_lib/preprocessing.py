import numpy as np
from scipy.linalg import eigh as scipy_eigh
import scipy.stats as stat

from ML_lib.utils import to_col, get_var

class NoTransform:
    def __init__(self, D, L= None): pass
    def transform(self, X): return X
    def __str__(self): return str('NoTransform')
    def __repr__(self): return str('NoTransform')
    def getName(): return str('NoTransform')


class Normalizer:
    def __init__(self, D, L= None):
        self.mu = to_col(D.mean(axis= 1))

    def transform(self, D): return D - self.mu

    def __str__(self): return str('Normalizer')
    def __repr__(self): return str('Normalizer')
    def getName(): return str('Normalizer')

class Standardizer:
    def __init__(self, D, L= None):
        self.mu = to_col(D.mean(axis= 1))
        self.std = to_col(np.sqrt(np.array([get_var(D[i]) for i in range(D.shape[0])])))

    def transform(self, D): return (D - self.mu) / self.std

    def __str__(self): return str('Standardizer')
    def __repr__(self): return str('Standardizer')
    def getName(): return str('Standardizer')

class Gaussianizer:
    def __init__(self, D, L= None):
        self.D = D

    def transform(self, X):
        M, N = X.shape
        X_new = np.zeros((M, N))
        for m in range(M):
            for i in range(N):
                X_new[m, i] += ((self.D[m, :] < X[m, i]).sum() + 1) / (self.D.shape[1] + 2)
        X_new = stat.norm.ppf(X_new)
        return X_new
    
    def __str__(self): return str('Gaussianizer')
    def __repr__(self): return str('Gaussianizer')
    def getName(): return str('Gaussianizer')


class PCA:
    def __init__(self, D, L= None):
        if abs(D.mean().sum()) > 1e-10: D_in = Normalizer(D).transform(D)
        else: D_in = D.copy()
        self.cov = (D_in@D_in.T) / D_in.shape[1]
        self.eig_val, self.eig_vect = np.linalg.eigh(self.cov)
        self.PCA_mat = self.eig_vect[:, ::-1]
        

    def transform(self, D, m= False, svd= False):
        if svd:
            self.eig_vect, self.eig_val, _ = np.linalg.svd(self.cov)
            self.PCA_mat = self.eig_vect
        if m: return self.PCA_mat[:, :m].T @ D
        else: return self.PCA_mat.T @ D

    def getPCAmat(self, m= False): return self.PCA_mat if not m else self.PCA_mat[:, :m]

    def getEig(self): return self.eig_val[::-1]

    def __str__(self): return str('PCA')
    def __repr__(self): return str('PCA')
    def getName(): return str('PCA')


## LDA

class LDA:
    def __init__(self, D, L):
        M, N = D.shape
        self.K = len(set(L))
        Sb = np.zeros((M, M), dtype= np.float32)
        Sw = np.zeros((M, M), dtype= np.float32)
        mu = np.array(D.mean(axis= 1)).reshape(M, 1)
        for c in range(self.K):
            Dc = D[:, L==c]
            muc = Dc.mean(axis= 1).reshape(M, 1)
            mid = Dc - muc
            Sw += mid @ mid.T
            mcmu = muc - mu
            Sb += Dc.shape[1] * mcmu @ mcmu.T
        self.Sb = Sb / N
        self.Sw = Sw / N

    def transform(self, D, m= False, whitening= False):
        if not whitening:
            try:
                s, U = scipy_eigh(self.Sb, self.Sw)
                self.W = U[:, ::-1][:, :self.K-1]
            except:
                print('whiten anyway')
                whitening = True

        if whitening:
            U, s, _ = np.linalg.svd(self.Sw)
            P1 = U @ np.diag(1.0/(s**0.5)) @ U.T
            Sbt = P1 @ self.Sb @ P1.T
            eig_val, eig_vect = scipy_eigh(Sbt)
            P2 = eig_vect[:, ::-1][:, :self.K-1]
            self.W = -(P1.T @ P2)

        if not m or m > self.K - 1: m = self.K - 1
        return self.W[:, :m].T @ D

    def get_basis(self, whitening= False):
        if not whitening:
            try:
                s, U = scipy_eigh(self.Sb, self.Sw)
                self.W = U[:, ::-1][:, :self.K-1]
            except:
                print('whiten anyway')
                whitening = True

        if whitening:
            U, s, _ = np.linalg.svd(self.Sw)
            P1 = U @ np.diag(1.0/(s**0.5)) @ U.T
            Sbt = P1 @ self.Sb @ P1.T
            eig_val, eig_vect = scipy_eigh(Sbt)
            P2 = eig_vect[:, ::-1][:, :self.K-1]
            self.W = -(P1.T @ P2)

        UW, _, _ = np.linalg.svd(self.W)
        return np.copy(UW[:, 0:self.K-1])
    
    def __str__(self): return str('LDA')
    def __repr__(self): return str('LDA')
    def getName(): return str('LDA')
