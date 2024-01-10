import numpy as np
from scipy.optimize import fmin_l_bfgs_b

from ML_lib.utils import to_col, to_row

def get_Ldual(H, prime= True):
    def L(al):
        al = to_col(al)
        Ha = H @ al
        return ( 0.5 * (al.T @ Ha).ravel() - al.sum())
    def L_gradL(al):
        al = to_col(al)
        Ha = H @ al
        return (( 0.5 * (al.T @ Ha).ravel() - al.sum()), (Ha.ravel() - 1))
    if prime: return L_gradL
    else: return L

def get_H_linear(D, Z):
    ZD = D * Z
    return ZD.T @ ZD

def get_kernel_poly(c, d, eps):
    # c ->
    # d ->
    # eps -> constant that substitutes the bias, that in this formulation is removed
    def kernel_poly(X1, X2):
        return ((X1.T @ X2) + c)**d + eps
    return kernel_poly

def get_kernel_RBF(lam, eps):
    # lam -> 
    # eps -> constant that substitutes the bias, that in this formulation is removed

    def kernel_RBF(X1, X2):                                                                                 #return np.exp(-lam * (to_col((X1**2).sum(0)) + to_row((X2**2).sum(0)) - 2 * (X1.T @ X2))) #prof(? -> cause it doesn't work)
        return np.array([[np.exp(-lam * np.linalg.norm(x1 - x2)**2) for x2 in X2.T] for x1 in X1.T]) + eps
    return kernel_RBF

def get_H_kernel(D, Z, kernel_function):
    Zc = to_col(Z)
    return (Zc @ Zc.T) * (kernel_function(D, D))


def SVM(DTR, LTR, DTE, LTE, K, C, kernel_function= None, pT= None, maxiter= 15000, retModel= False, retScores = False, retAssigned= False):
    # K -> larger K mean less bias b regularization (b regularization is sub-optimal) but also means more iterations
    # C -> tradeoff between w regularization and loss
    M, NTR = DTR.shape

    Z = LTR * 2 - 1
    DTRb = np.vstack([DTR, K * np.ones(NTR)]) if kernel_function is None else DTR
    if pT is not None:
        if type(pT) == float: pT = [1-pT, pT]
        pTrue = LTR.sum() / NTR
        pFalse = 1 - pTrue
        bounds = np.array([(0, C * pT[LTR[i]] / (pTrue if LTR[i]==1 else pFalse)) for i in range(NTR)])
    else: bounds = np.array([(0, C) for _ in range(NTR)])
    H = get_H_linear(DTRb, Z) if kernel_function is None else get_H_kernel(DTRb, Z, kernel_function)
    al, dual_loss, _ = fmin_l_bfgs_b(get_Ldual(H), np.zeros(NTR), bounds= bounds, maxiter= maxiter)#, factr= 1.0
    #print('%.7f' % -dual_loss)
    if retModel: return al, kernel_function

    if kernel_function is None:
        #w = (al * Z * DTRb).sum(axis= 1)
        
        #idx_al = np.argwhere(al)
        #idx_al = idx_al.reshape(idx_al.size,)
        #w = (al[idx_al] * Z[idx_al] * DTRb[:, idx_al]).sum(axis= 1)

        w = np.array([al[i] * Z[i] * DTRb[:, i] for i in np.argwhere(al)]).sum(axis= 0)

        #primal_loss = 0.5 * pow(np.linalg.norm(w), 2) + C * np.array([max(0, 1 - Z[i]*(w.T @ DTRb[:, i])) for i in range(NTR)], dtype= np.ndarray).sum()
        #print('primal: %.6f  -  dual: %.6f  -  gap: %f' % (primal_loss, -dual_loss, primal_loss + dual_loss))
        w1 = w[:-1]
        b1 = w[-1]
        scores = w1.T @ DTE + K * b1
    else:
        #scores = (al * Z * kernel_function(DTR, DTE).T).sum(axis= 1)
        
        idx_al = np.argwhere(al)
        idx_al = idx_al.reshape(idx_al.size,)
        scores = (al[idx_al] * Z[idx_al] * kernel_function(DTR[:, idx_al], DTE).T).sum(axis= 1)

        #scores = np.array([al[i] * Z[i] * kernel_function(DTR[:, i], DTE).T for i in np.argwhere(al)[:, 0]]).sum(axis= 0)
    
    if retScores: return scores.reshape(DTE.shape[1], )

    assigned = scores>0
    
    if retAssigned: return assigned.reshape(DTE.shape[1],)

    correct = (LTE == assigned).sum()
    NTE = LTE.size
    return (NTE - correct) * 100 / NTE

class SupportVectorMachine:
    def __init__(self, DTR, LTR, K= None, C= None, kernel_function= None, pT= None, maxiter= 15000, threshold= None):
        if (K is None) or (C is None): print('K or C not passed -- class SupportVectorMachine')
        M, NTR = DTR.shape
        self.K = K
        self.C = C
        self.kernel_function = kernel_function

        self.threshold = threshold
        self.Z = LTR * 2 - 1
        
        DTRb = np.vstack([DTR, self.K * np.ones(NTR)]) if kernel_function is None else DTR
        self.DTRb = DTRb

        if pT is not None:
            pTrue = LTR.sum() / NTR
            pFalse = 1 - pTrue
            bounds = np.array([(0, C * pT[LTR[i]] / (pTrue if LTR[i]==1 else pFalse)) for i in range(NTR)])
        else: bounds = np.array([(0, C) for _ in range(NTR)])
        H = get_H_linear(DTRb, self.Z) if self.kernel_function is None else get_H_kernel(DTRb, self.Z, self.kernel_function)
        self.al, dual_loss, _ = fmin_l_bfgs_b(get_Ldual(H), np.zeros(NTR), bounds= bounds, maxiter= maxiter)

    def getScores(self, DTE):

        if np.argwhere(self.al).shape[0] == 0:
            print('error on np.argwhere in getScores of SupportVectorMachine\n\n')
            print(f'params -> K:{self.K} - C: {self.C}\n\n')
            return np.zeros(DTE.shape[1])

        if self.kernel_function is None:

            w = np.array([self.al[i] * self.Z[i] * self.DTRb[:, i] for i in np.argwhere(self.al)]).sum(axis= 0)

            w1 = w[:-1]
            b1 = w[-1]

            return (w1.T @ DTE + self.K * b1).reshape(DTE.shape[1])
        else:
            
            idx_al = np.argwhere(self.al)
            idx_al = idx_al.reshape(idx_al.size,)
            return (self.al[idx_al] * self.Z[idx_al] * self.kernel_function(self.DTRb[:, idx_al], DTE).T).sum(axis= 1)
        
    def predict(self, DTE):
        if self.threshold is None: return self.getScores(DTE)>0
        else: return self.getScores(DTE) > self.threshold



def use_SVM(DTR, LTR, DTE, K, al, kernel_function= False):
    Z = LTR * 2 - 1
    if not kernel_function:
        DTRb = np.vstack([DTR, K * np.ones(DTR.shape[1])])
        DTEb = np.vstack([DTE, K * np.ones(DTE.shape[1])])
        w = np.array([al[i] * Z[i] * DTRb[:, i] for i in np.argwhere(al)]).sum(axis= 0)
        w1 = w[:-1]
        b1 = w[-1]
        scores = w1.T @ DTE + b1
    else:
        scores = np.array([al[i] * Z[i] * kernel_function(DTR[:, i], DTE).T for i in np.argwhere(al)[:, 0]]).sum(axis= 0)

    assigned = scores>0
    return assigned