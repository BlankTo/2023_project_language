import numpy as np
import scipy.optimize as opt

import My_Lib_ML.common as com

def get_Ldual(H, prime= True):
    def L(al):
        al = com.to_col(al)
        Ha = H @ al
        return ( 0.5 * (al.T @ Ha).ravel() - al.sum())
    def L_gradL(al):
        al = com.to_col(al)
        Ha = H @ al
        return (( 0.5 * (al.T @ Ha).ravel() - al.sum()), (Ha.ravel() - 1))
    if prime: return L_gradL
    else: return L

def get_H_linear(D, Z):
    ZD = D * Z
    return ZD.T @ ZD

def get_kernel_poly(c, d):
    # c -> 
    def kernel_poly(X1, X2):
        return ((X1.T @ X2) + c)**d
    return kernel_poly

def get_kernel_RBF(lam):
    # lam -> 
    def kernel_RBF(X1, X2):
        #prof = np.exp(-lam * (to_col((X1**2).sum(0)) + to_row((X1**2).sum(0)) - 2 * (X1.T @ X2)))
        return np.array([[np.exp(-lam * np.linalg.norm(x1 - x2)**2) for x2 in X2.T] for x1 in X1.T])
    return kernel_RBF

def get_H_kernel(D, Z, eps, kernel_function):
    # eps -> constant that substitutes the bias, that in this formulation is removed
    Zc = com.to_col(Z)
    return (Zc @ Zc.T) * (kernel_function(D, D) + eps)

def SVM(DTR, LTR, DTE, LTE, K, C, kernel_function= False, pT= False, retModel= False, retScores = False, retAssigned= False):
    # K -> larger K mean less bias b regularization (b regularization is sub-optimal) but also means more iterations
    # C -> tradeoff between w regularization and loss
    M, NTR = DTR.shape

    Z = LTR * 2 - 1
    DTRb = np.vstack([DTR, K * np.ones(NTR)]) if not kernel_function else DTR
    if not type(pT) == bool:
        if type(pT) == float: pT = [1-pT, pT]
        pTrue = LTR.sum() / NTR
        pFalse = 1 - pTrue
        #pFalse = (LTR==0).sum() / NTR
        #pTrue = (LTR==1).sum() / NTR
        bounds = np.array([(0, C * pT[LTR[i]] / (pTrue if LTR[i]==1 else pFalse)) for i in range(NTR)])
    else: bounds = np.array([(0, C) for i in range(NTR)])
    H = get_H_linear(DTRb, Z) if not kernel_function else get_H_kernel(DTRb, Z, K**2, kernel_function)
    al, _, _ = opt.fmin_l_bfgs_b(get_Ldual(H), np.zeros(NTR), bounds= bounds, factr= 1.0)

    if retModel: return al, kernel_function

    if not kernel_function:
        w = DTRb @ (com.to_col(al) * com.to_col(Z))
        w1 = w[:-1]
        b1 = w[-1]
        scores = w1.T @ DTE + b1
    else:
        scores = (al * Z * kernel_function(DTR, DTE).T).sum(axis= 1)
        #scores = np.array([al[i]*Z[i]*kernel_function(DTR[:, i], DTE[:, i]) for i in range(DTE.shape[1])])
    
    if retScores: return scores.reshape(DTE.shape[1], )

    assigned = scores>0
    
    if retAssigned: return assigned.reshape(DTE.shape[1],)

    correct = (LTE == assigned).sum()
    NTE = LTE.size
    return correct

def use_SVM(DTR, LTR, DTE, K, al, kernel_function= False):
    Z = LTR * 2 - 1
    if not kernel_function:
        DTRb = np.vstack([DTR, K * np.ones(DTR.shape[1])])
        DTEb = np.vstack([DTE, K * np.ones(DTE.shape[1])])
        w = (al * Z * DTRb).sum(axis= 1)
        #w = DTRb @ (to_col(al) * to_col(Z))
        w1 = w[:-1]
        b1 = w[-1]
        scores = w1.T @ DTE + b1
    else:
        scores = (al * Z * kernel_function(DTR, DTE).T).sum(axis= 1)

    assigned = scores>0
    return assigned