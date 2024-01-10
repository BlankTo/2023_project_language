import numpy as np
import scipy.special
import scipy.optimize as opt

import My_Lib_ML.common as com

def getCrossEntropy_binary_linear(D, L, hyp, pT= False):
    Z = L * 2 - 1
    M, N = D.shape

    def R(params):
        w = params[:-1]
        b = params[-1]
        summ_0 = (np.logaddexp(0, -Z[L==0]*(w.T @ D[:, L==0] + b))).sum()
        summ_1 = (np.logaddexp(0, -Z[L==1]*(w.T @ D[:, L==1] + b))).sum()
        if type(pT) == bool: summ = (hyp * np.linalg.norm(w)**2 / 2) + (summ_0 + summ_1) / N
        else : summ = (hyp * np.linalg.norm(w)**2 / 2) + (summ_0 * (1 - pT) / D[:, L==0].shape[1]) + (summ_1 * pT / D[:, L==1].shape[1])
        return summ
        
    return R

def binary_linear_log_reg(DTR, LTR, DTE, LTE, hyp, pT= False, retModel= False, retScores= False, retAssigned= False):
    M = DTR.shape[0]
    
    minimum, _, _ = opt.fmin_l_bfgs_b(getCrossEntropy_binary_linear(DTR, LTR, hyp, pT= pT), np.zeros(M+1), approx_grad= True, factr= 1.)
    w, b = (minimum[:-1], minimum[-1])
    if retModel: return w, b

    NTE = DTE.shape[1]
    scores = w.T @ DTE + b
    if retScores: return scores

    assigned = np.array([1 if scores[i]>=0 else 0 for i in range(NTE)], dtype= int)
    if retAssigned: return assigned

    correct = (assigned==LTE).sum()
    return correct

def binary_quadratic_log_reg(DTR, LTR, DTE, LTE, hyp, pT= False, retModel= False, retScores= False, retAssigned= False):
    M, NTR = DTR.shape

    XX = []
    for i in range(NTR):
        x = DTR[:, i].reshape(M, 1)
        XX.append(np.vstack([com.vec(x @ x.T), x]))
    XX = np.hstack(XX)

    minimum, _, _ = opt.fmin_l_bfgs_b(getCrossEntropy_binary_linear(XX, LTR, hyp, pT= pT), np.zeros(M*(M+1) + 1), approx_grad= True, factr= 1.)
    w, b = minimum[:-1], minimum[-1]
    if retModel: return w

    NTE = DTE.shape[1]
    YY = []
    for i in range(NTE):
        y = DTE[:, i].reshape(M, 1)
        YY.append(np.vstack([com.vec(y @ y.T), y]))
    YY = np.hstack(YY)

    scores = w.T @ YY + b
    if retScores: return scores
    assigned = np.array([1 if scores[i]>=0 else 0 for i in range(NTE)], dtype= int)
    if retAssigned: return assigned

    correct = (assigned==LTE).sum()
    print(' -- LOGISTIC REGRESSION --')
    print('correct: ' + str(correct) + ' / ' + str(NTE))
    print(str((NTE-correct)*100/NTE))
    print('------------------------\n')
    return correct

def use_linear_log_reg(DTE, w, b, retScores= False):
    NTE = DTE.shape[1]
    scores = w.T @ DTE + b
    if retScores: return scores
    assigned = np.array([1 if scores[i]>=0 else 0 for i in range(NTE)])
    return assigned





#class prova:
#    def __init__(self):
#        self.cc=0
#    def get(self):
#        self.cc += 1
#        return self.cc-1
#
#def getCrossEntropy_linear(D, L, K, hyp, pp):
#    M, N = D.shape
#    T = np.array([[1 if c==L[i] else 0 for i in range(N)] for c in range(K)])
#
#    def R(params):
#        cc = pp.get()
#        if cc%1000 == 0: print('iter ' + str(cc))
#        w = params[:-K].reshape(M, K)
#        b = params[-K:].reshape(K, 1)
#
#        S = w.T @ D + b
#        lse = scipy.special.logsumexp(S, axis=0)
#        Y = S - lse
#        J = Y * T
#
#        reg = hyp * (w*w).sum() / 2
#
#        return reg - J.sum()/N
#        
#    return R
#
#def linear_log_reg(DTR, LTR, DTE, LTE, K, hyp, pp, results= False, messages= True, retModel= True):
#    M, NTE = DTE.shape
#
#    minimum, val, _ = opt.fmin_l_bfgs_b(getCrossEntropy_linear(DTR, LTR, K, hyp, pp), np.zeros((M+1)*K), approx_grad= True)
#    w, b = (minimum[:-K].reshape(M, K), minimum[-K:].reshape(K, 1))
#
#    scores = w.T @ DTE + b
#
#    assigned = np.argmax(scores, axis= 0)
#    correct = (assigned==LTE).sum()
#
#    if messages:
#        print(' -- LOGISTIC REGRESSION --')
#        print('correct: ' + str(correct) + ' / ' + str(NTE))
#        print(str((NTE-correct)*100/NTE))
#        print('------------------------\n')
#    return correct