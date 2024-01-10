import numpy as np
from scipy.optimize import fmin_l_bfgs_b

from ML_lib.utils import vec


def getCrossEntropy(D, L, hyp, pT= False, retPrime= False):
    Z = L * 2 - 1
    N = D.shape[1]

    def R(params):
        w = params[:-1]
        b = params[-1]
        summ_0 = (np.logaddexp(0, -Z[L==0]*(w.T @ D[:, L==0] + b))).sum()
        summ_1 = (np.logaddexp(0, -Z[L==1]*(w.T @ D[:, L==1] + b))).sum()
        if type(pT) == bool: summ = (hyp * np.linalg.norm(w)**2 / 2) + (summ_0 + summ_1) / N
        else : summ = (hyp * np.linalg.norm(w)**2 / 2) + (summ_0 * (1 - pT) / D[:, L==0].shape[1]) + (summ_1 * pT / D[:, L==1].shape[1])
        if retPrime: return summ, [...]
        return summ
        
    return R


class lin_log_reg:
    def __init__(self, DTR, LTR, args_in= {}):
        args = {'lam': -1, 'priors': False, 'approx_grad': True, 'factr': 1.0}
        for key in args_in.keys():
            args[key] = args_in[key]
        if args['lam'] < 0: print('wrong lam in args_in -- class lin_log_reg')
        if args['approx_grad']: crossEntropy = getCrossEntropy(DTR, LTR, args['lam'], pT= args['priors'])
        else: getCrossEntropy(DTR, LTR, args['lam'], pT= args['priors'], retPrime= True)

        minimum, _, _ = fmin_l_bfgs_b(crossEntropy, np.zeros(DTR.shape[0] + 1), approx_grad= args['approx_grad'], factr= args['factr'])
        self.w, self.b = (minimum[:-1], minimum[-1])

    def getScores(self, DTE): return self.w.T @ DTE + self.b

    def getPredictions(self, DTE):
        scores = self.getScores(DTE)
        return np.array([1 if scores[i]>=0 else 0 for i in range(DTE.shape[1])], dtype= int)
    
class LinearRegressionClassifier:
    def __init__(self, DTR, LTR, lam= -1, priors= False, approx_grad= True, factr= 1.0):
        if lam < 0: print('wrong lam in args_in -- class LinearLogisticRegression')
        if approx_grad: crossEntropy = getCrossEntropy(DTR, LTR, lam, pT= priors)
        else: getCrossEntropy(DTR, LTR, lam, pT= priors, retPrime= True)

        minimum, _, _ = fmin_l_bfgs_b(crossEntropy, np.zeros(DTR.shape[0] + 1), approx_grad= approx_grad, factr= factr)
        self.w, self.b = (minimum[:-1], minimum[-1])

    def getScores(self, DTE): return self.w.T @ DTE + self.b

    def predict(self, DTE):
        scores = self.getScores(DTE)
        return np.array([1 if scores[i]>=0 else 0 for i in range(DTE.shape[1])], dtype= int)
    
class QuadraticRegressionClassifier:
    def __init__(self, DTR, LTR, hyp= -1, priors= False, approx_grad= True, factr= 1.0):
        if hyp < 0: print('wrong hyp in args_in -- class QuadraticLogisticRegression')

        self.M, NTR = DTR.shape

        XX = []
        for i in range(NTR):
            x = DTR[:, i].reshape(self.M, 1)
            XX.append(np.vstack([vec(x @ x.T), x]))
        XX = np.hstack(XX)

        if approx_grad: minimum, _, _ = fmin_l_bfgs_b(getCrossEntropy(XX, LTR, hyp, pT= priors), np.zeros(self.M*(self.M+1) + 1), approx_grad= True, factr= 1.)
        else: minimum, _, _ = fmin_l_bfgs_b(getCrossEntropy(XX, LTR, hyp, pT= priors), np.zeros(self.M*(self.M+1) + 1), approx_grad= False, factr= 1.)
        self.w, self.b = minimum[:-1], minimum[-1]

    def getScores(self, DTE):
        NTE = DTE.shape[1]
        YY = []
        for i in range(NTE):
            y = DTE[:, i].reshape(self.M, 1)
            YY.append(np.vstack([vec(y @ y.T), y]))
        YY = np.hstack(YY)

        return self.w.T @ YY + self.b

    def predict(self, DTE):
        return np.array([1 if self.getScores(DTE)[i]>=0 else 0 for i in range(DTE.shape[1])], dtype= int)


#################################


#def binary_linear_log_reg(DTR, LTR, DTE, LTE, hyp, pT= False, retModel= False, retScores= False, retAssigned= False):
#    M = DTR.shape[0]
#    
#    minimum, _, _ = fmin_l_bfgs_b(getCrossEntropy(DTR, LTR, hyp, pT= pT), np.zeros(M+1), approx_grad= True, factr= 1.)
#    w, b = (minimum[:-1], minimum[-1])
#    if retModel: return w, b
#
#    NTE = DTE.shape[1]
#    scores = w.T @ DTE + b
#    if retScores: return scores
#
#    assigned = np.array([1 if scores[i]>=0 else 0 for i in range(NTE)], dtype= int)
#    if retAssigned: return assigned
#
#    correct = (assigned==LTE).sum()
#    return correct
#
#def binary_quadratic_log_reg(DTR, LTR, DTE, LTE, hyp, pT= False, retModel= False, retScores= False, retAssigned= False):
#    M, NTR = DTR.shape
#
#    XX = []
#    for i in range(NTR):
#        x = DTR[:, i].reshape(M, 1)
#        XX.append(np.vstack([vec(x @ x.T), x]))
#    XX = np.hstack(XX)
#
#    minimum, _, _ = fmin_l_bfgs_b(getCrossEntropy(XX, LTR, hyp, pT= pT), np.zeros(M*(M+1) + 1), approx_grad= True, factr= 1.)
#    w, b = minimum[:-1], minimum[-1]
#    if retModel: return w
#
#    NTE = DTE.shape[1]
#    YY = []
#    for i in range(NTE):
#        y = DTE[:, i].reshape(M, 1)
#        YY.append(np.vstack([vec(y @ y.T), y]))
#    YY = np.hstack(YY)
#
#    scores = w.T @ YY + b
#    if retScores: return scores
#    assigned = np.array([1 if scores[i]>=0 else 0 for i in range(NTE)], dtype= int)
#    if retAssigned: return assigned
#
#    correct = (assigned==LTE).sum()
#    print(' -- LOGISTIC REGRESSION --')
#    print('correct: ' + str(correct) + ' / ' + str(NTE))
#    print(str((NTE-correct)*100/NTE))
#    print('------------------------\n')
#    return correct
#
#def use_linear_log_reg(DTE, w, b, retScores= False):
#    NTE = DTE.shape[1]
#    scores = w.T @ DTE + b
#    if retScores: return scores
#    assigned = np.array([1 if scores[i]>=0 else 0 for i in range(NTE)])
#    return assigned
#