import numpy as np
import numpy.linalg as np_lng

from scipy.special import logsumexp

## check the formula

def log_pdf_MVG(X, mu, C):
    Xmu = X - mu
    print(f'just a check:\n X.shape[0] is {X.shape[0]} and C.shape[0] is {C.shape[0]}')
    return - ( np.log(2*np.pi)*X.shape[0] + np_lng.slogdet(C)[1] + np.diag(Xmu.T @ np_lng.inv(C) @ Xmu) ) / 2

def MVG_estimate(D, version= 'base'):
    M, N = D.shape
    mu = D.mean(axis= 1).reshape(M, 1)
    Dmu = D - mu
    if 'naive' not in version: C = (Dmu @ Dmu.T)
    #else: C = np.diag(np.array([Dmu[m, :]@Dmu.T[:, m] for m in range(M)]))
    else: C = np.diag(np.diag((Dmu @ Dmu.T)))
    if 'tied' not in version: C = C / N
    return np.array(mu), np.array(C)

def multiclass_MVG_estimate(D, L, version= 'base'):
    K = len(set(L))
    mu = []
    C = []
    for c in range(K):
        muC = MVG_estimate(D[:, L==c], version)
        mu.append(muC[0])
        C.append(muC[1])
    if 'tied' in version: C = [np.array(C).sum(axis= 0) / D.shape[1] for _ in range(K)]
    return np.array(mu), np.array(C)

def multiclass_log_likelihood(D, mu, C, log_pdf):
    K = mu.shape[0]
    ll = []
    for c in range(K):
        ll.append(log_pdf(D, mu[c], C[c]))
    return np.array(ll)

def log_joint(log_S, priors= False):
    K = log_S.shape[0]
    if not priors: return log_S + np.log(1/K)
    else: return log_S + np.log(np.array(priors).reshape(K, 1))

def log_marginal(log_S_joint): return logsumexp(log_S_joint, axis= 0).reshape(1, log_S_joint.shape[1])

def log_posterior(log_S_joint, log_S_marginal): return log_S_joint - log_S_marginal


class GaussianClassifier:
    def __init__(self, DTR, LTR, version= 'base', log= True, priors= False):
        self.mu, self.C = multiclass_MVG_estimate(DTR, LTR, version= version)
        self.log = log
        self.priors = priors

    def getPosteriors(self, DTE):
        
        log_S = multiclass_log_likelihood(DTE, self.mu, self.C, log_pdf_MVG)
        log_S_joint = log_joint(log_S, priors= self.priors)
        log_S_marginal = log_marginal(log_S_joint)
        log_posteriors = log_posterior(log_S_joint, log_S_marginal)

        if self.log:
            return log_posteriors
        else:
            return np.exp(log_posteriors)

    def predict(self, DTE): return np.argmax(self.getPosteriors(DTE), axis= 0)

    ########################
    ########################
    ##### e qui ########
    ########################
    ########################
    
    def getScores(self, DTE):
        posteriors = self.getPosteriors(DTE)
        #posteriors[posteriors == 0.] = -1e-15
        return posteriors[0] - posteriors[1]
    
    #####################
    #####################
    #####################