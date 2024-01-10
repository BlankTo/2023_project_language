import numpy as np
import numpy.linalg as np_lng

from scipy.special import logsumexp

from ML_lib.utils import to_col, to_row, get_covMatrix


def log_pdf_MVG(X, mu, C):
    Xmu = X - mu
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

def log_likelihood(D, mu, C, pdf): return pdf(D, mu, C)

def multiclass_log_likelihood(D, mu, C, pdf):
    K = mu.shape[0]
    ll = []
    for c in range(K):
        ll.append(log_likelihood(D, mu[c], C[c], pdf))
    return np.array(ll)

def log_joint(log_S, priors= False):
    K = log_S.shape[0]
    if not priors: return log_S + np.log(1/K)
    else: return log_S + np.log(np.array(priors).reshape(K, 1))

def log_marginal(log_S_joint): return logsumexp(log_S_joint, axis= 0).reshape(1, log_S_joint.shape[1])

def log_posterior(log_S_joint, log_S_marginal): return log_S_joint - log_S_marginal

def multiclass_likelihood(D, mu, C, log_pdf): return np.exp(multiclass_log_likelihood(D, mu, C, log_pdf))

def joint(S, priors= False):
    K = S.shape[0]
    if not priors: return S * 1/K
    else: return S * np.array(priors).reshape(K, 1)

def marginal(S_joint): return S_joint.sum(axis= 0).reshape(1, S_joint.shape[1])

def posterior(S_joint, S_marginal): return S_joint / S_marginal

class MVG_classifier:
    def __init__(self, DTR, LTR, args_in= {}):
        args = {'log': True, 'priors': False, 'version': 'base'}
        for key in args_in.keys():
            args[key] = args_in[key]
        self.log = args['log']
        self.priors = args['priors']
        self.mu, self.C = multiclass_MVG_estimate(DTR, LTR, version= args['version'])

    def getNormPosteriors(self, DTE):
        S = multiclass_likelihood(DTE, self.mu, self.C, log_pdf_MVG)
        S_joint = joint(S, priors= self.priors)
        S_marginal = marginal(S_joint)
        return posterior(S_joint, S_marginal)
    
    def getLogPosteriors(self, DTE):
        log_S = multiclass_log_likelihood(DTE, self.mu, self.C, log_pdf_MVG)
        log_S_joint = log_joint(log_S, priors= self.priors)
        log_S_marginal = log_marginal(log_S_joint)
        self.log_S = log_S
        self.log_S_joint = log_S_joint
        self.log_S_marginal = log_S_marginal
        return log_posterior(log_S_joint, log_S_marginal)
    
    def getPosteriors(self, DTE):
        if not self.log: return self.getNormPosteriors(DTE)
        else: return self.getLogPosteriors(DTE)
    
    def getPredictions(self, DTE): return np.argmax(self.getPosteriors(DTE), axis= 0)

def Gaussian_Classifier_new(DTR, LTR, DTE, args_in= {}):

    args = {'log': True, 'priors': False, 'version': 'base', 'retPosteriors': False}
    for key in args_in.keys():
        args[key] = args_in[key]

    mu, C = multiclass_MVG_estimate(DTR, LTR, version= args['version'])
    
    if not args['log']:
        S = multiclass_likelihood(DTE, mu, C, log_pdf_MVG)
        S_joint = joint(S, priors= args['priors'])
        S_marginal = marginal(S_joint)
        posteriors = posterior(S_joint, S_marginal)
        predictions = np.argmax(posteriors, axis= 0)

    else:
        log_S = multiclass_log_likelihood(DTE, mu, C, log_pdf_MVG)
        log_S_joint = log_joint(log_S, priors= args['priors'])
        log_S_marginal = log_marginal(log_S_joint)
        posteriors = log_posterior(log_S_joint, log_S_marginal)
        predictions = np.argmax(posteriors, axis= 0)

    if args['retPosteriors']: return posteriors, predictions
    return predictions

def Gaussian_Classifier_new_2(DTR, LTR, DTE, log= True, priors= False, version= 'base', retPosteriors= False, retScores= False):

    mu, C = multiclass_MVG_estimate(DTR, LTR, version= version)
    
    if not log:
        S = multiclass_likelihood(DTE, mu, C, log_pdf_MVG)
        S_joint = joint(S, priors= priors)
        S_marginal = marginal(S_joint)
        posteriors = posterior(S_joint, S_marginal)
        predictions = np.argmax(posteriors, axis= 0)

    else:
        log_S = multiclass_log_likelihood(DTE, mu, C, log_pdf_MVG)
        log_S_joint = log_joint(log_S, priors= priors)
        log_S_marginal = log_marginal(log_S_joint)
        posteriors = log_posterior(log_S_joint, log_S_marginal)
        predictions = np.argmax(posteriors, axis= 0)

    #######################
    #check again this section
    #######################

    print('check the gaussina log or not log posteriors please')

    if retPosteriors: return posteriors, predictions
    #posteriors[posteriors == 0.] = -1e-15
    if retScores: return posteriors[0] / posteriors[1]
    #with or without /? is loglikelihood with - instead of /?
    return predictions

    #########################
    #########################
    #########################

class GaussianClassifier:
    def __init__(self, DTR, LTR, version= 'base', log= True, priors= False):
        self.mu, self.C = multiclass_MVG_estimate(DTR, LTR, version= version)
        self.log = log
        self.priors = priors

    def getPosteriors(self, DTE):
        if not self.log:
            S = multiclass_likelihood(DTE, self.mu, self.C, log_pdf_MVG)
            S_joint = joint(S, priors= self.priors)
            S_marginal = marginal(S_joint)
            posteriors = posterior(S_joint, S_marginal)

        else:
            log_S = multiclass_log_likelihood(DTE, self.mu, self.C, log_pdf_MVG)
            log_S_joint = log_joint(log_S, priors= self.priors)
            log_S_marginal = log_marginal(log_S_joint)
            posteriors = log_posterior(log_S_joint, log_S_marginal)
        
        return posteriors

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




#######################################   gaussian    ###################################################

#def Gaussian_Classifier(DTR, LTR, DTE, LTE, K, p= False, variant= False, result= True, retModel= False, retScores = False, retAssigned= False):
#    M, NTR = DTR.shape
#
#    mu, C = MVG_estimate(DTR, LTR, K)
#
#    if 'TIED' in str(variant):
#        C_TIED = np.zeros((M, M))
#        for c in range(K):
#            C_TIED += C[c] * DTR[:, LTR==c].shape[1]
#        C_TIED /= NTR
#    
#    for c in range(K):
#        C_use = C[c]
#        if variant:
#            if 'TIED' in variant:
#                C_use = C_TIED
#            if 'DIAG' in variant:
#                C_use = np.diag(np.diag(C_use))
#        C[c] = C_use ###############################################
#
#    if retModel: return mu, C
#
#    ll = []
#    for c in range(K):
#        ll.append(log_pdf_MVG(DTE, mu[c], C[c]))#C_use))
#    if type(p) == bool: p = np.array([1/K for _ in range(K)])
#    else:
#        if type(p) == float: p = np.array([1-p, p])
#    p = p.reshape(K, 1)
#    ll += np.log(p)
#    logS = np.vstack(ll)
#    logSmarginal = to_row(logsumexp(logS, axis=0))
#    logPosteriors = logS - logSmarginal
#    
#    if retScores: return logPosteriors[1] - logPosteriors[0]
#    
#    posteriors = np.exp(logPosteriors)
#
#    assigned = np.argmax(posteriors, axis= 0)
#    if retAssigned: return assigned
#
#    correct = (assigned == LTE).sum()
#    if result: print('error rate: ' + str((DTE.shape[1] - correct)*100/DTE.shape[1]))
#
#    return correct
#
#
#def use_GAU(DTE, K, mu, C_use, retScores= False):
#    ll = []
#    for c in range(K):
#        ll.append(log_pdf_MVG(DTE, mu[c], C_use))
#    ll += np.log(1/10)
#    logS = np.vstack(ll)
#    logSmarginal = to_row(logsumexp(logS, axis=0))
#    logPosteriors = logS - logSmarginal
#
#    if retScores: return logPosteriors[1] - logPosteriors[0]
#
#    posteriors = np.exp(logPosteriors)
#    assigned = np.argmax(posteriors, axis= 0)
#
#    return assigned
#
########################################   GMM    ###################################################
#
#def log_pdf_GMM(X, GMM, n_GMM, p= False):
#    S = np.vstack([log_pdf_MVG(X, GMM[g][1], (GMM[g][2])) + np.log(GMM[g][0]) for g in range(n_GMM)])
#    return S, logsumexp(S, axis=0)
#
#
#def EM_algo(X, init_GMM, n_GMM, maxiter= 1000, bound= False, variant= '', messages = False):
#    if messages: print('EM start')
#    GMM = init_GMM
#
#    try:
#        M, N = X.shape
#    except:
#        X = to_row(X)
#    M, N = X.shape
#
#    end_cycle = False
#    niter = -1
#    while (not end_cycle) and niter < maxiter :
#        niter += 1
#        if niter > 500: messages = True
#        S, log_marg_densities = log_pdf_GMM(X, GMM, n_GMM)
#
#        old_like = log_marg_densities.mean()
#
#        resp = np.exp(S - log_marg_densities)
#
#        first_stat = resp.sum(axis= 1).reshape(n_GMM, 1)
#
#        second_stat = resp @ X.T
#        
#        new_w = first_stat / N
#
#        new_mu = second_stat / first_stat
#
#        new_C = []
#        new_GMM = []
#        C_n = np.zeros((M, M))
#        for g in range(n_GMM):
#            gamma = resp[g, :]
#            third_stat = (X @ (to_row(gamma)* X).T)
#
#            new_C = third_stat / first_stat[g] - (to_col(new_mu[g]) @ to_row(new_mu[g]))
#
#            if 'DIAG' in str(variant):
#                new_C = np.diag(np.diag(new_C))
#            if 'TIED' in str(variant):
#                C_n += new_C * np.eye(M)
#                new_C = C_n
#            if bound:
#                U, s, _ = np_lng.svd(new_C)
#                s[s<bound] = bound
#                new_C = np.dot(U, to_col(s)*U.T)
#
#            if not ('TIED' in str(variant)): new_GMM.append((new_w[g], new_mu[g], new_C))
#        
#        if 'TIED' in str(variant):
#            for g in range(n_GMM): new_GMM.append((new_w[g], new_mu[g], C_n))
#            
#
#        _, new_marg = log_pdf_GMM(X, new_GMM, n_GMM)
#        new_like = new_marg.mean()
#
#        if messages: print('iter(' + str(niter) + ') - new: ' + str(new_like) + ' - old: ' + str(old_like) + ' = ' + str(new_like - old_like))
#
#        if new_like - old_like < 10**(-6) and niter > 1: end_cycle = True
#        else: GMM = new_GMM
#
#    return GMM
#
#
#def LBG_algo(X, alpha, maxG, bound= False, variant= False, retAll= False, messages_LBG= False, messages_EM= False, messages_G= False):
#    GMMs = []
#    try:
#        M, N = X.shape
#    except:
#        X = to_row(X)
#    M, N = X.shape
#    old_n_GMM = 1
#    if not (variant == 'TIED'): old_GMM = [(1., X.mean(axis= 1), get_covMatrix(X))]
#    else: old_GMM = [(1., X.mean(axis= 1), get_covMatrix(X) * np.eye(M))]
#    GMMs.append((old_GMM, old_n_GMM))
#    end_cycle = False
#
#    while not end_cycle:
#        if old_n_GMM < maxG:
#            GMM = []
#            for g in range(old_n_GMM):
#                if messages_G:
#                    print('------------------------')
#                    print(g)
#                    print(old_GMM[g])
#                    print('--')
#                w = old_GMM[g][0]
#                mu = old_GMM[g][1].reshape(M, 1)
#                C = old_GMM[g][2]
#
#                U, s, _ = np_lng.svd(C)
#                d = U[:, 0:1] * s[0]**0.5 * alpha
#
#                GMM.append((w/2, mu + d, C))
#                GMM.append((w/2, mu - d, C))
#                if messages_G:
#                    for i in [g*2, g*2+1]: print(GMM[i])
#            n_GMM = 2 * old_n_GMM
#        else:
#            GMM = old_GMM
#            n_GMM = old_n_GMM
#
#        GMM = EM_algo(X, GMM, n_GMM, bound= bound, variant= variant, messages= messages_EM)
#
#        _, old_marg = log_pdf_GMM(X, old_GMM, old_n_GMM)
#        old_like = old_marg.mean()
#        _, new_marg = log_pdf_GMM(X, GMM, n_GMM)
#        new_like = new_marg.mean()
#
#        if messages_LBG: print('new(' + str(n_GMM) + '): ' + str(new_like) + ' - old(' + str(old_n_GMM) + '): ' + str(old_like) + ' = ' + str(new_like - old_like))
#
#        if n_GMM >= maxG: end_cycle = True
#        else:
#            old_n_GMM = n_GMM
#            old_GMM = GMM
#            GMMs.append((old_GMM, old_n_GMM))
#    GMMs.append((GMM, n_GMM))
#    
#    if retAll: return GMMs
#    return GMM, n_GMM
#
#
#def use_GMM(GMMs, DTE, retScores= False):
#    scores = []
#    posteriors = []
#    for GMM, n_GMM in GMMs:
#        _, logSmarg = log_pdf_GMM(DTE, GMM, n_GMM)
#        print(logSmarg.shape)
#        scores.append(logSmarg[1] - logSmarg[0])
#        posteriors.append(np.exp(logSmarg))
#
#    scores = np.vstack(scores)
#    print(scores.shape)
#
#    if retScores: return scores
#
#    posteriors = np.vstack(posteriors)
#    return np.argmax(posteriors, axis= 0)