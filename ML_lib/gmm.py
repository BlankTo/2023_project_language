import numpy as np
from scipy.special import logsumexp

import ML_lib.utils as utils
from ML_lib.gaussian import log_pdf_MVG


def log_pdf_GMM(X, GMM, n_GMM, p= False):
    S = np.vstack([log_pdf_MVG(X, GMM[g][1].reshape(X.shape[0], 1), (GMM[g][2])) + np.log(GMM[g][0]) for g in range(n_GMM)])
    # joint_log_density, log_marginal
    return S, logsumexp(S, axis=0)


def EM_algo(X, init_GMM, n_GMM, maxiter= 1000, bound= False, variant= '', messages = False):
    if messages: print('EM start')
    GMM = init_GMM
    M, N = X.shape

    if bound:
        bound_GMM = []
        for gmm_gmm in init_GMM:
            U, s, _ = np.linalg.svd(gmm_gmm[2])
            s[s<bound] = bound
            bound_GMM_C = np.dot(U, utils.to_col(s)*U.T)
            bound_GMM.append([(gmm_gmm[0], gmm_gmm[1], bound_GMM_C)])

    end_cycle = False
    niter = -1
    while (not end_cycle) and niter < maxiter :
        niter += 1
        #if niter > 500: messages = True
        
        S, log_marg_densities = log_pdf_GMM(X, GMM, n_GMM)
        resp = np.exp(S - log_marg_densities)  #posteriors

        Zg = resp.sum(axis= 1).reshape(n_GMM, 1)
        Fg = resp @ X.T
        Sg = np.array([np.array([resp[g, i] * X[:, i].reshape(M, 1) @ X[:, i].reshape(1, M) for i in range(N)]).sum(axis = 0) for g in range(n_GMM)])
        
        new_w = Zg / N
        new_mu = Fg / Zg
        new_C = (Sg / Zg.reshape(n_GMM, 1, 1))  - np.einsum('ij,ik->ijk', new_mu, new_mu)

        if 'diag' in variant: new_C = np.array([np.diag(np.diag(new_C[g])) for g in range(n_GMM)])
        if 'tied' in variant:
            new_C = (new_C + Zg.reshape(n_GMM, 1, 1)).mean(axis= 0)
            new_C = np.array([new_C for _ in range(n_GMM)])

        if bound:
            bound_C = []
            for b_C in new_C:
                U, s, _ = np.linalg.svd(b_C)
                s[s<bound] = bound
                bound_C.append(np.dot(U, utils.to_col(s)*U.T))
            new_C = np.array(bound_C)

        new_GMM = [(new_w[g], new_mu[g], new_C[g]) for g in range(n_GMM)]

        old_like = log_marg_densities.mean()
        _, new_marg = log_pdf_GMM(X, new_GMM, n_GMM)
        new_like = new_marg.mean()

        if messages: print('iter(' + str(niter) + ') - new: ' + str(new_like) + ' - old: ' + str(old_like) + ' = ' + str(new_like - old_like))
        
        if new_like - old_like < 10**(-6) and niter > 1: end_cycle = True
        else: GMM = new_GMM

    return GMM


def LBG_algo(X, alpha, maxG, bound= False, variant= '', retAll= False, messages_LBG= False, messages_EM= False, messages_G= False):
    GMMs = []
    M, N = X.shape
    old_n_GMM = 1
    if 'diag' in variant: old_GMM = [(1., X.mean(axis= 1), np.diag(np.diag(utils.get_covMatrix(X))))]
    else: old_GMM = [(1., X.mean(axis= 1), utils.get_covMatrix(X))]
    if bound:
        U, s, _ = np.linalg.svd(old_GMM[0][2])
        s[s<bound] = bound
        old_C = np.dot(U, utils.to_col(s)*U.T)
        old_GMM = [(old_GMM[0][0], old_GMM[0][1], old_C)]
        old_GMM = EM_algo(X, old_GMM, old_n_GMM, bound= bound, variant= variant, messages= messages_EM)
    GMMs.append((old_GMM, old_n_GMM))
    end_cycle = False

    while not end_cycle:
        if old_n_GMM < maxG:
            GMM = []
            for g in range(old_n_GMM):
                if messages_G:
                    print('------------------------')
                    print(g)
                    print(old_GMM[g])
                    print('--')
                w = old_GMM[g][0]
                mu = old_GMM[g][1].reshape(M, 1)
                C = old_GMM[g][2]

                U, s, _ = np.linalg.svd(C)
                d = U[:, 0:1] * s[0]**0.5 * alpha

                GMM.append((w/2, mu + d, C))
                GMM.append((w/2, mu - d, C))
                if messages_G:
                    for i in [g*2, g*2+1]: print(GMM[i])
            n_GMM = 2 * old_n_GMM
        else:
            GMM = old_GMM
            n_GMM = old_n_GMM

        GMM = EM_algo(X, GMM, n_GMM, bound= bound, variant= variant, messages= messages_EM)

        _, old_marg = log_pdf_GMM(X, old_GMM, old_n_GMM)
        old_like = old_marg.mean()
        _, new_marg = log_pdf_GMM(X, GMM, n_GMM)
        new_like = new_marg.mean()

        if messages_LBG: print('new(' + str(n_GMM) + '): ' + str(new_like) + ' - old(' + str(old_n_GMM) + '): ' + str(old_like) + ' = ' + str(new_like - old_like))

        if n_GMM >= maxG: end_cycle = True
        else:
            old_n_GMM = n_GMM
            old_GMM = GMM
            GMMs.append((old_GMM, old_n_GMM))
    GMMs.append((GMM, n_GMM))
    
    if retAll: return GMMs
    return GMM, n_GMM


def use_GMM(GMMs, DTE, retScores= False):
    scores = []
    posteriors = []
    for GMM, n_GMM in GMMs:
        _, logSmarg = log_pdf_GMM(DTE, GMM, n_GMM)
        print(logSmarg.shape)
        scores.append(logSmarg[1] - logSmarg[0])
        posteriors.append(np.exp(logSmarg))

    scores = np.vstack(scores)
    print(scores.shape)

    if retScores: return scores

    posteriors = np.vstack(posteriors)
    return np.argmax(posteriors, axis= 0)


class GMMClassifier():
    def __init__(self, DTR, LTR, alpha, nG= [2, 2], bound= False, variant= '', retAll= False, messages_LBG= False, messages_EM= False, messages_G= False):
        setL = set(LTR)
        self.K = len(setL)
        GMMs = []
        for k in setL:
            GMMs.append(LBG_algo(DTR[:, LTR == k], alpha, nG[k], bound, variant, retAll, messages_LBG, messages_EM, messages_G))
        self.GMMs = GMMs

    def getGMMs(self): return self.GMMs

    def getScores(self, DTE):
        scores = []
        for GMM, n_GMM in self.GMMs:
            _, logSmarg = log_pdf_GMM(DTE, GMM, n_GMM)
            scores.append(logSmarg)

        scores = np.vstack(scores)
        scores = scores[1] - scores[0]
        return scores
    
    def predict(self, DTE): return (self.getScores(DTE) > 0).astype(int)

    def getScores_All(self, DTE):
        scores_all = []
        NnG = len(self.GMMs[0])
        for k in range(self.K):
            scores = []
            for nnG in range(NnG):
                _, logSmarg = log_pdf_GMM(DTE, self.GMMs[k][nnG][0], self.GMMs[k][nnG][1])
                scores.append(logSmarg)
            scores_all.append(scores)

        scores = []
        for nnG_0 in range(NnG):
            for nnG_1 in range(NnG):
                scores.append(scores_all[1][nnG_1] - scores_all[0][nnG_0])
        scores = np.array(scores)
        return scores
    
    def predict_All(self, DTE): return (self.getScores_All(DTE) > 0).astype(int)
