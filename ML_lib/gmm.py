import numpy as np
from scipy.special import logsumexp
from math import ceil
from tqdm import tqdm

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
        if len(nG) == 1: nG = [nG[0] for _ in range(self.K)]
        GMMs = []
        for k in setL:
            GMMs.append(LBG_algo(DTR[:, LTR == k], alpha, nG[k], bound, variant, retAll, messages_LBG, messages_EM, messages_G))
        self.GMMs = GMMs

    def getGMMs(self): return self.GMMs

    def getScores(self, DTE):

        if self.K > 2:
            print(F'ERROR - SCORES ARE VALID ONLY FOR 2 CLASSES - {self.k} CLASSES PASSED')
            exit(0)

        scores = []
        for GMM, n_GMM in self.GMMs:
            _, logSmarg = log_pdf_GMM(DTE, GMM, n_GMM)
            scores.append(logSmarg)

        scores = np.vstack(scores)
        scores = scores[1] - scores[0]
        return scores
    
    def get_lls(self, DTE):
        
        lls = []

        for k in range(self.K):
            gmm_class_k = self.GMMs[k][0]
            n_gmm_k = self.GMMs[k][1]

            _, ll_class_k = log_pdf_GMM(DTE, gmm_class_k, n_gmm_k)
            lls.append(ll_class_k)

        return np.vstack(lls)
    
    def predict(self, DTE): return self.get_lls(DTE).argmax(axis= 0)
    
    def get_lls_all(self, DTE):
        
        llss = []

        for k in range(self.K):

            lls = []

            for gmm_class_k, n_gmm_k in self.GMMs[k]:

                _, ll_class_k = log_pdf_GMM(DTE, gmm_class_k, n_gmm_k)
                lls.append((ll_class_k, n_gmm_k))

            llss.append(lls)

        return llss
    
    def predict_all(self, DTE):

        predictions = []

        llss = self.get_lls_all(DTE)

        for i in range(len(llss[0])):

            lls = np.vstack([llss[k][i][0] for k in range(len(llss))])

            predictions.append((lls.argmax(axis= 0), llss[0][i][0]))
        
        return predictions

    def getScores_all(self, DTE):

        if self.K > 2:
            print(F'ERROR - SCORES ARE VALID ONLY FOR 2 CLASSES - {self.k} CLASSES PASSED')
            exit(0)

        scores = []

        llss = self.get_lls_all(DTE)

        for i in range(len(llss[0])):

            lls = np.vstack([llss[k][i][0] for k in range(len(llss))])

            scores.append((lls[1] - lls[0], llss[0][i][1]))
        
        return scores

def cross_validation_gmm(D, L, n_folds= 10, prepro_class= None, dim_red= None, seed= 0, alpha= 0.1, nG_max= 2, bound= False, variant= '', messages= False):

    N = D.shape[1]
    np.random.seed(seed)
    permutation_index = np.random.permutation(N)
    
    D_folds = []
    L_folds = []

    if n_folds == -1: n_folds = N

    ## separation in folds (after shuffling)

    npf = ceil(N / n_folds)  # n per fold # cambiare a floor() se si preferisce l'ultima fold più grande delle altre piuttosto che più piccola
    for nf in range(n_folds-1):
        nf_idx = permutation_index[nf*npf:(nf+1)*npf]
        D_folds.append(D[:, nf_idx])
        L_folds.append(L[nf_idx])
    nf_idx = permutation_index[(nf+1)*npf:]
    D_folds.append(D[:, nf_idx])
    L_folds.append(L[nf_idx])

    ## 

    LTE = []
    scores = []

    for i_test in tqdm(range(n_folds)):
        DTR = D_folds.copy()
        LTR = L_folds.copy()
        DTE = DTR.pop(i_test)
        LTE.append(LTR.pop(i_test))
        DTR = np.hstack(DTR)
        LTR = np.hstack(LTR)
        DTE = np.vstack(DTE)

        if prepro_class is not None:
            prepro = prepro_class(DTR)
            DTR = prepro.transform(DTR)
            DTE = prepro.transform(DTE)

        if dim_red is not None:
            dim_red_class, m_dim_red = dim_red
            dr = dim_red_class(DTR, LTR)
            DTR = dr.transform(DTR, m_dim_red)
            DTE = dr.transform(DTE, m_dim_red)

        
        classifier = GMMClassifier(DTR, LTR, alpha, [nG_max], bound, variant, retAll= True, messages_LBG= messages)

        scores.append(classifier.getScores_all(DTE))

    scores_out = []

    for i in range(int(np.log2(nG_max)) + 1):

        scores_out_nG = []

        for j in range(n_folds):

            scores_out_nG.append(scores[j][i][0])
        
        scores_out.append((np.hstack(scores_out_nG), scores[j][i][1]))

    return np.hstack(LTE), scores_out