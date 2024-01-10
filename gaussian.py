import numpy as np
import numpy.linalg as np_lng
import scipy.special
import scipy.special as spe

import My_Lib_ML.common as com


def log_pdf_MVG(X, mu, C):
    try: M, N = X.shape
    except:
        X = com.to_row(X)
        M, N = X.shape

    mu = np.array(mu, dtype= float).reshape(M, 1)
    C = np.array(C, dtype= float).reshape(M, M)
    Xmu = X - mu
    return -(np.diag((Xmu.T @ (np_lng.inv(C) @ Xmu))) + (M * np.log(2*np.pi) + np_lng.slogdet(C)[1])) / 2

def MVG_estimate(D, L, k):
    try: M, N = D.shape
    except:
        D = com.to_row(D)
        M, N = D.shape
    
    mu = []
    C = []
    for c in range(k):
        Dc = D[:, L==c]
        mu.append(Dc.mean(axis= 1).reshape(M, 1))
        Dmu = Dc - mu[c]
        C.append((Dmu @ Dmu.T) / Dc.shape[1])
    return np.array(mu), np.array(C)


#######################################   gaussian    ###################################################

def Gaussian_Classifier(DTR, LTR, DTE, LTE, K, p= False, variant= False, result= True, retModel= False, retScores = False, retAssigned= False):
    M, NTR = DTR.shape

    mu, C = MVG_estimate(DTR, LTR, K)

    if 'TIED' in str(variant):
        C_TIED = np.zeros((M, M))
        for c in range(K):
            C_TIED += C[c] * DTR[:, LTR==c].shape[1]
        C_TIED /= NTR
    
    for c in range(K):
        C_use = C[c]
        if variant:
            if 'TIED' in variant:
                C_use = C_TIED
            if 'DIAG' in variant:
                C_use = np.diag(np.diag(C_use))
        C[c] = C_use ###############################################

    if retModel: return mu, C

    ll = []
    for c in range(K):
        ll.append(log_pdf_MVG(DTE, mu[c], C[c]))#C_use))
    if type(p) == bool: p = np.array([1/K for _ in range(K)])
    else:
        if type(p) == float: p = np.array([1-p, p])
    p = p.reshape(K, 1)
    ll += np.log(p)
    logS = np.vstack(ll)
    logSmarginal = com.to_row(scipy.special.logsumexp(logS, axis=0))
    logPosteriors = logS - logSmarginal
    
    if retScores: return logPosteriors[1] - logPosteriors[0]
    
    posteriors = np.exp(logPosteriors)

    assigned = np.argmax(posteriors, axis= 0)
    if retAssigned: return assigned

    correct = (assigned == LTE).sum()
    if result: print('error rate: ' + str((DTE.shape[1] - correct)*100/DTE.shape[1]))

    return correct


def use_GAU(DTE, K, mu, C_use, retScores= False):
    ll = []
    for c in range(K):
        ll.append(log_pdf_MVG(DTE, mu[c], C_use))
    ll += np.log(1/10)
    logS = np.vstack(ll)
    logSmarginal = com.to_row(scipy.special.logsumexp(logS, axis=0))
    logPosteriors = logS - logSmarginal

    if retScores: return logPosteriors[1] - logPosteriors[0]

    posteriors = np.exp(logPosteriors)
    assigned = np.argmax(posteriors, axis= 0)

    return assigned

#######################################   GMM    ###################################################

def log_pdf_GMM(X, GMM, n_GMM, p= False):
    S = np.vstack([log_pdf_MVG(X, GMM[g][1], (GMM[g][2])) + np.log(GMM[g][0]) for g in range(n_GMM)])
    return S, spe.logsumexp(S, axis=0)


def EM_algo(X, init_GMM, n_GMM, maxiter= 1000, bound= False, variant= '', messages = False):
    if messages: print('EM start')
    GMM = init_GMM

    try:
        M, N = X.shape
    except:
        X = com.to_row(X)
    M, N = X.shape

    end_cycle = False
    niter = -1
    while (not end_cycle) and niter < maxiter :
        niter += 1
        if niter > 500: messages = True
        S, log_marg_densities = log_pdf_GMM(X, GMM, n_GMM)

        old_like = log_marg_densities.mean()

        resp = np.exp(S - log_marg_densities)

        first_stat = resp.sum(axis= 1).reshape(n_GMM, 1)

        second_stat = resp @ X.T
        
        new_w = first_stat / N

        new_mu = second_stat / first_stat

        new_C = []
        new_GMM = []
        C_n = np.zeros((M, M))
        for g in range(n_GMM):
            gamma = resp[g, :]
            third_stat = (X @ (com.to_row(gamma)* X).T)

            new_C = third_stat / first_stat[g] - (com.to_col(new_mu[g]) @ com.to_row(new_mu[g]))

            if 'DIAG' in str(variant):
                new_C = np.diag(np.diag(new_C))
            if 'TIED' in str(variant):
                C_n += new_C * np.eye(M)
                new_C = C_n
            if bound:
                U, s, _ = np_lng.svd(new_C)
                s[s<bound] = bound
                new_C = np.dot(U, com.to_col(s)*U.T)

            if not ('TIED' in str(variant)): new_GMM.append((new_w[g], new_mu[g], new_C))
        
        if 'TIED' in str(variant):
            for g in range(n_GMM): new_GMM.append((new_w[g], new_mu[g], C_n))
            

        _, new_marg = log_pdf_GMM(X, new_GMM, n_GMM)
        new_like = new_marg.mean()

        if messages: print('iter(' + str(niter) + ') - new: ' + str(new_like) + ' - old: ' + str(old_like) + ' = ' + str(new_like - old_like))

        if new_like - old_like < 10**(-6) and niter > 1: end_cycle = True
        else: GMM = new_GMM

    return GMM


def LBG_algo(X, alpha, maxG, bound= False, variant= False, retAll= False, messages_LBG= False, messages_EM= False, messages_G= False):
    GMMs = []
    try:
        M, N = X.shape
    except:
        X = com.to_row(X)
    M, N = X.shape
    old_n_GMM = 1
    if not (variant == 'TIED'): old_GMM = [(1., X.mean(axis= 1), com.get_covMatrix(X))]
    else: old_GMM = [(1., X.mean(axis= 1), com.get_covMatrix(X) * np.eye(M))]
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

                U, s, _ = np_lng.svd(C)
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