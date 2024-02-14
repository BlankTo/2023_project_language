import os
import numpy as np
import time
from tqdm import tqdm

from math import ceil


def csv_to_npy(path, name= ''):
    if not os.path.isfile(path):
        print('file "' + path + '" not found')
        exit(0)
    savepath = 'data_npy/'
    if not name: name = path.split('/')[-1].split('.')[0]
    if os.path.isfile(savepath + 'DTR_' + name + '.npy'):
        print('file "' + path + '" already converted\n\nloading:\n' + savepath + 'DTR_' + name + '.npy\n' + savepath + 'LTR_' + name + '.npy\n' + savepath + 'label_set_' + name + '.npy\n')
        label_set = np.load(savepath + 'label_set_' + name + '.npy')
        return np.load(savepath + 'DTR_' + name + '.npy'), np.load(savepath + 'LTR_' + name + '.npy'), {i: label_set[i] for i in range(len(label_set))}
    DTR = []
    LTR = []
    label_set = []
    with open(path, 'r') as f:
        print('file "' + path + '" loaded\n')
        for line in f:
            line_split = line.strip('\n').split(',')
            x = [float(xj) for xj in line_split[:-1]]
            DTR.append(x)
            if line_split[-1] in label_set:
                LTR.append(label_set.index(line_split[-1]))
            else:
                label_set.append(line_split[-1])
                LTR.append(label_set.index(line_split[-1]))
    DTR = (np.array(DTR, dtype= np.float64)).T
    LTR = np.array(LTR, dtype= int)
    label_dict = {i: label_set[i] for i in range(len(label_set))}
    label_set = np.array(label_set, dtype= str)
    print('created files:')
    np.save(savepath + 'DTR_' + name, DTR)
    print(savepath + 'DTR_' + name + '.npy')
    np.save(savepath + 'LTR_' + name, LTR)
    print(savepath + 'LTR_' + name + '.npy')
    np.save(savepath + 'label_set_' + name, label_set)
    print(savepath + 'label_set_' + name + '.npy\n')
    return DTR, LTR, label_dict


def to_col(v): return v.reshape(v.size, 1)


def to_row(v): return v.reshape(1, v.size)


def vec(X): return np.vstack([X[:, i].reshape(X.shape[0], 1) for i in range(X.shape[1])])


def get_var(X): return ((X - X.mean())**2).sum() / (X.size-1)


def get_cov(X, Y): return (((X - X.mean())*(Y - Y.mean())).sum()) / (X.size-1)


def get_covMatrix(D):
    M = D.shape[0]
    covMat = np.zeros((M, M))
    for i in range(M):
        covMat[i, i] += get_var(D[i, :])
        for j in range(i+1, M):
            covMat[i, j] += get_cov(D[i, :], D[j, :])
            covMat[j, i] += get_cov(D[i, :], D[j, :])
    return covMat


def get_corrMatrix(D):
    M = D.shape[0]
    corrMat = get_covMatrix(D)
    for i in range(M):
        for j in range(i+1, M):
            corrMat[i, j] = round(corrMat[i, j] / np.sqrt(corrMat[i, i]*corrMat[j, j]), 3)
            corrMat[j, i] = round(corrMat[j, i] / np.sqrt(corrMat[i, i]*corrMat[j, j]), 3)
    for i in range(M): corrMat[i, i] = 1.
    return corrMat


def shuffle_and_divide(D, L, fraction_in_train, seed= 0):
    N = D.shape[1]
    np.random.seed(seed)
    permutation_index = np.random.permutation(N)
    D_out = D[:, permutation_index]
    L_out = L[permutation_index]
    division_index = int(N * fraction_in_train)
    return D_out[:, :division_index], L_out[:division_index], D_out[:, division_index:], L_out[division_index:]


def getConfusionMatrix(predictions, LTE):
    K = len(set(LTE))
    conf = np.zeros((K, K))

    for c_pred in range(K):
        for c_actual in range(K):
            conf[c_pred, c_actual] = np.logical_and(predictions==c_pred, LTE==c_actual).sum()

    return conf


def getBayesDecision(scores, costs): return np.argmin(costs @ scores, axis= 0)

def getBayesThreshold(p1, Cfp, Cfn): return np.log((1-p1)*Cfp) - np.log(p1*Cfn)

def logBinaryBayesDecision(log_S, Cfp, Cfn, p1= 0.5): return ((log_S[1] - log_S[0]) > getBayesThreshold(p1, Cfp, Cfn)).astype(int)

def getRatios(confusion):
    FPR = confusion[1, 0] / (confusion[1, 0] + confusion[0, 0])
    FNR = confusion[0, 1] / (confusion[0, 1] + confusion[1, 1])
    TPR = 1 - FNR
    TNR = 1 - FPR
    return FPR, FNR, TPR, TNR

def empiricalBayesRisk(p1, Cfp, Cfn, FPR, FNR): return p1 * Cfn * FNR + (1 - p1) * Cfp * FPR

def normalizedDCF(p1, Cfp, Cfn, FPR, FNR): return empiricalBayesRisk(p1, Cfp, Cfn, FPR, FNR) / min(p1*Cfn, (1-p1)*Cfp)

def getMinNormDCF(scores_in, labels_in, p1, Cfp, Cfn, retRatios= False, retThreshold= False):
    N = scores_in.shape[0]
    idx = np.argsort(scores_in)
    scores = scores_in.copy()[idx]
    labels = labels_in.copy()[idx]
    minDCF = 1000
    best_threshold = None

    TPRs, TNRs, FPRs, FNRs, nDCFs = [], [], [], [], []

    for i in range(scores.shape[0]):

        predictions = np.concatenate([np.zeros(i), np.ones(N-i)])

        TNR = (predictions[labels==0]==0).sum() / (labels==0).sum()
        TPR = (predictions[labels==1]==1).sum() / (labels==1).sum()
        FNR = (predictions[labels==1]==0).sum() / (labels==1).sum()
        FPR = (predictions[labels==0]==1).sum() / (labels==0).sum()

        TPRs.append(TPR)
        TNRs.append(TNR)
        FNRs.append(FNR)
        FPRs.append(FPR)

        nDCF = normalizedDCF(p1, Cfp, Cfn, FPR, FNR)

        nDCFs.append(nDCF)

        if nDCF < minDCF:
            minDCF = nDCF
            best_threshold = scores[i]

    if retRatios: return TPRs, TNRs, FPRs, FNRs, nDCFs
    if retThreshold: return minDCF, best_threshold
    return minDCF


def inter_BayesErrorPlot(llr, labels, eff):

    predictions = (llr > getBayesThreshold(eff, 1, 1)).astype(int)

    FNR = (predictions[labels==1]==0).sum() / (labels==1).sum()
    FPR = (predictions[labels==0]==1).sum() / (labels==0).sum()

    return normalizedDCF(eff, 1, 1, FPR, FNR), getMinNormDCF(llr, labels, eff, 1, 1)


def cross_validation_base(D, L, classifier_class, n_folds= 10, prepro_class= None, dim_red= None, seed= 0, args_in= [], ret_scores= False, noClass= False):

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
    predictions = []
    if ret_scores: scores = []

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

        if noClass: predictions.append(classifier_class(DTR, LTR, DTE, *args_in))
        else:
            classifier = classifier_class(DTR, LTR, *args_in)
            predictions.append(classifier.predict(DTE))
        if ret_scores: scores.append(classifier.getScores(DTE))

    if ret_scores: return np.hstack(LTE), np.hstack(predictions), np.hstack(scores)
    return np.hstack(LTE), np.hstack(predictions)


def get_metrics(scores, labels, p1= 0.5, Cfp= 1, Cfn= 1, print_err= False, ret_all= False):
    
    assigned = (scores > getBayesThreshold(p1, Cfp, Cfn)).astype(int)

    if print_err: error_rate = sum(assigned != labels) * 100 / scores.shape[0]

    FNR = (assigned[labels==1]==0).sum() / (labels==1).sum()
    FPR = (assigned[labels==0]==1).sum() / (labels==0).sum()

    nDCF = normalizedDCF(p1, Cfp, Cfn, FPR, FNR)
    minDCF = getMinNormDCF(scores, labels, p1, Cfp, Cfn)
    
    if print_err: print_string = str('normDCF(%.1f, %d, %d): %.3f  -  minNormDCF: %.3f  -  err: %.1f' % (p1, Cfp, Cfn, nDCF, minDCF, error_rate))
    else: print_string = str('normDCF(%.1f, %d, %d): %.3f  -  minNormDCF: %.3f' % (p1, Cfp, Cfn, nDCF, minDCF))

    if ret_all: return print_string, nDCF, minDCF, FPR, FNR
    else: print(print_string)

def get_metrics_threshold(scores, labels, threshold, p1= 0.5, Cfp= 1, Cfn= 1, print_err= False, ret_all= False):
    
    assigned = (scores > threshold).astype(int)

    if print_err: error_rate = sum(assigned != labels) * 100 / scores.shape[0]

    FNR = (assigned[labels==1]==0).sum() / (labels==1).sum()
    FPR = (assigned[labels==0]==1).sum() / (labels==0).sum()

    nDCF = normalizedDCF(p1, Cfp, Cfn, FPR, FNR)
    minDCF = getMinNormDCF(scores, labels, p1, Cfp, Cfn)
    
    if print_err: print_string = str('normDCF(%.1f, %d, %d): %.3f  -  minNormDCF: %.3f  -  err: %.1f' % (p1, Cfp, Cfn, nDCF, minDCF, error_rate))
    else: print_string = str('normDCF(%.1f, %d, %d): %.3f  -  minNormDCF: %.3f' % (p1, Cfp, Cfn, nDCF, minDCF))

    if ret_all: return print_string, nDCF, minDCF, FPR, FNR
    else: print(print_string)

def cross_validation(DTR_in, LTR_in, n_split, model, model_params, prepro= [[]], effective= [0.5], progress= False, save= False, filename= 'results\\cross_vall_no_name.txt', print_err= False, model_params_print= None):

    NTR = DTR_in.shape[1]

    np.random.seed(0)
    idx = np.random.permutation(NTR)
    DTR_perm = DTR_in[:, idx]
    LTR_perm = LTR_in[idx]

    if n_split == -1: n_split = NTR

    step = NTR / n_split
    considered = []
    scores_all = []

    n_ite = len(prepro) * len(model_params)

    if progress: print('- starting -\n')
    tot_time = 0
    for i_split in range(n_split):
        if progress: start_time = time.time()
        if progress: print(f'split: {i_split+1} / {n_split}\n')
        DTR_base = []
        LTR = []
        DTE_base = []
        LTE = []
        for i in range(n_split):
            if not i_split==i:
                DTR_base.append(DTR_perm[:, int(step * i):int(step * (i+1))])
                LTR.append(LTR_perm[int(step * i):int(step * (i+1))])
            else:
                DTE_base = DTR_perm[:, int(step * i):int(step * (i+1))]
                LTE = LTR_perm[int(step * i):int(step * (i+1))]
        DTR_base = np.hstack(DTR_base)
        LTR = np.hstack(LTR)
        DTE_base = np.array(DTE_base)
        considered.append(LTE)

        scores_i = []
        i_ite = 0

        for prep_ite in prepro:

            DTR_pre = DTR_base
            DTE_pre = DTE_base

            for prep in prep_ite:

                prep_t = prep[0](DTR_pre, LTR)
                DTR_pre = prep_t.transform(DTR_pre, *prep[1])
                DTE_pre = prep_t.transform(DTE_pre, *prep[1])

            for model_param in model_params:
                if progress:
                    print(f'\x1b[1A]\x1b[2K -- ite: {i_ite+1} / {n_ite}')
                    i_ite += 1
                model_init = model(DTR_pre, LTR, *model_param)

                scores_ii = model_init.getScores(DTE_pre)
                scores_i.append(scores_ii)

        scores_all.append(scores_i)
        if progress:
            tot_time += time.time() - start_time
            print('\x1b[1A]\x1b[2Ksplit completed in %.2fs - eta for completion is %.2fs' % (time.time() - start_time, (tot_time * (n_split - i_split - 1) / (i_split + 1))))
    if progress: print('all splits completed in %.2fs' % tot_time)

    considered = np.hstack(considered)

    if save:
        try: f = open(filename, 'a')
        except: f = open(filename, 'w')

    place = 0
    for prep_ite in prepro:

        i_mp = 0
        for model_param in model_params:

            scores = np.hstack([scores_all[i][place] for i in range(n_split)])
            place += 1

            spec = '\n'
            for prep in prep_ite: spec += f'{prep[0].getName()} with {prep[1]}  '
            spec += ' -- params: '
            if model_params_print is None:
                spec += '['
                for i_spec in model_param: 
                    if not callable(i_spec): spec += f'{i_spec}, '
                    else: spec += 'kernel'
                spec += ']'
            else: 
                spec += str(model_params_print[i_mp])
                i_mp += 1

            print_string = str(spec.ljust(70))
            save_string = str(spec.ljust(70))
            for eff in effective:
                p_string, nDCF, minDCF, FPR, FNR = get_metrics(scores, considered, eff, 1, 1, print_err= print_err, ret_all= True)
                print_string += str(' - %.2f: %.3f, %.3f (%.1f, %.1f) |' % (eff, nDCF, minDCF, FPR, FNR))
                save_string += str(' - minDCF(%.2f): %.3f' % (eff, minDCF))
            
            print(print_string)
            if save: f.write(save_string)
        if save: f.write('\n')

    if save: f.close()

#def cross_validation_backup(DTR_in, LTR_in, n_split, model, model_params, prepro= [[]], progress= False, save= False, filename= 'results\\cross_vall_no_name.txt', model_params_print= None, print_act= False, print_err= False):
#
#    NTR = DTR_in.shape[1]
#
#    np.random.seed(0)
#    idx = np.random.permutation(NTR)
#    DTR_perm = DTR_in[:, idx]
#    LTR_perm = LTR_in[idx]
#
#    step = NTR / n_split
#    considered = []
#    assigned_all = []
#    scores_all = []
#
#    n_ite = len(prepro) * len(model_params)
#
#    if progress: print('- starting -\n')
#    tot_time = 0
#    for i_split in range(n_split):
#        if progress: start_time = time.time()
#        if progress: print(f'split: {i_split+1} / {n_split}\n')
#        DTR_base = []
#        LTR = []
#        DTE_base = []
#        LTE = []
#        for i in range(n_split):
#            if not i_split==i:
#                DTR_base.append(DTR_perm[:, int(step * i):int(step * (i+1))])
#                LTR.append(LTR_perm[int(step * i):int(step * (i+1))])
#            else:
#                DTE_base = DTR_perm[:, int(step * i):int(step * (i+1))]
#                LTE = LTR_perm[int(step * i):int(step * (i+1))]
#        DTR_base = np.hstack(DTR_base)
#        LTR = np.hstack(LTR)
#        DTE_base = np.array(DTE_base)
#        considered.append(LTE)
#
#        assigned_i = []
#        scores_i = []
#        i_ite = 0
#
#        for prep_ite in prepro:
#
#            DTR_pre = DTR_base
#            DTE_pre = DTE_base
#
#            for prep in prep_ite:
#
#                prep_t = prep[0](DTR_pre, LTR)
#                DTR_pre = prep_t.transform(DTR_pre, *prep[1])
#                DTE_pre = prep_t.transform(DTE_pre, *prep[1])
#
#            for model_param in model_params:
#                if progress:
#                    print(f'\x1b[1A]\x1b[2K -- ite: {i_ite+1} / {n_ite}')
#                    i_ite += 1
#                model_init = model(DTR_pre, LTR, *model_param)
#
#                scores_ii = model_init.getScores(DTE_pre)
#                scores_i.append(scores_ii)
#                assigned_ii = (scores_ii > 0).astype(int)
#                assigned_i.append(assigned_ii)
#
#                #assigned_i.append(model_init.predict(DTE_pre))
#                #scores_i.append(model_init.getScores(DTE_pre))
#
#        assigned_all.append(assigned_i)
#        scores_all.append(scores_i)
#        if progress:
#            tot_time += time.time() - start_time
#            print('\x1b[1A]\x1b[2Ksplit completed in %.2fs - eta for completion is %.2fs' % (time.time() - start_time, (tot_time * (n_split - i_split - 1) / (i_split + 1))))
#    if progress: print('all splits completed in %.2fs' % tot_time)
#
#    considered = np.hstack(considered)
#    n_cons = considered.size
#
#    if save:
#        try: f = open(filename, 'a')
#        except: f = open(filename, 'w')
#
#    place = 0
#    for prep_ite in prepro:
#
#        i_mp = 0
#        for model_param in model_params:
#
#            assigned = np.hstack([assigned_all[i][place] for i in range(n_split)])
#            scores = np.hstack([scores_all[i][place] for i in range(n_split)])
#            place += 1
#
#            error_rate = (n_cons - (assigned == considered).sum()) * 100 / n_cons
#
#            FNR = (assigned[considered==1]==0).sum() / (considered==1).sum()
#            FPR = (assigned[considered==0]==1).sum() / (considered==0).sum()
#
#            DCF05 = FNR + FPR
#            DCF01 = (0.1 * FNR + 0.9 * FPR) / 0.1
#
#            ##
#            mini05 = 100
#            mini01 = 100
#            sort_idx = np.argsort(scores)
#            scores_s = scores[sort_idx]
#            LTE_s = considered[sort_idx]
#
#            for thresh in scores_s:
#                assigned_m = scores_s > thresh
#
#                FNR = (assigned_m[LTE_s==1]==0).sum() / (LTE_s==1).sum()
#                FPR = (assigned_m[LTE_s==0]==1).sum() / (LTE_s==0).sum()
#
#                DCF05_in = (FNR + FPR)
#                if DCF05_in < mini05: mini05 = DCF05_in
#
#                DCF01_in = (0.1 * FNR + 0.9 * FPR) / 0.1
#                if DCF01_in < mini01: mini01 = DCF01_in
#            ##
#
#            spec = ''
#            for prep in prep_ite: spec += f'{prep[0].getName()} with {prep[1]}  '
#            spec += ' -- params: '
#            if model_params_print is None:
#                spec += '['
#                for i_spec in model_param: 
#                    if not callable(i_spec): spec += f'{i_spec}, '
#                    else: spec += 'kernel'
#                spec += ']'
#            else: 
#                spec += str(model_params_print[i_mp])
#                i_mp += 1
#            if print_act:
#                print_string = str(spec.ljust(90)  + ''
#                                                + '->   minMetric: ' + str('%.3f' % ((mini05 + mini01) / 2))
#                                                + ' \tminDCF(05, 01): (' + str('%.3f' % mini05) + ', ' + str('%.3f' % mini01) + ')'
#                                                + ' \t   metric: ' + str('%.3f' % ((DCF05 + DCF01) / 2))
#                                                + ' \tminDCF(05, 01): (' + str('%.3f' % DCF05) + ', ' + str('%.3f' % DCF01) + ')'
#                                )
#            else:
#                print_string = str(spec.ljust(90)  + ''
#                                                + '->   minMetric: ' + str('%.3f' % ((mini05 + mini01) / 2))
#                                                + ' \tminDCF(05, 01): (' + str('%.3f' % mini05) + ', ' + str('%.3f' % mini01) + ')'
#                                )
#                
#            if print_err: print_string += f' - err: {error_rate}'
#
#            #if model_params_print is not None: 
#            #    print(model_params_print[i_mp])
#            #    i_mp += 1
#            print(print_string)
#            if save: f.write(print_string + '\n')
#
#    if save: f.close()