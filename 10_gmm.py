import numpy as np
import os

import My_Lib_ML.preprocessing as pre
import My_Lib_ML.gaussian as gau

DTR_base = np.load('DTR_wines.npy')
LTR_base = np.load('LTR_wines.npy')

n_class = 2
M, NTR = DTR_base.shape

M, N = DTR_base.shape
np.random.seed(0)
idx = np.random.permutation(N)
DTR_perm = DTR_base[:, idx]
LTR_perm = LTR_base[idx]

pTR = 1/3
pT = 1/3
n_split = 100
n_pow = 6
maxG = 2**n_pow
alpha = 0.1
bound = 0.01
prepre_range = [False, 'stand', 'gauss']
mPCA_range_base = [False, 10, 9]
mLDA_range_base = [False, 1]

step = N / n_split
considered = []
for i in range(n_split):
    Lte = []
    for j in range(n_split):
        if i==j:
            Lte = LTR_base[int(step * j):int(step * (j+1))]
    considered.append(Lte)
considered = np.hstack(considered)
n_cons = considered.size

print_string = 'n_split= ' + str(n_split) + '  -  pT= ' + str(pT) + '\n(gauss, PCA, LDA, alpha, bound, variant, nG)\n'

for prepre in prepre_range:
    if prepre:
        mPCA_range = mPCA_range_base
        mLDA_range = mLDA_range_base
    else:
        mPCA_range = [False]
        mLDA_range = [False]
    for mPCA in mPCA_range:
        for mLDA in mLDA_range:
            for variant in ['DIAG', 'TIED']:
                print_string += '\n'
                scores = []
                for i in range(n_split):
                    print_string_1 = print_string + '\n' + str((prepre, mPCA, mLDA, alpha, bound, variant)) + ' -> ' +  str(i)
                    os.system('cls' if os.name == 'nt' else 'clear')
                    print(print_string_1)
                    Dtr_base = []
                    Ltr = []
                    Dte_base = []
                    Lte = []
                    for j in range(n_split):
                        if not i==j:
                            Dtr_base.append(DTR_perm[:, int(step * j):int(step * (j+1))])
                            Ltr.append(LTR_perm[int(step * j):int(step * (j+1))])
                        else:
                            Dte_base = DTR_perm[:, int(step * j):int(step * (j+1))]
                            Lte = LTR_perm[int(step * j):int(step * (j+1))]
                    Dtr_base = np.hstack(Dtr_base)
                    Ltr = np.hstack(Ltr)

                    if prepre:
                        if prepre == 'stand': 
                            Dtr_pre, mu, std = pre.standardize(Dtr_base)
                            Dte_pre = (Dte_base - mu) / std
                        if prepre == 'gauss': 
                            Dtr_pre = pre.gaussianize(Dtr_base, Dtr_base)
                            Dte_pre = pre.gaussianize(Dte_base, Dtr_base)
                    else: 
                        Dtr_pre = np.copy(Dtr_base)
                        Dte_pre = np.copy(Dte_base)

                    if mPCA: 
                        PCA_mat = pre.get_PCA_matrix(Dtr_pre, mPCA)
                        Dtr_p = PCA_mat.T @ Dtr_pre
                        Dte_p = PCA_mat.T @ Dte_pre
                    else: 
                        Dtr_p = np.copy(Dtr_pre)
                        Dte_p = np.copy(Dte_pre)

                    if mLDA: 
                        LDA_mat = pre.get_LDA_matrix(Dtr_p, Ltr, 2, mLDA)
                        Dtr_l = LDA_mat.T @ Dtr_p
                        Dte_l = LDA_mat.T @ Dte_p
                    else: 
                        Dtr_l = np.copy(Dtr_p)
                        Dte_l = np.copy(Dte_p)

                    print('poor')
                    GMMs_poor = gau.LBG_algo(Dtr_l[:, Ltr==0], alpha, maxG, bound, variant, messages_LBG = True, retAll= True)
                    print('good')
                    GMMs_good = gau.LBG_algo(Dtr_l[:, Ltr==1], alpha, maxG, bound, variant, messages_LBG = True, retAll= True)
                    scores_in = []
                    for kk in range(n_pow+1):
                        print_string_2 = print_string_1 + str('-' + str(2*kk))
                        os.system('cls' if os.name == 'nt' else 'clear')
                        print(print_string_2)
                        scores_in.append(gau.use_GMM([GMMs_poor[kk], GMMs_good[kk]], Dte_l, retScores= True))
                    scores_in = np.vstack(scores_in)
                    scores.append(scores_in)
                scores = np.hstack(scores)

                for kk in range(n_pow+1):

                    effective = 1/2
                    assigned = np.array([0 if scores[kk][i]<0 else 1 for i in range(n_cons)], dtype= int)
                    correct02 = (n_cons - (assigned == considered).sum()) * 100 / n_cons
                    FNR = (assigned[considered==1]==0).sum() / (considered==1).sum()
                    FPR = (assigned[considered==0]==1).sum() / (considered==0).sum()
                    DCFu = effective * FNR + (1 - effective) * FPR
                    Bdummy = min(effective, 1 - effective)
                    DCF05 = DCFu / Bdummy

                    effective = 1/3
                    assigned = np.array([0 if scores[kk][i]<-np.log(0.5) else 1 for i in range(n_cons)], dtype= int)
                    correct03 = (n_cons - (assigned == considered).sum()) * 100 / n_cons
                    FNR = (assigned[considered==1]==0).sum() / (considered==1).sum()
                    FPR = (assigned[considered==0]==1).sum() / (considered==0).sum()
                    DCFu = effective * FNR + (1 - effective) * FPR
                    Bdummy = min(effective, 1 - effective)
                    DCF03 = DCFu / Bdummy

                    ##
                    mini03 = 100
                    mini05 = 100
                    sort_idx = np.argsort(scores)
                    scores = scores[sort_idx]
                    LTE_s = considered[sort_idx]

                    for i in range(n_cons):
                        assigned = np.hstack([np.zeros(i), np.ones(n_cons-i)])

                        FNR = (assigned[LTE_s==1]==0).sum() / (LTE_s==1).sum()
                        FPR = (assigned[LTE_s==0]==1).sum() / (LTE_s==0).sum()

                        effective = 1/3
                        DCFu = effective * FNR + (1 - effective) * FPR
                        Bdummy = min(effective, 1 - effective)
                        DCF03_in = DCFu / Bdummy
                        if DCF03_in < mini03: mini03 = DCF03_in

                        effective = 1/2
                        DCFu = effective * FNR + (1 - effective) * FPR
                        Bdummy = min(effective, 1 - effective)
                        DCF05_in = DCFu / Bdummy
                        if DCF05_in < mini05: mini05 = DCF05_in
                    ##
                    spec = str('(' + str(prepre if prepre else 'raw  ') + ', ' + str(mPCA if mPCA==10 else (str(mPCA) + ' ' if mPCA else 'no')) + ', ' + str(str(mLDA) + ' ' if mLDA else 'no') + ', ' + str(2**kk) + '\t)')
                    print_string = str(spec + '->   err 1/2: ' + str('%.2f' % correct02)
                                            + '  err 1/3: ' + str('%.2f' % correct03)
                                            + '  |  minDCF 1/2: ' + str('%.3f' % mini05)
                                            + '  minDCF 1/3: ' + str('%.3f' % mini03)
                                            + '  |  DCF 1/2: ' + str('%.3f' % DCF05)
                                            + '  DCF 1/3: ' + str('%.3f' % DCF03))
                    print(print_string)
                    with open('results/gmm_results.txt', 'w') as f:
                        f.write(print_string)