import numpy as np

import My_Lib_ML.preprocessing as pre
import My_Lib_ML.svm as svm

DTR_base = np.load('DTR_wines.npy')
LTR = np.load('LTR_wines.npy')

n_class = 2
M_base, NTR = DTR_base.shape
print(DTR_base.shape)


####  euclidean classifier  ####

save = True
newfile = False
filename = 'results/kernel_svm_poly2.txt'
progress = True
n_split = 100
prior_range = [1/2]#[1/2, 1/3, False]
prepre_range = ['stand']#[False, 'stand', 'gauss']
mPCA_range_base = [False]
mLDA_range_base = [False]
d = 2
c_range = [1.]#[0.01, 1., 100.]
C_range = [1.]
K_range = [1e-6]

np.random.seed(0)
idx = np.random.permutation(NTR)
DTR_perm = DTR_base[:, idx]
LTR_perm = LTR[idx]

step = NTR / n_split
considered = []
scores_all = []

for i in range(n_split):
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

    considered.append(Lte)
    scores_i = []
    place = 0
    
    if progress: print('-')
    for prior in prior_range:
        for prepre in prepre_range:

            if prepre:
                if prepre == 'stand': 
                    Dtr_pre, mu, std = pre.standardize(Dtr_base)
                    Dte_pre = (Dte_base - mu) / std
                if prepre == 'gauss': 
                    Dtr_pre = pre.gaussianize(Dtr_base, Dtr_base)
                    Dte_pre = pre.gaussianize(Dte_base, Dtr_base)
                mPCA_range = mPCA_range_base
                mLDA_range = mLDA_range_base
            else: 
                Dtr_pre = np.copy(Dtr_base)
                Dte_pre = np.copy(Dte_base)
                mPCA_range = [False]
                mLDA_range = [False]

            for mPCA in mPCA_range:

                if mPCA: 
                    PCA_mat = pre.get_PCA_matrix(Dtr_pre, mPCA)
                    Dtr_p = PCA_mat.T @ Dtr_pre
                    Dte_p = PCA_mat.T @ Dte_pre
                else: 
                    Dtr_p = np.copy(Dtr_pre)
                    Dte_p = np.copy(Dte_pre)

                for mLDA in mLDA_range:

                    if mLDA: 
                        LDA_mat = pre.get_LDA_matrix(Dtr_p, Ltr, 2, mLDA)
                        Dtr_l = LDA_mat.T @ Dtr_p
                        Dte_l = LDA_mat.T @ Dte_p
                    else: 
                        Dtr_l = np.copy(Dtr_p)
                        Dte_l = np.copy(Dte_p)

                    
                    for c in c_range:
                        kernel_f = svm.get_kernel_poly(c, d)
                        for C in C_range:
                            for K in K_range:
                                if progress: print('(' + str(place) + ' - ' + str(i) + '/' + str(n_split) + ')')
                                scores_i.append(svm.SVM(Dtr_l, Ltr, Dte_l, 0, K, C, kernel_f, prior, retScores= True))
                                place += 1

    scores_all.append(scores_i)

considered = np.hstack(considered)
n_cons = considered.size

if save: 
    if newfile: f = open(filename, 'w')
    else: f = open(filename, 'a')

place = 0
for prior in prior_range:
    for prepre in prepre_range:
        if prepre:
            mPCA_range = mPCA_range_base
            mLDA_range = mLDA_range_base
        else: 
            mPCA_range = [False]
            mLDA_range = [False]
        for mPCA in mPCA_range:
            for mLDA in mLDA_range:
                if save: f.write('\n')
                for c in c_range:
                    if save: f.write('\n')
                    print('\n')
                    for C in C_range:
                        for K in K_range:
                            #assigned = np.hstack([assigned_all[i][place] for i in range(n_split)])
                            scores = np.hstack([scores_all[i][place] for i in range(n_split)])
                            np.save('scores_poly2', scores)
                            np.save('cons_poly2', considered)
                            place += 1

                            effective = 1/2
                            assigned = np.array([0 if scores[i]<0 else 1 for i in range(n_cons)], dtype= int)
                            correct02 = (n_cons - (assigned == considered).sum()) * 100 / n_cons
                            FNR = (assigned[considered==1]==0).sum() / (considered==1).sum()
                            FPR = (assigned[considered==0]==1).sum() / (considered==0).sum()
                            DCFu = effective * FNR + (1 - effective) * FPR
                            Bdummy = min(effective, 1 - effective)
                            DCF05 = DCFu / Bdummy

                            effective = 1/3
                            assigned = np.array([0 if scores[i]<-np.log(0.5) else 1 for i in range(n_cons)], dtype= int)
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

                            spec = str('(' + str('%.2f' % prior) + ', ' + str(prepre if prepre else 'raw  ') + ', ' + str(mPCA if mPCA==10 else (str(mPCA) + ' ' if mPCA else 'no')) + ', ' + str(str(mLDA) + ' ' if mLDA else 'no') + ', ' + str(c) + '\t, ' + str(C) + '\t, ' + str(K) + '\t)')
                            print_string = str(spec + '->   err 1/2: ' + str('%.2f' % correct02)
                                                    + '  err 1/3: ' + str('%.2f' % correct03)
                                                    + '  |  minDCF 1/2: ' + str('%.3f' % mini05)
                                                    + '  minDCF 1/3: ' + str('%.3f' % mini03)
                                                    + '  |  DCF 1/2: ' + str('%.3f' % DCF05)
                                                    + '  DCF 1/3: ' + str('%.3f' % DCF03))
                            print(print_string)
                            if save: f.write(print_string + '\n')
place = 0
if save: f.close()