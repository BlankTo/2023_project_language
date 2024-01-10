import numpy as np

import My_Lib_ML.preprocessing as pre
import My_Lib_ML.others as oth

DTR_base = np.load('DTR_pulsar.npy')
LTR = np.load('LTR_pulsar.npy')

n_class = 2
M_base, NTR = DTR_base.shape
print(DTR_base.shape)


####  euclidean classifier  ####

save = True
newfile = False
filename = 'results/mahalanobis.txt'
progress = True
n_split = 100
prepre_range = ['', 'stand', 'gauss']
mPCA_range_base = [False, 7, 6]
mLDA_range_base = [False, 1]

np.random.seed(0)
idx = np.random.permutation(NTR)
DTR_perm = DTR_base[:, idx]
LTR_perm = LTR[idx]

step = NTR / n_split
considered = []
assigned_all = []

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
    assigned_i = []
    place = 0
    
    if progress: print('-')
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

                    
                    if progress: print('(' + str(place) + ' - ' + str(i) + '/' + str(n_split) + ')')
                    assigned_i.append(oth.mahalanobis_classifier(Dtr_l, Ltr, Dte_l, 2))
                    place += 1

    assigned_all.append(assigned_i)

considered = np.hstack(considered)
n_cons = considered.size

if save: 
    if newfile: f = open(filename, 'w')
    else: f = open(filename, 'a')

place = 0
for prepre in prepre_range:
        if prepre:
            mPCA_range = mPCA_range_base
            mLDA_range = mLDA_range_base
        else: 
            mPCA_range = [False]
            mLDA_range = [False]
        for mPCA in mPCA_range:
            for mLDA in mLDA_range:
            
                assigned = np.hstack([assigned_all[i][place] for i in range(n_split)])
                place += 1

                correct = (n_cons - (assigned == considered).sum()) * 100 / n_cons

                FNR = (assigned[considered==1]==0).sum() / (considered==1).sum()
                FPR = (assigned[considered==0]==1).sum() / (considered==0).sum()

                effective = 1/2
                DCFu = effective * FNR + (1 - effective) * FPR
                Bdummy = min(effective, 1 - effective)
                DCF05 = DCFu / Bdummy

                effective = 1/3
                DCFu = effective * FNR + (1 - effective) * FPR
                Bdummy = min(effective, 1 - effective)
                DCF03 = DCFu / Bdummy

                spec = str('(' + str(prepre if prepre else 'raw  ') + ', ' + str(mPCA if mPCA==10 else (str(mPCA) + ' ' if mPCA else 'no')) + ', ' + str(str(mLDA) + ' ' if mLDA else 'no') + ')')
                print_string = str(spec + '\t->\terr: ' + str('%.2f' % correct)
                                        + '\t\tdcf 1/2: ' + str('%.3f' % DCF05)
                                        + '\tdcf 1/3: ' + str('%.3f' % DCF03)
                                        + '\t\tFNR: ' + str('%.3f' % FNR)
                                        + '\tFPR: ' + str('%.3f' % FPR))
                print(print_string)
                if save: f.write(print_string + '\n')
place = 0
if save: f.close()