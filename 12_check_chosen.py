import numpy as np

import My_Lib_ML.preprocessing as pre
import My_Lib_ML.log_reg as log_reg
import My_Lib_ML.svm as svm

DTR = np.load('DTR_wines.npy')
LTR = np.load('LTR_wines.npy')

n_class = 2
M, NTR = DTR.shape
print(DTR.shape)

DTE = np.load('DTE_wines.npy')
LTE = np.load('LTE_wines.npy')

NTE = DTE.shape[1]
print(DTE.shape)
#threshold_range = [0, 1.5404166240266687e-05, -np.log(0.5)] # rbf
threshold_range = [0, 0.12861949789654048, 0.4987071939137113, -np.log(0.5)] # poly2

prior = 1/2
prepre = 'stand'
mPCA = False
mLDA = False
c = 0.1
lam = 1.
C = 1.
K = 1e-6
if prepre:
    if prepre == 'stand': 
        Dtr_pre, mu, std = pre.standardize(DTR)
        Dte_pre = (DTE - mu) / std
    if prepre == 'gauss': 
        Dtr_pre = pre.gaussianize(DTR, DTR)
        Dte_pre = pre.gaussianize(DTE, DTR)
else: 
    Dtr_pre = np.copy(DTR)
    Dte_pre = np.copy(DTE)

if mPCA: 
    PCA_mat = pre.get_PCA_matrix(Dtr_pre, mPCA)
    Dtr_p = PCA_mat.T @ Dtr_pre
    Dte_p = PCA_mat.T @ Dte_pre
else: 
    Dtr_p = np.copy(Dtr_pre)
    Dte_p = np.copy(Dte_pre)

if mLDA: 
    LDA_mat = pre.get_LDA_matrix(Dtr_p, LTR, 2, mLDA)
    Dtr_l = LDA_mat.T @ Dtr_p
    Dte_l = LDA_mat.T @ Dte_p
else: 
    Dtr_l = np.copy(Dtr_p)
    Dte_l = np.copy(Dte_p)

#scores = log_reg.binary_quadratic_log_reg(Dtr_l, LTR, Dte_l, 0, lam, prior, retScores= True)
#scores = svm.SVM(Dtr_l, LTR, Dte_l, 0, K, C, svm.get_kernel_RBF(lam), prior, retScores= True).reshape(NTE, )
scores = svm.SVM(Dtr_l, LTR, Dte_l, 0, K, C, svm.get_kernel_poly(c, 2), prior, retScores= True).reshape(NTE, )

tit = ['stand03', 'stand05', 'gau03', 'gau05']
gg = 0
#for a, b in [(3.11387703, -1.4721455425645245), (3.09579339, -0.7779630743424855), (4.04535188, -1.8972802926298407), (3.90882307, -1.175265386531638)]: #rbf
for a, b in [(3.11866692, -1.543937769502656), (3.05921802, -0.8587445674080892), (2.80798796, -1.420971521496354), (2.81129951, -0.7545855485754364)]: #poly2
    scores_log_reg = scores * a + b


    assigned = np.array([0 if scores_log_reg[i]<0 else 1 for i in range(NTE)], dtype= int)

    correct = (NTE - (assigned==LTE).sum()) * 100 / NTE

    FNR = (assigned[LTE==1]==0).sum() / (LTE==1).sum()
    FPR = (assigned[LTE==0]==1).sum() / (LTE==0).sum()

    effective = 1/2
    DCFu = effective * FNR + (1 - effective) * FPR
    Bdummy = min(effective, 1 - effective)
    DCF05 = DCFu / Bdummy

    effective = 1/3
    DCFu = effective * FNR + (1 - effective) * FPR
    Bdummy = min(effective, 1 - effective)
    DCF03 = DCFu / Bdummy
    print_string = str(tit[gg] + ' - 05 ->\terr: ' + str('%.2f' % correct)
                                + '\tdcf 1/2: ' + str('%.3f' % DCF05)
                                + '\tdcf 1/3: ' + str('%.3f' % DCF03)
                                )
    print(print_string)

    assigned = np.array([0 if scores_log_reg[i]<-np.log(0.5) else 1 for i in range(NTE)], dtype= int)

    correct = (NTE - (assigned==LTE).sum()) * 100 / NTE

    FNR = (assigned[LTE==1]==0).sum() / (LTE==1).sum()
    FPR = (assigned[LTE==0]==1).sum() / (LTE==0).sum()

    effective = 1/2
    DCFu = effective * FNR + (1 - effective) * FPR
    Bdummy = min(effective, 1 - effective)
    DCF05 = DCFu / Bdummy

    effective = 1/3
    DCFu = effective * FNR + (1 - effective) * FPR
    Bdummy = min(effective, 1 - effective)
    DCF03 = DCFu / Bdummy
    print_string = str(tit[gg] + ' - 03 ->\terr: ' + str('%.2f' % correct)
                                + '\tdcf 1/2: ' + str('%.3f' % DCF05)
                                + '\tdcf 1/3: ' + str('%.3f' % DCF03)
                                )
    print(print_string)
    gg += 1


for threshold in threshold_range:
    assigned = np.array([0 if scores[i]<threshold else 1 for i in range(NTE)], dtype= int)

    correct = (NTE - (assigned==LTE).sum()) * 100 / NTE

    FNR = (assigned[LTE==1]==0).sum() / (LTE==1).sum()
    FPR = (assigned[LTE==0]==1).sum() / (LTE==0).sum()

    effective = 1/2
    DCFu = effective * FNR + (1 - effective) * FPR
    Bdummy = min(effective, 1 - effective)
    DCF05 = DCFu / Bdummy

    effective = 1/3
    DCFu = effective * FNR + (1 - effective) * FPR
    Bdummy = min(effective, 1 - effective)
    DCF03 = DCFu / Bdummy

    ##
    mini03 = 100
    mini05 = 100
    sort_idx = np.argsort(scores)
    LTE_s = LTE[sort_idx]

    for i in range(NTE):
        assigned = np.hstack([np.zeros(i), np.ones(NTE-i)])

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

#    spec = str('(' + str('%.2f' % prior) + ', ' + str('%.2f' % threshold) + ', ' + str(prepre if prepre else 'raw  ') + ', ' + str(mPCA if mPCA==10 else (str(mPCA) + ' ' if mPCA else 'no')) + ', ' + str(str(mLDA) + ' ' if mLDA else 'no') + ', ' + str(lam) + ')')
#    print_string = str(spec + '->\terr: ' + str('%.2f' % correct)
#                            + '\tdcf 1/2: ' + str('%.3f' % DCF05)
#                            + '\tdcf 1/3: ' + str('%.3f' % DCF03))
#    print(print_string)

    print_string = str(str('%.2f' % threshold) + '->\terr: ' + str('%.2f' % correct)
                            + '\tdcf 1/2: ' + str('%.3f' % DCF05)
                            + '\tdcf 1/3: ' + str('%.3f' % DCF03)
                            + '\tmindcf 1/2: ' + str('%.3f' % mini05)
                            + '\tmindcf 1/3: ' + str('%.3f' % mini03)
                            )
    print(print_string)