import numpy as np
import My_Lib_ML.preprocessing as pre
import My_Lib_ML.log_reg as log_reg
import matplotlib.pyplot as plp

DTR_base = np.load('scores/scores_poly2.npy')
print(DTR_base.shape)
NTR = DTR_base.size

LTR = np.load('cons.npy')
print(LTR.shape)

#DTR_sta, _, _ = pre.standardize(DTR_base.reshape(1, NTR))
#print((0, ((NTR - (np.array(DTR_sta>0, dtype= int)==LTR).sum()) * 100 / NTR)))
#print((-np.log(0.5), ((NTR - (np.array(DTR_sta>-np.log(0.5), dtype= int)==LTR).sum()) * 100 / NTR)))
#
#DTR_gau = pre.gaussianize(DTR_base.reshape(1, NTR), DTR_base.reshape(1, NTR))
#print((0, ((NTR - (np.array(DTR_gau>0, dtype= int)==LTR).sum()) * 100 / NTR)))
#print((-np.log(0.5), ((NTR - (np.array(DTR_gau>-np.log(0.5), dtype= int)==LTR).sum()) * 100 / NTR)))

xplot = [1/10, 1/9, 1/8, 1/7, 1/6, 1/5, 1/4, 1/3, 1/2, 2/3, 3/4, 4/5, 5/6, 6/7, 7/8, 8/9, 9/10]

##
minis = []
thresholds = []
DCFs = []
sort_idx = np.argsort(DTR_base)
DTR_s = DTR_base[sort_idx]
LTE_s = LTR[sort_idx]
for effective in xplot:

    assigned = np.array(DTR_base>-np.log(effective/(1-effective)), dtype= int)

    FNR = (assigned[LTR==1]==0).sum() / (LTR==1).sum()
    FPR = (assigned[LTR==0]==1).sum() / (LTR==0).sum()

    
    DCFu = effective * FNR + (1 - effective) * FPR
    Bdummy = min(effective, 1 - effective)
    DCF_in = DCFu / Bdummy
    DCFs.append(DCF_in)

    if effective==1/3: DCF35 = FNR + FPR
    if effective==1/2: DCF53 = FNR + 2 * FPR

    mini = 100
    for i in range(NTR):
        assigned = np.hstack([np.zeros(i), np.ones(NTR-i)])

        FNR = (assigned[LTE_s==1]==0).sum() / (LTE_s==1).sum()
        FPR = (assigned[LTE_s==0]==1).sum() / (LTE_s==0).sum()

        
        DCFu = effective * FNR + (1 - effective) * FPR
        Bdummy = min(effective, 1 - effective)
        DCF_in = DCFu / Bdummy
        if DCF_in < mini: 
            mini = DCF_in
            threshold = DTR_s[i]
    minis.append(mini)
    thresholds.append(threshold)
##

DCFs_cross3 = []
assigned_cross = np.array((DTR_base>thresholds[7]), dtype= int)
for effective in xplot:

    FNR = (assigned_cross[LTR==1]==0).sum() / (LTR==1).sum()
    FPR = (assigned_cross[LTR==0]==1).sum() / (LTR==0).sum()

    
    DCFu = effective * FNR + (1 - effective) * FPR
    Bdummy = min(effective, 1 - effective)
    DCF_in = DCFu / Bdummy
    DCFs_cross3.append(DCF_in)

DCFs_cross5 = []
assigned_cross = np.array((DTR_base>thresholds[8]), dtype= int)
for effective in xplot:

    FNR = (assigned_cross[LTR==1]==0).sum() / (LTR==1).sum()
    FPR = (assigned_cross[LTR==0]==1).sum() / (LTR==0).sum()

    
    DCFu = effective * FNR + (1 - effective) * FPR
    Bdummy = min(effective, 1 - effective)
    DCF_in = DCFu / Bdummy
    DCFs_cross5.append(DCF_in)

##

lam = 0
prior = 1/2

DTR_log_reg, _, _ = pre.standardize(DTR_base.reshape(1, NTR))
a_stand05, b_stand05 = log_reg.binary_linear_log_reg(DTR_log_reg, LTR, 0, 0, lam, prior, retModel= True)
scores_log_reg = DTR_base * a_stand05.reshape(1) + b_stand05.reshape(1)

DCFs_log_reg_stand05 = []
sort_idx_log_reg = np.argsort(scores_log_reg)
DTR_s_log_reg = scores_log_reg[sort_idx_log_reg]
LTE_s_log_reg = LTR[sort_idx_log_reg]
for effective in xplot:

    assigned = np.array(scores_log_reg>-np.log(effective/(1-effective)), dtype= int)

    FNR = (assigned[LTR==1]==0).sum() / (LTR==1).sum()
    FPR = (assigned[LTR==0]==1).sum() / (LTR==0).sum()

    
    DCFu = effective * FNR + (1 - effective) * FPR
    Bdummy = min(effective, 1 - effective)
    DCF_in = DCFu / Bdummy
    DCFs_log_reg_stand05.append(DCF_in)

##
DTR_log_reg = pre.gaussianize(DTR_base.reshape(1, NTR), DTR_base.reshape(1, NTR))
a_gau05, b_gau05 = log_reg.binary_linear_log_reg(DTR_log_reg, LTR, 0, 0, lam, prior, retModel= True)
scores_log_reg = DTR_base * a_gau05.reshape(1) + b_gau05.reshape(1)

DCFs_log_reg_gau05 = []
sort_idx_log_reg = np.argsort(scores_log_reg)
DTR_s_log_reg = scores_log_reg[sort_idx_log_reg]
LTE_s_log_reg = LTR[sort_idx_log_reg]
for effective in xplot:

    assigned = np.array(scores_log_reg>-np.log(effective/(1-effective)), dtype= int)

    FNR = (assigned[LTR==1]==0).sum() / (LTR==1).sum()
    FPR = (assigned[LTR==0]==1).sum() / (LTR==0).sum()

    
    DCFu = effective * FNR + (1 - effective) * FPR
    Bdummy = min(effective, 1 - effective)
    DCF_in = DCFu / Bdummy
    DCFs_log_reg_gau05.append(DCF_in)

prior = 1/3

DTR_log_reg, _, _ = pre.standardize(DTR_base.reshape(1, NTR))
a_stand03, b_stand03 = log_reg.binary_linear_log_reg(DTR_log_reg, LTR, 0, 0, lam, prior, retModel= True)
scores_log_reg = DTR_base * a_stand03.reshape(1) + b_stand03.reshape(1)

DCFs_log_reg_stand03 = []
sort_idx_log_reg = np.argsort(scores_log_reg)
DTR_s_log_reg = scores_log_reg[sort_idx_log_reg]
LTE_s_log_reg = LTR[sort_idx_log_reg]
for effective in xplot:

    assigned = np.array(scores_log_reg>-np.log(effective/(1-effective)), dtype= int)

    FNR = (assigned[LTR==1]==0).sum() / (LTR==1).sum()
    FPR = (assigned[LTR==0]==1).sum() / (LTR==0).sum()

    
    DCFu = effective * FNR + (1 - effective) * FPR
    Bdummy = min(effective, 1 - effective)
    DCF_in = DCFu / Bdummy
    DCFs_log_reg_stand03.append(DCF_in)

##
DTR_log_reg = pre.gaussianize(DTR_base.reshape(1, NTR), DTR_base.reshape(1, NTR))
a_gau03, b_gau03 = log_reg.binary_linear_log_reg(DTR_log_reg, LTR, 0, 0, lam, prior, retModel= True)
scores_log_reg = DTR_base * a_gau03.reshape(1) + b_gau03.reshape(1)

DCFs_log_reg_gau03 = []
sort_idx_log_reg = np.argsort(scores_log_reg)
DTR_s_log_reg = scores_log_reg[sort_idx_log_reg]
LTE_s_log_reg = LTR[sort_idx_log_reg]
for effective in xplot:

    assigned = np.array(scores_log_reg>-np.log(effective/(1-effective)), dtype= int)

    FNR = (assigned[LTR==1]==0).sum() / (LTR==1).sum()
    FPR = (assigned[LTR==0]==1).sum() / (LTR==0).sum()

    
    DCFu = effective * FNR + (1 - effective) * FPR
    Bdummy = min(effective, 1 - effective)
    DCF_in = DCFu / Bdummy
    DCFs_log_reg_gau03.append(DCF_in)

##




plp.figure()
plp.plot(xplot, DCFs, color= 'r', label= 'baseDCF')
plp.plot(xplot, minis, color= 'g', label= 'minDCF')
plp.plot(xplot, DCFs_log_reg_stand05, label= 'logDCF_stand05')
plp.plot(xplot, DCFs_log_reg_gau05, color= 'b', label= 'logDCF_gau05')
plp.plot(xplot, DCFs_log_reg_stand03, label= 'logDCF_stand03')
plp.plot(xplot, DCFs_log_reg_gau03, label= 'logDCF_gau03')
plp.plot(xplot, DCFs_cross3, color= 'y', label= 'cross03')
plp.plot(xplot, DCFs_cross5, color= 'm', label= 'cross05')
plp.legend()
plp.show()
##

print('base03 -> ' + str('%.3f' % DCF35) + '(' + str('%.3f' % DCFs[7]) + ')')
print('base05 -> ' + str('%.3f' % DCFs[8]) + '(' + str('%.3f' % DCF53) + ')')
print('log_reg_stand03_threshold: ' + str(a_stand03) + ', ' + str(b_stand03) + ' -> ' + str('%.3f' % DCFs_log_reg_stand03[8]) + '(' + str('%.3f' % DCFs_log_reg_stand03[7]) + ')')
print('log_reg_stand05_threshold: ' + str(a_stand05) + ', ' + str(b_stand05) + ' -> ' + str('%.3f' % DCFs_log_reg_stand05[8]) + '(' + str('%.3f' % DCFs_log_reg_stand05[7]) + ')')
print('log_reg_gau03_threshold: ' + str(a_gau03) + ', ' + str(b_gau03) + ' -> ' + str('%.3f' % DCFs_log_reg_gau03[8]) + '(' + str('%.3f' % DCFs_log_reg_gau03[7]) + ')')
print('log_reg_gau05_threshold: ' + str(a_gau05) + ', ' + str(b_gau05) + ' -> ' + str('%.3f' % DCFs_log_reg_gau05[8]) + '(' + str('%.3f' % DCFs_log_reg_gau05[7]) + ')')
print('cross03_threshold: ' + str(thresholds[7]) + ' -> ' + str('%.3f' % DCFs_cross3[8]) + '(' + str('%.3f' % DCFs_cross3[7]) + ')')
print('cross05_threshold: ' + str(thresholds[8]) + ' -> ' + str('%.3f' % DCFs_cross5[8]) + '(' + str('%.3f' % DCFs_cross5[7]) + ')')