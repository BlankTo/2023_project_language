import numpy as np
import matplotlib.pyplot as plp

import preprocessing as pre
import data_analysis as da

DTR_base = np.load('DTR_language.npy')
LTR = np.load('LTR_language.npy')

n_class = 2
M_base, NTR = DTR_base.shape
print(DTR_base.shape)

print(DTR_base[:, LTR==0].shape)
print(DTR_base[:, LTR==1].shape)

####  Pearson correlation  ####

#for prepre in ['', 'standard', 'gauss']:
#    if prepre:
#        if prepre == 'standard': DTR_pre, _, _ = pre.standardize(DTR_base)
#        if prepre == 'gauss': DTR_pre = pre.gaussianize(DTR_base, DTR_base)
#    else: DTR_pre = np.copy(DTR_base)
#
#    for mPCA in [False, 7]:
#        if prepre or not mPCA:
#            if mPCA: DTR_p = pre.get_PCA_matrix(DTR_pre, mPCA).T @ DTR_pre
#            else: DTR_p = np.copy(DTR_pre)
#
#            print('\n--------------------------------------')
#            print(str(prepre + str(' PCA' + str(mPCA) if mPCA else '') if prepre else 'noPre'))
#            da.PearsonAnalysis(DTR_p, 'All', value= 0.6)
#            da.PearsonAnalysis(DTR_p[:, LTR==0], 'Poor', value= 0.6)
#            da.PearsonAnalysis(DTR_p[:, LTR==1], 'Good', value= 0.6)
#            print('______________________________________\n')


####  check on features 6 and 7  ####

#DTR = DTR_base
#m1 = 6
#m2 = 7
#plp.figure()
##plp.scatter(DTR[m1], DTR[m2], s= 10)
#plp.scatter(DTR[m1, LTR==0], DTR[m2, LTR==0], color= 'r', s= 10)
#plp.scatter(DTR[m1, LTR==1], DTR[m2, LTR==1], color= 'g', s= 10)
#plp.show()


####  PCA analysis  ####

for prepre in [False]:#, 'norm', 'stand', 'gauss']:
    if prepre:
        if prepre == 'norm': DTR, _ = pre.normalize(DTR_base)
        if prepre == 'stand': DTR, _, _ = pre.standardize(DTR_base)
        if prepre == 'gauss': DTR = pre.gaussianize(DTR_base, DTR_base)
    else: DTR = DTR_base

    PCA_mat, eig_val = pre.get_PCA_matrix(DTR, 6, True)

    eig_sum = eig_val.sum()
    perc_var = eig_val*100/eig_val.sum()
    print(str(prepre))
    print(perc_var.reshape(perc_var.size, 1))

    x_plot = np.linspace(0, perc_var.size-1, perc_var.size)
    y_plot = np.linspace(0, 100, 21)
    plp.figure()
    plp.title('percent variance - ' + str(prepre))
    plp.grid()
    plp.xticks(x_plot)
    plp.yticks(y_plot)
    plp.plot(x_plot, perc_var)#, s= 10)
    plp.show()

    x_plot = np.linspace(0, perc_var.size, perc_var.size)
    cpv = np.array([perc_var[0:m].sum() for m in range(1, perc_var.size)])
    plp.figure()
    plp.title('cumulative percent variance - ' + str(prepre))
    plp.grid()
    plp.xticks(x_plot)
    plp.yticks(y_plot)
    plp.plot(x_plot, np.hstack([np.array([0.]), cpv]))
    plp.plot(np.array([0, 6]), np.array([95, 95]))
    plp.show()


####  Data Visualization  ####

#gauss_range = [False, True]
#PCA_range = [False, 7, 6] #il risultato di 9 si vede da 10 per gli hist
#LDA_range = [False, 1]
#hists = True
#scatters = True
#
#for prepre in ['', 'standard', 'gauss']:
#    if prepre:
#        if prepre == 'standard': DTR_pre, _, _ = pre.standardize(DTR_base)
#        if prepre == 'gauss': DTR_pre = pre.gaussianize(DTR_base, DTR_base)
#    else: DTR_pre = np.copy(DTR_base)
#
#    for mPCA in PCA_range:
#        if prepre or not mPCA:
#            if mPCA: DTR_p = pre.get_PCA_matrix(DTR_pre, mPCA).T @ DTR_pre
#            else: DTR_p = np.copy(DTR_pre)
#
#            if scatters:
#                da.plotSparse(DTR_p, LTR, mPCA if mPCA else M_base, 2, title= str(prepre + str(' PCA' + str(mPCA) if mPCA else '') if prepre else 'noPre'))
#
#            if hists:
#                for mLDA in LDA_range:
#                    if prepre or not mLDA:
#                        if mLDA: DTR_l = pre.get_LDA_matrix(DTR_p, LTR, 2, mLDA).T @ DTR_p
#                        else: DTR_l = np.copy(DTR_p)
#
#                        if mLDA or mPCA==10:
#                            M_l = mLDA if mLDA else (mPCA if mPCA else M_base)
#                            title = str(prepre + str(' PCA' + str(mPCA) if mPCA else '') + str(' LDA' + str(mLDA) if mLDA else '') if prepre else 'noPre')
#                            if M_l>1: da.plotHist(DTR_l, LTR, M_l, 2, title= title)
#                            else:
#                                plp.figure()
#                                plp.title(title)
#                                plp.hist(DTR_l[0, LTR==0], density=True, ls='dashed', alpha = 0.5, lw=3, color= 'r', label= 'poor wines')
#                                plp.hist(DTR_l[0, LTR==1], density=True, ls='dotted', alpha = 0.5, lw=3, color= 'g', label= 'good wines')
#                                plp.legend()
#                                plp.show()