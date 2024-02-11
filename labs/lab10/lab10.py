import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

import ML_lib.gmm as gmm
import ML_lib.utils as utils
import ML_lib.preprocessing as pre

def save_gmm(gmm, filename):
    gmmJson = [(i, j.tolist(), k.tolist()) for i, j, k in gmm]
    with open(filename, 'w') as f:
        json.dump(gmmJson, f)
    
def load_gmm(filename):
    with open(filename, 'r') as f:
        gmm = json.load(f)
    return [(i, np.asarray(j), np.asarray(k)) for i, j, k in gmm]

###############################

#X_4D = np.load('labs/lab10/GMM_data_4D.npy')
#GMM_4D_3D_init = load_gmm('labs/lab10/GMM_4D_3G_init.json')
#_, my_ll = gmm.log_pdf_GMM(X_4D, GMM_4D_3D_init, 3)
#prof_ll = np.load('labs/lab10/GMM_4D_3G_init_ll.npy')
#print(f'4D diff: {(my_ll - prof_ll).sum()}')
#
#X_1D = np.load('labs/lab10/GMM_data_1D.npy')
#GMM_1D_3D_init = load_gmm('labs/lab10/GMM_1D_3G_init.json')
#_, my_ll = gmm.log_pdf_GMM(X_1D, GMM_1D_3D_init, 3)
#prof_ll = np.load('labs/lab10/GMM_1D_3G_init_ll.npy')
#print(f'1D diff: {(my_ll - prof_ll).sum()}')

###############################

#
#X_4D = np.load('labs/lab10/GMM_data_4D.npy')
#print(X_4D.shape)
#M, N = X_4D.shape
#GMM_4D_3D_init = load_gmm('labs/lab10/GMM_4D_3G_init.json')
#
#print('##############  GMM_init ######################')
#for k in range(3):
#    print(f'---{k}---')
#    print(f'w: {GMM_4D_3D_init[k][0]}')
#    print(f'mu: {GMM_4D_3D_init[k][1]}')
#    print(f'cov: {GMM_4D_3D_init[k][2]}')
#
#my_EM = gmm.EM_algo(X_4D, GMM_4D_3D_init, 3)
#prof_EM = load_gmm('labs/lab10/GMM_4D_3G_EM.json')
#kkk = 0
#print('##############  GMM_EM ######################')
#for k in range(3):
#    print(f'---{k}---')
#    print(f'w: {my_EM[k][0]}')
#    print(f'mu: {my_EM[k][1]}')
#    print(f'cov: {my_EM[k][2]}')
#    print(f'w_err: {prof_EM[k][0] - my_EM[k][0]}')
#    print(f'mu_err: {prof_EM[k][1] - my_EM[k][1]}')
#    print(f'cov_err: {prof_EM[k][2] - my_EM[k][2]}')

##########################

#X_1D = np.load('labs/lab10/GMM_data_1D.npy')
#M, N = X_1D.shape
#GMM_1D_3D_init = load_gmm('labs/lab10/GMM_1D_3G_init.json')
#
#my_EM = gmm.EM_algo(X_1D, GMM_1D_3D_init, 3)
#print(my_EM)
#prof_EM = load_gmm('labs/lab10/GMM_1D_3G_EM.json')
#print(prof_EM)
#plt.figure()
##plt.hist(X_1D.T, bins= 30, density= True, edgecolor='black')
#x_plot = np.linspace(-12, 6, 1000).reshape(1, 1000)
#_, ll_EM = gmm.log_pdf_GMM(x_plot, my_EM, 3)
#plt.plot(x_plot.T, np.exp(ll_EM))
#_, ll_EM = gmm.log_pdf_GMM(x_plot, prof_EM, 3)
#plt.plot(x_plot.T, np.exp(ll_EM), ls= 'dotted')
#plt.show()

##########################


#X_4D = np.load('labs/lab10/GMM_data_4D.npy')
#print(X_4D.shape)
#M, N = X_4D.shape
#
#my_LBG, n_GMM = gmm.LBG_algo(X_4D, 0.1, 4, messages_LBG= True)#, messages_EM= True)
#print('mine')
#print(my_LBG)
#prof_LBG = load_gmm('labs/lab10/GMM_4D_4G_EM_LBG.json')
#print('##############  GMM_LBG ######################')
#for k in range(4):
#    print(f'---{k}---')
#    print(f'w: {my_LBG[k][0]}')
#    print(f'mu: {my_LBG[k][1]}')
#    print(f'cov: {my_LBG[k][2]}')
#    print(f'---{k}---')
#    print(f'w_err: {prof_LBG[k][0] - my_LBG[k][0]}')
#    print(f'mu_err: {prof_LBG[k][1] - my_LBG[k][1]}')
#    print(f'cov_err: {prof_LBG[k][2] - my_LBG[k][2]}')

#X_1D = np.load('labs/lab10/GMM_data_1D.npy')
#print(X_1D.shape)
#M, N = X_1D.shape
#
#my_LBG, n_GMM = gmm.LBG_algo(X_1D, 0.1, 4, messages_LBG= True)#, messages_EM= True)
#print('mine')
#print(my_LBG)
#
#plt.figure()
##plt.hist(X_1D.T, bins= 30, density= True, edgecolor='black')
#x_plot = np.linspace(-12, 6, 1000).reshape(1, 1000)
#_, ll_LBG = gmm.log_pdf_GMM(x_plot, my_LBG, n_GMM)
#plt.plot(x_plot.T, np.exp(ll_LBG))
#prof_LBG = load_gmm('labs/lab10/GMM_1D_4G_EM_LBG.json')
#_, ll_LBG = gmm.log_pdf_GMM(x_plot, prof_LBG, n_GMM)
#plt.plot(x_plot.T, np.exp(ll_LBG), ls= 'dotted')
#plt.show()

################################################


#D, L, label_dict = utils.csv_to_npy('data_raw/iris.csv')
#DTR, LTR, DTE, LTE = utils.shuffle_and_divide(D, L, 2.0/3.0, 0)
#print(DTR.shape)
#print(LTR.shape)
#print(DTE.shape)
#print(LTE.shape)
#print('-----------------------')
#
#for variant in ['', 'naive', 'tied']:
#    print(f'########### {variant} ###########')
#    GMM_class_0 = gmm.LBG_algo(DTR[:, LTR==0], 0.1, 16, bound= 0.01, retAll= True, variant= variant)
#    GMM_class_1 = gmm.LBG_algo(DTR[:, LTR==1], 0.1, 16, bound= 0.01, retAll= True, variant= variant)
#    GMM_class_2 = gmm.LBG_algo(DTR[:, LTR==2], 0.1, 16, bound= 0.01, retAll= True, variant= variant)
#    #for kkk in range(5):
#    #    print(f'n_GMM: {GMM_class_0[kkk][1]}')
#    #    for i_gmm in range(GMM_class_0[kkk][1]):
#    #        print(f'w: {GMM_class_0[kkk][0][i_gmm][0]}')
#    #        print(f'mu: {GMM_class_0[kkk][0][i_gmm][1]}')
#    #        print(f'cov: {GMM_class_0[kkk][0][i_gmm][2]}')
#    #exit()
#
#    for kk in range(5):
#        n_GMM = GMM_class_0[kk][1]
#        print(f'n_GMM: {n_GMM}')
#        gmm_class_0 = GMM_class_0[kk][0]
#        gmm_class_1 = GMM_class_1[kk][0]
#        gmm_class_2 = GMM_class_2[kk][0]
#
#        _, ll_class_0 = gmm.log_pdf_GMM(DTE, gmm_class_0, n_GMM)
#        _, ll_class_1 = gmm.log_pdf_GMM(DTE, gmm_class_1, n_GMM)
#        _, ll_class_2 = gmm.log_pdf_GMM(DTE, gmm_class_2, n_GMM)
#
#        lls = np.vstack([ll_class_0, ll_class_1, ll_class_2])
#        pred = lls.argmax(axis= 0)
#        correct = (pred == LTE).sum()
#        print((LTE.size - correct) * 100 / LTE.size)



##################################


#D = np.load('data_npy/MNIST_data.npy')
#L = np.load('data_npy/MNIST_target.npy')
##
#D = D[:, L < 2]
#L = L[L < 2]
#
#DTR, LTR, DTE, LTE = utils.shuffle_and_divide(D, L, 2.0/3.0, 0)
#print(DTR.shape)
#print(LTR.shape)
#print(DTE.shape)
#print(LTE.shape)
#
#pca = pre.PCA(DTR)
#DTR = pca.transform(DTR, 50)
#DTE = pca.transform(DTE, 50)
#print(DTR.shape)
#print(DTE.shape)
#print('-----------------------')
#
#n_class = len(set(LTR))
#n_max_GMM = 4
#
#GMM_all_class = []
#for c in range(n_class):
#    print(f'############## {c}/{n_class} ##################')
#    GMM_all_class.append(gmm.LBG_algo(DTR[:, LTR==c], 0.1, n_max_GMM, bound= 0.01, retAll= True, messages_LBG= True, variant= ''))
#
#for kk in range(int(np.sqrt(n_max_GMM)) + 1):
#    n_GMM = 2 ** kk
#    print(f'n_GMM: {n_GMM}')
#    ll_all_class = []
#    for c in range(n_class):
#        print(f'-------------- {c}/{n_class} -------------')
#        _, ll_c = gmm.log_pdf_GMM(DTE, GMM_all_class[c][kk][0], n_GMM)
#        ll_all_class.append(ll_c)
#
#    lls = np.vstack(ll_all_class)
#    pred = lls.argmax(axis= 0)
#    correct = (pred == LTE).sum()
#    print(f'error_ rate: {(LTE.size - correct) * 100 / LTE.size}')


#n_class = len(set(L))
#n_max_GMM = 2
#
#print(D.shape)
#print(L.shape)
#
#utils.cross_validation(D, L, 10, gmm.GMMClassifier, [[0.1, n_max_GMM, 0.01, '', False, True]], progress= True, prepro= [
##    [(pre.NoTransform, [])],
##    [(pre.Gaussianizer, [])],
#    [(pre.PCA, [50])],
#    ])

##################


#D_BASE = np.load('data_npy\\DTR_language.npy')
#L = np.load('data_npy\\LTR_language.npy')
#
#N_CLASS = 2
#M_BASE, NTR = D_BASE.shape
#
#model = gmm.GMMClassifier(D_BASE, L, 0.1, 4, 0.01, '', True, messages_LBG= True)
#
#GMMs = model.getGMMs()
#
#for K in range(2):
#    for NN in range(3):
#        print('----------------------------------------------------------------------------------')
#        print('----------------------------------------------------------------------------------')
#        print(f'nG: {GMMs[K][NN][1]}')
#        for NNN in range(GMMs[K][NN][1]):
#            print('----------------------------------------------------------------------------------')
#            print(GMMs[K][NN][0][0][0])
#            print(GMMs[K][NN][0][0][1])
#            print(GMMs[K][NN][0][0][2])