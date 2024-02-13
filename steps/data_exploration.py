import numpy as np
import matplotlib.pyplot as plt

import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import ML_lib.preprocessing as pre
import ML_lib.plot_lib as pltl

##### loading data #####

D_BASE = np.load('data_npy\\DTR_language.npy')
L = np.load('data_npy\\LTR_language.npy')

N_CLASS = 2
M_BASE, NTR = D_BASE.shape

print('all:  ' + str(D_BASE.shape))
print('class 0:  ' + str(D_BASE[:, L==0].shape))
print('class 1:  ' + str(D_BASE[:, L==1].shape))

## check NaN

print(np.isnan(D_BASE).any())

#################

## check variance

print(np.mean(np.square(D_BASE - D_BASE.mean(axis= 1).reshape(D_BASE.shape[0], 1)), axis= 1).shape)
print(np.mean(np.square(D_BASE - D_BASE.mean(axis= 1).reshape(D_BASE.shape[0], 1)), axis= 1))

##############

##### PCA analysis ####

for prepre in [False, 'gauss']:
    if prepre == 'gauss': DTR = pre.Gaussianizer(D_BASE).transform(D_BASE)
    else: DTR = pre.Standardizer(D_BASE).transform(D_BASE)

    pca = pre.PCA(DTR)
    PCA_mat = pca.getPCAmat()
    eig_val = pca.getEig()

    eig_sum = eig_val.sum()
    perc_var = eig_val*100/eig_val.sum()
    print(str(prepre))
    print(perc_var.reshape(perc_var.size, 1))

    x_plot = np.arange(1, perc_var.size + 1)
    y_plot = np.linspace(0, 100, 21)
    plt.figure()
    plt.title('percent variance - ' + str(prepre))
    plt.grid()
    plt.xticks(x_plot)
    plt.yticks(y_plot)
    plt.plot(x_plot, perc_var)#, s= 10)
    plt.show()

    x_plot = np.arange(0, perc_var.size+1)
    cpv = np.array([perc_var[0:m].sum() for m in range(1, perc_var.size+1)])
    plt.figure()
    plt.title('cumulative percent variance - ' + str(prepre))
    plt.grid()
    plt.xticks(x_plot)
    plt.yticks(y_plot)
    plt.plot(x_plot, np.hstack([np.array([0.]), cpv]))
    plt.plot(np.array([0, 6]), np.array([95, 95]))
    plt.show()

###############

##### correlation #####

###################

## correlation with no preprocessing

pltl.plot_corr_D(
            D_BASE,
            feature_names= ['F1', 'F2', 'F3', 'F4', 'F5', 'F6'],
            title= str('raw')
            )

pltl.plot_corr_D(
            D_BASE[:, L == 0],
            feature_names= ['F1', 'F2', 'F3', 'F4', 'F5', 'F6'],
            title= str('class 0 - raw')
            )

pltl.plot_corr_D(
            D_BASE[:, L == 1],
            feature_names= ['F1', 'F2', 'F3', 'F4', 'F5', 'F6'],
            title= str('class 1 - raw')
            )

### correlation with gaussianized features

D_gau = pre.Gaussianizer(D_BASE).transform(D_BASE)
pltl.plot_corr_D(
            D_gau,
            feature_names= ['F1', 'F2', 'F3', 'F4', 'F5', 'F6'],
            title= str('gau')
            )

pltl.plot_corr_D(
            D_gau[:, L == 0],
            feature_names= ['F1', 'F2', 'F3', 'F4', 'F5', 'F6'],
            title= str('class 0 - gau')
            )

pltl.plot_corr_D(
            D_gau[:, L == 1],
            feature_names= ['F1', 'F2', 'F3', 'F4', 'F5', 'F6'],
            title= str('class 1 - gau')
            )

###########################

############################

# histogram visualization with no preprocessing

pltl.plotHist(D_BASE, L, attr_names= ['F1', 'F2', 'F3', 'F4', 'F5', 'F6'], title= 'no preprocessing', show= False)

# histogram visualization with gaussianized features

D_gau = pre.Gaussianizer(D_BASE).transform(D_BASE)
pltl.plotHist(D_gau, L, attr_names= ['F1', 'F2', 'F3', 'F4', 'F5', 'F6'], title= 'gaussianized features', show= False)

# histogram visualization with PCA-reduced features

D_stand = pre.Standardizer(D_BASE).transform(D_BASE)
pca = pre.PCA(D_stand)
m_pca = 6
pltl.plotHist(pca.transform(D_stand, m_pca), L, attr_names= ['F1', 'F2', 'F3', 'F4', 'F5', 'F6'], title= 'PCA' + str(m_pca) + '-reduced features', show= False)
# histogram visualization with gaussianized and PCA-reduced features

pca = pre.PCA(D_gau)
m_pca = 6
pltl.plotHist(pca.transform(D_gau, m_pca), L, attr_names= ['F1', 'F2', 'F3', 'F4', 'F5', 'F6'], title= 'gaussianized and PCA-reduced features', show= False)

# histogram visualization with LDA-reduced features

lda = pre.LDA(D_BASE, L)
pltl.plotHist(lda.transform(D_BASE), L, title= 'LDA-reduced features', show= False)

# histogram visualization with gaussianized and LDA-reduced features

lda = pre.LDA(D_gau, L)
pltl.plotHist(lda.transform(D_gau), L, title= 'gaussianized and LDA-reduced features', show= False)

for m_pca in [5, 4]:
# histogram visualization with PCA-reduced features and LDA-reduced features
    D_pca = pca.transform(D_BASE, m_pca)
    lda = pre.LDA(D_pca, L)
    pltl.plotHist(lda.transform(D_pca), L, title= f'{m_pca}PCA_LDA-reduced features', show= False)

    ## histogram visualization with gaussianized, PCA-reduced features and LDA-reduced features
    D_gau_pca = pca.transform(D_gau, m_pca)
    lda = pre.LDA(D_gau_pca, L)
    pltl.plotHist(lda.transform(D_gau_pca), L, title= f'gaussianized and {m_pca}PCA_LDA-reduced features', show= False)

 
plt.show()
#



### sparse visualization with no preprocessing

pltl.plotSparse(D_BASE, L, attr_names= ['F1', 'F2', 'F3', 'F4', 'F5', 'F6'], title= 'no preprocessing', show= False)

## sparse visualization with gaussianized features

pltl.plotSparse(D_gau, L, attr_names= ['F1', 'F2', 'F3', 'F4', 'F5', 'F6'], title= 'gaussianized features', show= False)

## sparse visualization with PCA-reduced features

D_stand = pre.Standardizer(D_BASE).transform(D_BASE)
pca = pre.PCA(D_stand)
m_pca = 6
pltl.plotSparse(pca.transform(D_stand, m_pca), L, attr_names= ['F1', 'F2', 'F3', 'F4', 'F5', 'F6'], title= 'PCA' + str(m_pca) + '-reduced features', show= False)
plt.show()
### sparse visualization with gaussianized and PCA-reduced features

pca = pre.PCA(D_gau)
m_pca = 6
pltl.plotSparse(pca.transform(D_gau, m_pca), L, attr_names= ['F1', 'F2', 'F3', 'F4', 'F5', 'F6'], title= 'gaussianized and PCA-reduced features', show= False)

##

plt.show()