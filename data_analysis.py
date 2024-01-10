import numpy as np
import matplotlib.pyplot as plp

import common as com

def PearsonAnalysis(D, name= '', value= 0.6):
       print('\nPearson linear correlation analysis (' + name + '):')
       corrMat = com.get_corrMatrix(D)
       for i in range(D.shape[0]):
              for j in range(i+1, D.shape[0]):
                     if np.abs(corrMat[i, j])>value: print(str((i, j)) + ' -> ' + str(np.abs(corrMat[i, j])))



def plotHist(D, L, M, K, show=True, title= '', saveImage= False):
    xplot = int(np.sqrt(M))
    yplot = int(M//xplot)
    if xplot*yplot < M: yplot += 1
    cx = 0
    cy = 0
    fig, axs = plp.subplots(yplot, xplot)
    if title: fig.suptitle(str(title))
    for m in range(M):
        axs[cy, cx].title.set_text('attribute %d' % m)
        for c in range(K):
            if K==2:
                if c==0:
                    axs[cy, cx].hist(D[m, L==c], density=True, ls='dashed', alpha = 0.5, lw=3, color= 'r')
                else: axs[cy, cx].hist(D[m, L==c], density=True, ls='dotted', alpha = 0.5, lw=3, color= 'g')
            else: axs[cy, cx].hist(D[m, L==c], density=True)
        cx += 1
        if cx>=xplot:
            cx = 0
            cy += 1
    if saveImage: plp.savefig(title)
    if show: plp.show()

def plotSparse(D, L, M, K, show=True, title= '', saveImage= False):
    nplots = (M**2 - M) / 2
    xplot = int(np.sqrt(nplots))
    yplot = int(nplots//xplot)
    if xplot*yplot < nplots: yplot += 1
    cx = 0
    cy = 0
    fig, axs = plp.subplots(yplot, xplot)
    if title: fig.suptitle(str(title))
    for m1 in range(M):
        for m2 in range(M)[m1+1:]:
            axs[cy, cx].axis('off')
            axs[cy, cx].title.set_text('%d vs %d' % (m1, m2))
            for c in range(K):
                if c==0:
                    axs[cy, cx].scatter(D[m1, L==c], D[m2, L==c], color= 'r', alpha= 0.6, s= 10)
                else:
                    if c==1:
                        axs[cy, cx].scatter(D[m1, L==c], D[m2, L==c], color= 'g', alpha= 0.2, s= 10)
                    else: axs[cy, cx].scatter(D[m1, L==c], D[m2, L==c], alpha= 0.2, s= 10)
            cx += 1
            if cx>=xplot:
                cx = 0
                cy += 1
    if saveImage: plp.savefig(title)
    if show: plp.show()