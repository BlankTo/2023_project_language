import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from math import ceil

from ML_lib.utils import get_corrMatrix, getMinNormDCF, inter_BayesErrorPlot

def plot_corr(cm, axp= False, feature_names = [], fontsize= 14, title= None, xrotation= 0, yrotation= 0):

    if not axp: fig, ax = plt.subplots(1, 1)
    else: 
        ax = axp
        ax.set_facecolor('red')
    sns.heatmap(
                np.absolute(cm),
                ax= ax,
                annot= cm,
                fmt= '',
                cmap= 'Blues',
                vmin= 0,
                vmax= 1,
                cbar= True,

                linewidths= 1, linecolor='black',
                annot_kws= {"fontsize": fontsize},
                xticklabels= feature_names,
                yticklabels= feature_names
                )
    if feature_names:
        ax.set_yticklabels(ax.get_yticklabels(), size=fontsize, rotation=yrotation)
        ax.set_xticklabels(ax.get_xticklabels(), size=fontsize, rotation=xrotation)
        ax.tick_params(axis= 'both', which= 'major', labelsize= 10, left= False, labelbottom= False, bottom= False, labeltop= True, top= False)
    if title:
        title = f"{title}"
        ax.set_title(title)
    if not axp: fig.waitforbuttonpress(-1)


def plot_corr_D(D, axp= False, feature_names = [], fontsize= 14, title= None, xrotation= 0, yrotation= 0):
      plot_corr(get_corrMatrix(D), axp, feature_names, fontsize, title, xrotation, yrotation)


def plotHist(D, L, axp= False, label_dict= False, attr_names= False, title= '', figsize= (9, 6), saveImage= False, show= True):
    M = D.shape[0]
    K = len(set(L))
    xplot = ceil(np.sqrt(M))
    yplot = ceil(M/xplot)
    if not axp: fig, axs = plt.subplots(yplot, xplot, figsize= figsize)
    else: fig, axs = axp.subplots(yplot, xplot)
    if title: fig.suptitle(str(title))
    axs = np.array(axs).ravel()
    if not attr_names: attr_names = ['F ' + str(m) for m in range(M)]
    if not label_dict: label_dict = {li: 'class ' + str(li) for li in range(K)}
    for m in range(M):
        axs[m].title.set_text(attr_names[m])
        for c in range(K):
            if K==2:
                if c==0:
                    axs[m].hist(D[m, L==c], density=True, ls='dashed', alpha = 0.5, lw=3, color= 'r')
                else: axs[m].hist(D[m, L==c], density=True, ls='dotted', alpha = 0.5, lw=3, color= 'g')
            else: axs[m].hist(D[m, L==c], density=True, alpha = 0.5)
    fig.tight_layout(pad=1.0)
    fig.subplots_adjust(top= 0.85, bottom= 0.05, right= 0.95, left= 0.05)
    fig.legend(label_dict.values())
    axs = axs.reshape(yplot, xplot)
    if saveImage: plt.savefig(title)
    if (not axp) and show: plt.show()


def plotSparse(D, L, axp= False, label_dict= False, attr_names= False, title= '', figsize= (10, 7), saveImage= False, show= True):
    M = D.shape[0]
    if M < 2:
        print('cannot plot M < 3')
        return
    K = len(set(L))
    nplots = int((M**2 - M) / 2)
    xplot = ceil(np.sqrt(nplots))
    yplot = ceil(nplots/xplot)
    if not axp: fig, axs = plt.subplots(yplot, xplot, figsize= figsize)
    else: fig, axs = axp.subplots(yplot, xplot)
    if title: fig.suptitle(str(title))
    axs = np.array(axs).ravel()
    if not attr_names: attr_names = ['attribute ' + str(m) for m in range(M)]
    if not label_dict: label_dict = {li: 'class ' + str(li) for li in range(K)}
    m = 0
    for m1 in range(M):
        for m2 in range(M)[m1+1:]:
            axs[m].set_xlabel(attr_names[m1])
            axs[m].xaxis.set_label_position('top')
            axs[m].xaxis.tick_top()
            axs[m].set_ylabel(attr_names[m2])
            axs[m].yaxis.set_label_position('left')
            for c in range(K):
                if K == 2:
                    if c==0:
                        axs[m].scatter(D[m1, L==c], D[m2, L==c], color= 'r', alpha= 0.5, s= 10)
                    else:
                        if c==1:
                            axs[m].scatter(D[m1, L==c], D[m2, L==c], color= 'g', alpha= 0.5, s= 10)
                else: axs[m].scatter(D[m1, L==c], D[m2, L==c], alpha= 0.4, s= 10)
            m += 1
    fig.tight_layout(pad=2.0)
    fig.subplots_adjust(top= 0.80, bottom= 0.05, right= 0.95, left= 0.1)
    fig.legend(label_dict.values())
    axs = axs.reshape(yplot, xplot)
    if saveImage: plt.savefig(title)
    if (not axp) and show: plt.show()


def ROCcurve(llr, labels, p1, Cfp, Cfn):
    _, PRs = getMinNormDCF(llr, labels, p1, Cfp, Cfn, retPR= True)

    plt.figure()
    plt.plot(PRs[0], PRs[1], label= 'p1: ' + str(p1) + ' Cfp: ' + str(Cfp) + ' Cfn: ' + str(Cfn))
    plt.xticks(np.arange(0., 1.1, 0.1))
    plt.yticks(np.arange(0., 1.1, 0.1))
    plt.grid()
    plt.show()


def BayesErrorPlot(llr, labels, effPriorLogOdds, legend_text= '', stack= False):
    effPriors = 1 / (1 + np.exp(-effPriorLogOdds))
    DCFs_eff = []
    minDCFs_eff = []

    for eff in effPriors:
        DCF_eff, minDCF_eff = inter_BayesErrorPlot(llr, labels, eff)
        DCFs_eff.append(DCF_eff)
        minDCFs_eff.append(minDCF_eff)

    if not stack:
        plt.figure()
        plt.plot(effPriorLogOdds, DCFs_eff, label='DCF  ' + legend_text)
        plt.plot(effPriorLogOdds, minDCFs_eff, label='min DCF  ' + legend_text)
        plt.ylim([0, 1.1])
        plt.xlim([-3, 3])
        plt.grid()
        plt.legend()
        plt.show()
    else:
        plt.plot(effPriorLogOdds, DCFs_eff, label='DCF  ' + legend_text)
        plt.plot(effPriorLogOdds, minDCFs_eff, label='min DCF  ' + legend_text)
