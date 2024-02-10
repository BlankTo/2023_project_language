import os
import sys
import numpy as np
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import ML_lib.gmm as gmm
import ML_lib.utils as utils
import ML_lib.preprocessing as pre

D_BASE = np.load('data_npy\\DTR_language.npy')
L = np.load('data_npy\\LTR_language.npy')

N_CLASS = 2
M_BASE, NTR = D_BASE.shape

model_params = [[al, nG, bound, var, True] for al in [0.1] for nG in [[64, 64]] for bound in [1e-6] for var in ['']] #['', 'diag', 'tied', 'diag-tied']]

#utils.cross_validation(D_BASE, L, 10, gmm.GMMClassifier, model_params, progress= True, prepro= [
#    [(pre.NoTransform, [])],
#    [(pre.Gaussianizer, [])],
#    [(pre.Standardizer, []), (pre.PCA, [5])],
#    ])

DTR_in = D_BASE
LTR_in = L
n_split = 10
prepro = [
#    [(pre.NoTransform, [])],
    [(pre.Gaussianizer, [])],
#    [(pre.Standardizer, []), (pre.PCA, [5])],
    ]
progress = True
save = False
filename = 'results\\cross_val_gmm.txt'
print_act = False

NTR = DTR_in.shape[1]

np.random.seed(0)
idx = np.random.permutation(NTR)
DTR_perm = DTR_in[:, idx]
LTR_perm = LTR_in[idx]

step = NTR / n_split
considered = []
assigned_all = []
scores_all = []

n_ite = len(prepro) * len(model_params)

if progress: print('- starting -\n')
tot_time = 0
for i_split in range(n_split):
    if progress: start_time = time.time()
    if progress: print(f'split: {i_split+1} / {n_split}\n')
    DTR_base = []
    LTR = []
    DTE_base = []
    LTE = []
    for i in range(n_split):
        if not i_split==i:
            DTR_base.append(DTR_perm[:, int(step * i):int(step * (i+1))])
            LTR.append(LTR_perm[int(step * i):int(step * (i+1))])
        else:
            DTE_base = DTR_perm[:, int(step * i):int(step * (i+1))]
            LTE = LTR_perm[int(step * i):int(step * (i+1))]
    DTR_base = np.hstack(DTR_base)
    LTR = np.hstack(LTR)
    DTE_base = np.array(DTE_base)
    considered.append(LTE)

    assigned_i = []
    scores_i = []
    i_ite = 0

    for prep_ite in prepro:

        DTR_pre = DTR_base
        DTE_pre = DTE_base

        for prep in prep_ite:

            prep_t = prep[0](DTR_pre, LTR)
            DTR_pre = prep_t.transform(DTR_pre, *prep[1])
            DTE_pre = prep_t.transform(DTE_pre, *prep[1])

        for model_param in model_params:
            if progress:
                print(f'\x1b[1A]\x1b[2K -- ite: {i_ite+1} / {n_ite}')
                i_ite += 1
            model_init = gmm.GMMClassifier(DTR_pre, LTR, *model_param)

            scores_ii = model_init.getScores_All(DTE_pre)
            for sco_i in scores_ii: scores_i.append(sco_i)
            assigned_ii = (scores_ii > 0).astype(int)
            for ass_i in assigned_ii: assigned_i.append(ass_i)

    assigned_all.append(assigned_i)
    scores_all.append(scores_i)
    if progress:
        tot_time += time.time() - start_time
        print('\x1b[1A]\x1b[2Ksplit completed in %.2fs - eta for completion is %.2fs' % (time.time() - start_time, (tot_time * (n_split - i_split - 1) / (i_split + 1))))
if progress: print('all splits completed in %.2fs' % tot_time)

considered = np.hstack(considered)
n_cons = considered.size

if save:
    try: f = open(filename, 'a')
    except: f = open(filename, 'w')

place = 0
for prep_ite in prepro:

    i_mp = 0
    for model_param in model_params:

        for nnG_0 in range(int(np.log2(model_param[1][0]) + 1)):
            for nnG_1 in range(int(np.log2(model_param[1][1]) + 1)):

                assigned = np.hstack([assigned_all[i][place] for i in range(n_split)])
                scores = np.hstack([scores_all[i][place] for i in range(n_split)])
                place += 1

                error_rate = (n_cons - (assigned == considered).sum()) * 100 / n_cons

                FNR = (assigned[considered==1]==0).sum() / (considered==1).sum()
                FPR = (assigned[considered==0]==1).sum() / (considered==0).sum()

                DCF05 = FNR + FPR
                DCF01 = (0.1 * FNR + 0.9 * FPR) / 0.1

                ##
                mini05 = 100
                mini01 = 100
                sort_idx = np.argsort(scores)
                scores_s = scores[sort_idx]
                LTE_s = considered[sort_idx]

                for thresh in scores_s:
                    assigned_m = scores_s > thresh

                    FNR = (assigned_m[LTE_s==1]==0).sum() / (LTE_s==1).sum()
                    FPR = (assigned_m[LTE_s==0]==1).sum() / (LTE_s==0).sum()

                    DCF05_in = (FNR + FPR)
                    if DCF05_in < mini05: mini05 = DCF05_in

                    DCF01_in = (0.1 * FNR + 0.9 * FPR) / 0.1
                    if DCF01_in < mini01: mini01 = DCF01_in
                ##

                spec = ''
                for prep in prep_ite: spec += f'{prep[0].getName()} with {prep[1]}  '
                spec += ' -- params: '
                spec += '['
                spec += f'alpha: {model_param[0]}, '
                spec += f'nG_0: {2 ** nnG_0}, '
                spec += f'nG_1: {2 ** nnG_1}, '
                spec += f'bound: {model_param[2]}, '
                spec += f'variant: {model_param[3]}'
                spec += ']'
                if print_act:
                    print_string = str(spec.ljust(90)  + ''
                                                    + '->   minMetric: ' + str('%.3f' % ((mini05 + mini01) / 2))
                                                    + ' \tminDCF(05, 01): (' + str('%.3f' % mini05) + ', ' + str('%.3f' % mini01) + ')'
                                                    + ' \t   metric: ' + str('%.3f' % ((DCF05 + DCF01) / 2))
                                                    + ' \tminDCF(05, 01): (' + str('%.3f' % DCF05) + ', ' + str('%.3f' % DCF01) + ')'
                                    )
                else:
                    print_string = str(spec.ljust(90)  + ''
                                                    + '->   minMetric: ' + str('%.3f' % ((mini05 + mini01) / 2))
                                                    + ' \tminDCF(05, 01): (' + str('%.3f' % mini05) + ', ' + str('%.3f' % mini01) + ')'
                                    )
                #if model_params_print is not None: 
                #    print(model_params_print[i_mp])
                #    i_mp += 1
                print(print_string)
                if save: f.write(print_string + '\n')

if save: f.close()