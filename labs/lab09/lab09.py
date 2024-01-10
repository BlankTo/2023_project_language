import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

import ML_lib.svm as svm
import ML_lib.utils as utils
import ML_lib.preprocessing as pre


D, L, label_dict = utils.csv_to_npy('data_raw/iris.csv')
D = D[:, L!=0]
L = L[L!=0]
L[L==2] = 0
DTR, LTR, DTE, LTE = utils.shuffle_and_divide(D, L, 2.0/3.0, 0)
print(DTR.shape)
print(LTR.shape)
print(DTE.shape)
print(LTE.shape)
print('______________________________________')

#for K in [1, 10]:
#    for C in [0.1, 1., 10.]:
#        print('K: %.1f\tC: %.1f\t-  errorRate: %.1f' % (K, C, svm.SVM(DTR, LTR, DTE, LTE, K, C)))
#
#print('------------------------------------------------')
#
C = 1.
#
#for c in [0., 1.]:
#    for K in [0., 1.]:
#        print('K %.1f\tC %.1f\tkernel poly d 2 c %d\t-  errorRate: %.1f' % (K, C, c, svm.SVM(DTR, LTR, DTE, LTE, K, C, svm.get_kernel_poly(c, 2, K**2))))
#
#print('------------------------------------------------')
#
#for K in [0.]:#[0., 1.]:
#    for lam in [1.]:#[1., 10.]:
#        print('K %.1f\tC %.1f\tkernel rbf lam %.1f\t-  errorRate: %.1f' % (K, C, lam, svm.SVM(DTR, LTR, DTE, LTE, K, C, svm.get_kernel_RBF(lam, K**2))))
#


### prova MNIST

L = np.load('labs/lab09/MNIST_target.npy')
D = np.load('labs/lab09/MNIST_data.npy')

DTR_base = D[:, :60000]
LTR_base = L[:60000]
DTE_base = D[:, 60000:]
LTE_base = L[60000:]

M, NTR = DTR_base.shape
NTE = DTE_base.shape[1]
n_class = 10
print_string = '----------------------------------------\n'
K = 1.
C = 0.1

print_string += str('SVM linear - K= ' + str(K) + ' - C= ' + str(C) + ' -> ')
print(print_string)
correct = 0
considered = 0

#ker = svm.get_kernel_poly(0., 2, K**2)
ker = svm.get_kernel_RBF(2., K**2)

for m1 in range(n_class):
    for m2 in range(m1+1, n_class):
        print('m1: %d/%d  -  m2: %d/%d' % (m1, n_class, m2, n_class))
        DTR = np.hstack([DTR_base[:, LTR_base==m1], DTR_base[:, LTR_base==m2]])
        LTR = np.hstack([LTR_base[LTR_base==m1], LTR_base[LTR_base==m2]])
        DTE = np.hstack([DTE_base[:, LTE_base==m1], DTE_base[:, LTE_base==m2]])
        LTE = np.hstack([LTE_base[LTE_base==m1], LTE_base[LTE_base==m2]])


        DTR = DTR[:, :3600]
        LTR = LTR[:3600]
        DTE = DTE[:, :630]
        LTE = LTE[:630]
        print(DTR.shape)
        print(DTE.shape)

        M, NTR = DTR.shape
        NTE = DTE.shape[1]
        considered += NTE
        for i in range(NTR):
            if LTR[i]==m1: LTR[i] = 0
            else: LTR[i] = 1
        for i in range(NTE):
            if LTE[i]==m1: LTE[i] = 0
            else: LTE[i] = 1

        new_assigned = svm.SVM(DTR, LTR, DTE, LTE, K, C, kernel_function= ker, maxiter= 1000, retAssigned= True)
        new_correct = (LTE == new_assigned).sum()
        print((NTE - new_correct) * 100 / NTE)
        correct += new_correct
print((considered - correct) * 100 / considered)