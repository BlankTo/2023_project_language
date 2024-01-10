import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

import ML_lib.utils as utils
import ML_lib.log_reg as log_reg


D, L, label_dict = utils.csv_to_npy('data_raw/iris.csv')
D = D[:, L!=0]
L = L[L!=0]
L[L==2] = 0
DTR, LTR, DTE, LTE = utils.shuffle_and_divide(D, L, 2.0/3.0, 0)
print(DTR.shape)
print(LTR.shape)
print(DTE.shape)
print(LTE.shape)


for lam in [1e-6, 1e-3, 0.1, 1.]:
    print('--------------')
    print(lam)

    predictions = log_reg.LinearRegressionClassifier(DTR, LTR, lam).predict(DTE)

    correct = (predictions==LTE).sum()
    accuracy = correct / LTE.size
    error = 1 - accuracy
    print('error: ' + str(error*100) + '%')
