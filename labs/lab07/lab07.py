import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

import ML_lib.utils as utils
import ML_lib.log_reg as log_reg
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

## binary classification


for lam in [1e-6, 1e-3, 0.1, 1.]:
    print('--------------')
    print(lam)

    predictions = log_reg.LinearRegressionClassifier(DTR, LTR, lam).predict(DTE)

    correct = (predictions==LTE).sum()
    accuracy = correct / LTE.size
    error = 1 - accuracy
    print('error: ' + str(error*100) + '%')


## just a try on cross_validation
    
print('\n::::::::::::::::::::::::::::::::::')
print('::::::::::::::::::::::::::::::::::')
print('Cross validation base')
print('::::::::::::::::::::::::::::::::::')
print('::::::::::::::::::::::::::::::::::\n')

for lam in [1e-6, 1e-3, 0.1, 1.]:
    print(f'\nlambda: {lam}')
    LTE_cross, predictions, scores = utils.cross_validation_base(D, L, log_reg.LinearRegressionClassifier, 10, None, None, 0, [lam], True, False)
    utils.get_metrics(scores, LTE_cross, 0.5, 1, 1, print_err= True)
    utils.get_metrics(scores, LTE_cross, 0.1, 1, 1, print_err= True)
    
print('\n::::::::::::::::::::::::::::::::::')
print('::::::::::::::::::::::::::::::::::')
print('Cross validation v2')
print('::::::::::::::::::::::::::::::::::')
print('::::::::::::::::::::::::::::::::::\n')

utils.cross_validation(D, L, 10, log_reg.LinearRegressionClassifier, 
    model_params= [
    [1e-6],
    [1e-3],
    [0.1],
    [1.],
    ],
    effective=
    [0.25,
     0.5,
     0.75,
     ],
    print_err= True,
    prepro= [
    [(pre.NoTransform, [])],
    [(pre.Standardizer, [])],
    ])