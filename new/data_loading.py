import os
import sys
import numpy as np
from scipy.linalg import eigh as scipy_eigh

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

DTR = []
LTR = []
with open('data_raw\\Train_language.txt', 'r') as f:
    for line in f:
        line_split = line.split(',')
        x = [float(xj) for xj in line_split[:-1]]
        DTR.append(line_split[:-1])
        LTR.append(int(line_split[-1]))
DTR = (np.array(DTR, dtype= np.float64)).T
LTR = np.array(LTR, dtype= int)
print(DTR.shape)
print(LTR.shape)
np.save('data_npy\\DTR_language', DTR)
np.save('data_npy\\LTR_language', LTR)

#DTE = []
#LTE = []
#with open('data_raw\\Test_language.txt', 'r') as f:
#    for line in f:
#        line_split = line.split(',')
#        x = [float(xj) for xj in line_split[:-1]]
#        DTE.append(line_split[:-1])
#        LTE.append(int(line_split[-1]))
#DTE = (np.array(DTE, dtype= np.float64)).T
#LTE = np.array(LTE, dtype= int)
#print(DTE.shape)
#print(LTE.shape)
#np.save('data_npy\\DTE_language', DTE)
#np.save('data_npy\\LTE_language', LTE)