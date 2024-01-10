import numpy as np

DTR = []
LTR = []
with open('Train.txt', 'r') as f:
    for line in f:
        line_split = line.split(',')
        x = [float(xj) for xj in line_split[:-1]]
        DTR.append(line_split[:-1])
        LTR.append(int(line_split[-1]))
DTR = (np.array(DTR, dtype= np.float64)).T
LTR = np.array(LTR, dtype= int)
print(DTR.shape)
print(LTR.shape)
np.save('DTR_language', DTR)
np.save('LTR_language', LTR)

#DTE = []
#LTE = []
#with open('Test.txt', 'r') as f:
#    for line in f:
#        line_split = line.split(',')
#        x = [float(xj) for xj in line_split[:-1]]
#        DTE.append(line_split[:-1])
#        LTE.append(int(line_split[-1]))
#DTE = (np.array(DTE, dtype= np.float64)).T
#LTE = np.array(LTE, dtype= int)
#print(DTE.shape)
#print(LTE.shape)
#np.save('DTE_pulsar', DTE)
#np.save('LTE_pulsar', LTE)