import numpy as np
import scipy.linalg as lng
import numpy.linalg as np_lng

def euclidean_classifier(D_train, L_train, D_test, n_class, retScores= False):
    assigned_L = []
    if retScores and n_class == 2: scores_L = []
    class_means = np.array([D_train[:, L_train==c].mean(axis= 1) for c in range(n_class)])
    for xt in D_test.T:
        distances = np.array([(lng.norm(xt-c_mean))**2 for c_mean in class_means])
        assigned_L.append(distances.argsort()[0])
        if retScores and n_class == 2: scores_L.append(distances[0]/distances[1])
    if retScores and n_class == 2: return scores_L
    return np.array(assigned_L)


#def mahalanobis_dist(DTR):
#    mu = DTR.mean(axis=1).reshape(DTR.shape[0], 1)
#    C = com.get_covMatrix(DTR)
#    MD = np.zeros(DTR.shape[1])
#    DTR_mu = DTR - mu
#    for i in range(MD.size): MD[i] += np.sqrt(DTR_mu[:, i].T @ np_lng.inv(C) @ DTR_mu[:, i])
#    return MD

#def mahalanobis_classifier(DTR, LTR, DTE, K):
#    M, NTE = DTE.shape
#    MD = np.zeros((K, NTE))
#    for c in range(K):
#        DTR_c = DTR[:, LTR==c]
#        DTE_mu_c = DTE - DTR_c.mean(axis=1).reshape(M, 1)
#        for i in range(NTE): MD[c, i] += np.sqrt(DTE_mu_c[:, i].T @ np_lng.inv(com.get_covMatrix(DTR_c)) @ DTE_mu_c[:, i])
#    assigned = np.argmin(MD, axis=0)
#    return assigned

#def load_iris():
#    lab_to_num = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
#    D, L = [], []
#    with open('iris.csv', 'r') as file:
#        for line in file:
#            split = line.rstrip().split(',')
#            L.append(lab_to_num[split[-1]])
#            D.append(com.to_col(np.array([float(x) for x in split[0:4]])))
#    return np.hstack(D), np.array(L, dtype=np.int32)