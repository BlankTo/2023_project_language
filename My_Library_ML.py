import numpy as np
import scipy.linalg as lng

def to_col(v): return v.reshape(v.size, 1)
def to_row(v): return v.reshape(1, v.size)

def load_iris():
    lab_to_num = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    D, L = [], []
    with open('iris.csv', 'r') as file:
        for line in file:
            split = line.rstrip().split(',')
            L.append(lab_to_num[split[-1]])
            D.append(to_col(np.array([float(x) for x in split[0:4]])))
    return np.hstack(D), np.array(L, dtype=np.int32)

def recenter(D):
    mu = to_col(D.mean(axis= 1))
    return D - mu, mu

def get_PCA_matrix(D, m):
    cov = (D@D.T) / D.shape[1]
    eig_val, eig_vect = np.linalg.eigh(cov)
    return np.copy(eig_vect[:, ::-1][:, 0:m])

def get_LDA_matrix(D, L, N_class, m, whitening= True, basis= False):
    M, N = D.shape
    mu = np.array(D.mean(axis= 1)).reshape(M, 1)
    Sb = np.zeros((M, M), dtype= np.float32)
    Sw = np.zeros((M, M), dtype= np.float32)
    for c in range(N_class):
        Dc = D[:, L==c]
        mc = Dc.mean(axis= 1).reshape(M, 1)
        mcmu = mc - mu
        Sb += Dc.shape[1] * mcmu @ mcmu.T
        mid = Dc - mc
        Sw += mid @ mid.T
    Sb = Sb / N
    Sw = Sw / N

    if not whitening:
        try:
            s, U = lng.eigh(Sb, Sw)
            W = U[:, ::-1][:, 0:m]
        except:
            print('whiten anyway')
            whitening = True

    if whitening:
        U, s, _ = np.linalg.svd(Sw)
        P1 = U @ np.diag(1.0/(s**0.5)) @ U.T
        Sbt = P1 @ Sb @ P1.T
        eig_val, eig_vect = lng.eigh(Sbt)
        P2 = eig_vect[:, ::-1][:, 0:m]
        W = -(P1.T @ P2)

    if not basis: return np.copy(W)
    else:
        UW, _, _ = np.linalg.svd(W)
        return np.copy(UW[:, 0:m])

def shuffle_and_divide(D, L, fraction_in_train):
    N = D.shape[1]
    permutation_index = np.random.permutation(N)
    D = D[:, permutation_index]
    L = L[permutation_index]
    division_index = int(N * fraction_in_train)
    return D[:, :division_index], L[:division_index], D[:, division_index:], L[division_index:]


def euclidean_classifier(D_train, L_train, D_test, n_class):
    assigned_L = []
    class_means = np.array([D_train[:, L_train==i].mean(axis= 1) for i in range(n_class)])
    for xt in D_test.T:
        distances = np.array([(lng.norm(xt-c_mean))**2 for c_mean in class_means])
        assigned_L.append(distances.argsort()[0])
    return np.array(assigned_L)