import numpy as np


def to_col(v): return v.reshape(v.size, 1)


def to_row(v): return v.reshape(1, v.size)


def vec(X): return np.vstack([X[:, i].reshape(X.shape[0], 1) for i in range(X.shape[1])])


def get_var(X): return ((X - X.mean())**2).sum() / (X.size-1)


def get_cov(X, Y): return (((X - X.mean())*(Y - Y.mean())).sum()) / (X.size-1)


def get_covMatrix(D):
       M = D.shape[0]
       covMat = np.zeros((M, M))
       for i in range(M):
              covMat[i, i] += get_var(D[i, :])
              for j in range(i+1, M):
                     covMat[i, j] += get_cov(D[i, :], D[j, :])
                     covMat[j, i] += get_cov(D[i, :], D[j, :])
       return covMat


def get_corr(X, Y): return get_cov(X, Y) / np.sqrt(get_var(X)*get_var(Y))


def get_corrMatrix(D):
       M = D.shape[0]
       corrMat = get_covMatrix(D)
       for i in range(M):
              for j in range(i+1, M):
                     corrMat[i, j] /= np.sqrt(corrMat[i, i]*corrMat[j, j])
                     corrMat[j, i] /= np.sqrt(corrMat[i, i]*corrMat[j, j])
       for i in range(M): corrMat[i, i] = 1.
       return corrMat


def shuffle_and_divide(D_in, L_in, fraction_in_train, seed= 666):
    N = D_in.shape[1]
    np.random.seed(seed)
    permutation_index = np.random.permutation(N)
    D = D_in[:, permutation_index]
    L = L_in[permutation_index]
    division_index = int(N * fraction_in_train)
    return D[:, :division_index], L[:division_index], D[:, division_index:], L[division_index:]