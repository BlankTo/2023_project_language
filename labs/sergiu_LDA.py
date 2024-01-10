import numpy as np

######## LDA sergiu

def LDA_matrix(dataMat, labels, m):
    data_class=[]   #list of matrices, each matrix having samples of the same label as columns
    mean_class=[]

    mean=dataMat.mean(1)
    mean=mean.reshape(mean.size, 1)
    Sb=np.zeros((mean.shape[0], mean.shape[0]), dtype=np.float32)
    Sw=np.zeros((mean.shape[0], mean.shape[0]), dtype=np.float32)

    for i in range(0,3):
        data_labeled=dataMat[:, labels==i]
        
        mean_labeled=data_labeled.mean(1)
        mean_labeled=mean_labeled.reshape(mean_labeled.size, 1) #reshape the mean as a (2D) column vector

        nc=data_labeled.shape[1]    #nr of samples in the current class
        N=dataMat.shape[1]

        #compute between class covariance matrix Sb
        e=mean_labeled-mean
        term=np.dot(e, e.T)
        term=term*nc
        term=term/N
        Sb=Sb+term

        #compute within class covariance matrix Sw
        data_labeled_centered=data_labeled-mean_labeled     #recall: mean_labeled is already shaped as a column vector, so broadcasting occurs as wanted
        covariance_mat=np.dot(data_labeled_centered, data_labeled_centered.T)
        covariance_mat=covariance_mat/N
        Sw=Sw+covariance_mat

    s, U=np.scipy.linalg.eigh(Sb, Sw)
    W=U[:, ::-1][:, 0:m]

    return W