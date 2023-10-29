import numpy as np

def eigenface_core(T):
    # Calculating the mean image
    m = np.mean(T, axis=0) # Computing the average face image m = (1/P)*sum(Tj's)    (j = 1 : P)

    Train_Number = T.shape[0]

    # Calculating the deviation of each image from mean image
    A = []
    for i in range(Train_Number):
        temp = T[i,:] - m
        A.append(temp)
    A = np.array(A).T
    # Snapshot method of Eigenface method
    # We know from linear algebra theory that for a PxQ matrix, the maximum
    # number of non-zero eigenvalues that the matrix can have is min(P-1,Q-1).
    # Since the number of training images (P) is usually less than the number
    # of pixels (M*N), the most non-zero eigenvalues that can be found are equal
    # to P-1. So we can calculate eigenvalues of A'*A (a PxP matrix) instead of
    # A*A' (a M*NxM*N matrix). It is clear that the dimensions of A*A' is much
    # larger that A'*A. So the dimensionality will decrease.
    L = np.dot(A.T, A) # L is the surrogate of covariance matrix C=A*A'.
    D, V = np.linalg.eig(L) # Diagonal elements of D are the eigenvalues for both L=A'*A and C=A*A'.
    # idx = D.argsort() # Sorting the eigenvalues
    # D = D[idx]
    # V = V[:,idx]
    # So1rting and eliminating eigenvalues
    # All eigenvalues of matrix L are sorted and those who are less than a
    # specified threshold, are eliminated. So the number of non-zero
    # eigenvectors may be less than (P-1).
    L_eig_vec = []
    for i in range(len(V[1])):
        if D[i] > 1:
            L_eig_vec.append(V[:,i])
    L_eig_vec = np.array(L_eig_vec).T
    # Calculating the eigenvectors of covariance matrix 'C'
    # Eigenvectors of covariance matrix C (or so-called "Eigenfaces")
    # can be recovered from L's eiegnvectors.
    Eigenfaces = np.dot(A, L_eig_vec) # A: centered image vectors

    return m, A, Eigenfaces