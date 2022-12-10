import numpy as np 
from numpy import linalg as LA
from scipy.linalg import norm
import matplotlib.pyplot as plt
import time
from numpy import *

def arnoldi_iteration(A, b, n: int, eps = 1e-12):
    
    """
    Computes a basis of the (n + 1)-Krylov subspace of A: the space
    spanned by {b, Ab, ..., A^n b}.

    Arguments
      A: m x m array
      b: initial vector (length m)
      n: dimension of Krylov subspace, must be >= 1
    
    Returns
      Q: m x (n + 1) array, the columns are an orthonormal basis of the
        Krylov subspace.
      h: (n + 1) x n array, A on basis Q. It is upper Hessenberg.  
    """
    
    print("This is A")
    print()
    print(A)
    
    h = np.zeros((n+1,n))
    Q = np.zeros((A.shape[0],n+1))
    q = b/np.linalg.norm(b,2)  # Normalize the input vector
    Q[:,0] =  q.transpose()  # Use it as the first Krylov vector
    print(Q)
    for k in range(1,n+1):
        print("------------------------------------------------")
        print("k",k)
        v = np.dot(A,Q[:,k-1])  # Generate a new candidate vector
        for j in range(k):  # Subtract the projections on previous vectors
            h[j,k-1] = np.dot(Q[:,j].T, v)
            print("h[j,k-1]: ", h[j,k-1])
            v = v - h[j,k-1] * Q[:,j]
        h[k,k-1] = np.linalg.norm(v,2)
        if h[k,k-1] > eps:  # Add the produced vector to the list, unless
            Q[:,k] = v/h[k,k-1]
        else:  # If that happens, stop iterating.
            return Q, h
    return Q, h


def lanczos(A, b, n:int):
    """
    Lanczos algorithm (Krylov approximation of a matrix)
        input: 
            A: matrix to approximate
            v0: initial vector (should be in matrix form) 
            k: number of Krylov steps 
        output: 
            V: matrix (large, N*k) containing the orthogonal vectors
            H: matrix (small, k*k) containing the Krylov approximation of A
    """
    print('LANCZOS METHOD !')
    V = np.mat(b.copy() / norm(b) )
    alpha =  np.zeros(n)  
    beta =  np.zeros(n+1)  
    for m in range(n):
        vt = np.dot(A , V[ :, m])
        if m > 0: 
            vt -= beta[m] * V[:, m-1]
        alpha[m] = (V[:, m].H * vt)[0, 0]
        vt -= alpha[m] * V[:, m]
        beta[m+1] = norm(vt)
        V = np.hstack((V, vt.copy() / beta[m+1]))
    rbeta = beta[1:-1]    
    H = np.diag(alpha) + np.diag(rbeta, 1) + np.diag(rbeta, -1)
    return V, H



def main():
    A = np.array(
        [
        [1,1,1,0,1,0,2,1,3,1],
        [1,4,1,4,1,3,2,1,3,1],
        [2,1,1,2,1,0,5,1,3,1],
        [1,1,1,4,1,2,2,1,3,4],
        [1,1,2,3,1,2,2,1,4,1],
        [1,1,3,4,1,2,3,1,3,4],
        [1,1,1,3,1,2,2,1,3,2],
        [1,1,1,4,1,5,2,4,3,3],
        [1,4,1,4,1,4,2,1,3,4],
        [3,1,1,3,1,3,2,1,3,3]
        ]
    )
    
    b = np.array(
                [ 
                [1],
                [2],
                [1],
                [2],
                [1],
                [2],
                [1],
                [2],
                [1],
                [2]
                ]
            ) 
    
    Q, h = arnoldi_iteration(A, b, 10, eps = 1e-12)
    
    print()
    
    print("This is Q")
    # Matrix Q represent ortonormal bases of krylov spaces 
    print(Q)
    print()
    
    print("This is H")
    print()
    print(h)
    
    u,v = LA.eigh(A)
    
    
    print("This is u")
    # The eigenvalues in ascending order, each repeated according to its multiplicity.
    # eigh = u.tolist()
    # eigh.reverse()
    # l_eigh = eigh
    # eight_find = list(h[0])

    # print(eight_find)
    # print(l_eigh)
    
    # print()
    
    
    # plt.plot(l_eigh)
    # plt.plot(eight_find)
    # plt.show()
    
    # plt.plot(h)
    # plt.show()
    
    k = 10
    Preps = 1000
    Nmin = 10
    lNmin = log(Nmin)/log(10)
    Nmax = 2000
    lNmax = log(Nmax)/log(10)
    Nvals = array(logspace(lNmin, lNmax), dtype=int)

    type_to_name = {float32:'float32', float64:'float64', complex64:'complex64', complex128:'complex128'}
 
    for T in [float32, float64, complex64, complex128]:
        print("current type:", type_to_name[T])
        Tc = []
        Tp = []
        for n in Nvals:
            ctime = inf
            ptime = inf
            for r in range(Preps):# time python 
                t = time.time()
                Vloc, Hloc = arnoldi_iteration(A, b ,10)
                ptime = min(ptime, time.time()-t)
            Tc.append(ctime)
            Tp.append(ptime)

        plt.loglog(Nvals, Tp, '-o', label="Python")
        plt.legend(loc='best')
        plt.xlabel('N', fontsize=16)
        plt.ylabel('Time', fontsize=16)
        plt.title('Timing of Arnoldi Algorithm with $k=%d$. Type: %s'%(k, type_to_name[T]))
        ax = plt.axis()
        plt.axis((Nmin, Nmax, ax[2], ax[3]))
        plt.draw()
        plt.show()
    
if __name__ == "__main__":
    main()

