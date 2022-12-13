import numpy as np 
from numpy import linalg as LA
from scipy.linalg import norm
import matplotlib.pyplot as plt
import time
from numpy import *
import pandas as pd
from scipy.sparse import rand
import scipy.sparse as sparse
import scipy.stats as stats

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

"""
    Function to calc relative error 
"""
def relative_error(xnew, xold):
    return abs((xnew-xold)/xnew)*100

"""
    Implementation of power method 
"""
def power_method(A, b, iteration, eps = 1e-12):
    
    result = pd.DataFrame(columns=['eigenvalue', 'error']) # Create df result 
    eign_power = [] # List of eighenvalue find with power method
    eigenvalue = 0 # Inizialize eig
    oldeigenvalue = 0 # Inizialize old eig
    
    for i in range(iteration):
        
        b = np.dot(A,b) # Multiply vector matrix 
        eigenvalue = np.linalg.norm(b) # Calcolate norm of eigenvalue
        b = b/eigenvalue 
        error = relative_error(eigenvalue, oldeigenvalue) # Calcolate relative error between old an new eigh
        result.loc[i] = [eigenvalue, error] # Insert eigen and error to df 
        eign_power.append(eigenvalue) # Insert eigen find in a list 
        
        if error < eps: # Check accuracy 
            break
        
        oldeigenvalue = eigenvalue # Swap
    
    print(eign_power)
    return(eign_power)

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
    # plt.imshow(Q)
    # plt.colorbar()
    # plt.show()
    
    print("--------------------")
    
    print("This is H")
    print()
    print(h)
    # plt.imshow(h)
    # plt.colorbar()
    # plt.show()
    
    u,v = LA.eigh(A)

    
    
    print("--------------------")
    
    
    print("This is u")
    # The eigenvalues in ascending order, each repeated according to its multiplicity.
    # U is the eigh find by method 
    eigh = u.tolist()
    eigh.reverse()

    # Eight find is the eigh of the arnoldi method 
    eight_find = list(h[0])

    # Print find eigh
    print(eight_find)
    
    # Print method eigh
    print(eigh)
    
    #print(eig_h)
    
    
    """
        Generate 1 matrix large sparse 
        
        Generate 1 vector large sparse
        
        for test convergenza power method ( OK ? )
        
        Il metodo delle potenze sembra essere applicabile a matrici di grnandi dimensioni 
    """
    large = sparse.random(10000, 10000, density=0.25, data_rvs=np.ones)
    large_matrix = large.toarray()
    
    vector = sparse.random(10000, 1, density=0.25, data_rvs=np.ones)
    large_vector = vector.toarray()
              
    # Recall power method 
    prova = power_method(large_matrix, large_vector,iteration=1000, eps = 1e-10)

    # Print verify
    print(prova)
   
    # # plt.plot(eigh)
    # # plt.plot(eight_find)
    
    """
        Plot convergenza matrici a grandi dimensioni
    """
    plt.plot(prova)
    plt.show()
    
    
    """
        Lancozos
    """
    
    # V, h = lanczos(A,b,10)
    
    # print("--------------------")
    # print("This is V")
    # print(V)
    # plt.imshow(V)
    # plt.colorbar()
    # plt.show()
    
    # print("--------------------")
    # print("This is h")
    # print(h)
    # plt.imshow(h)
    # plt.colorbar()
    # plt.show()
    
    
    # print("--------------------")
    # eight_find = list(h[0])
    # print(eigh)
    # print(eight_find)
    
  
        
 

    
if __name__ == "__main__":
    main()
    

    
    
    """
    #TODO Tempo di esecuzione arnoldi iter da correggere?
    
    
    
    k = 10
    Preps = 100
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
    """