import numpy as np 
from numpy import linalg as LA
from scipy.linalg import norm
import matplotlib.pyplot as plt
import time
from numpy import *
import pandas as pd
from scipy.sparse import rand
import scipy.sparse as sparse


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

def power_method(A, b, iteration:100, eps = 1e-10):
    
    result = pd.DataFrame(columns=['eigenvalue', 'error']) # Create df result 
    eign_power = [] # List of eighenvalue find with power method
    error_power = []
    eigenvalue = 0 # Inizialize eig
    oldeigenvalue = 0 # Inizialize old eig
    
    for i in range(iteration):
        
        b = np.dot(A,b) # Multiply vector matrix 
        eigenvalue = np.linalg.norm(b) # Calcolate norm of eigenvalue
        b = b/eigenvalue  #Re normalize the vector
        error = relative_error(eigenvalue, oldeigenvalue) # Calcolate relative error between old an new eigh
        result.loc[i] = [eigenvalue, error] # Insert eigen and error to df 
        
        if error < eps: # Check accuracy 
            break
        
        oldeigenvalue = eigenvalue # Swap
    
    return result



def main():
    # A = np.array(
    #     [
    #     [1,1,1,0,1,0,2,1,3,1],
    #     [1,4,1,4,1,3,2,1,3,1],
    #     [2,1,1,2,1,0,5,1,3,1],
    #     [1,1,1,4,1,2,2,1,3,4],
    #     [1,1,2,3,1,2,2,1,4,1],
    #     [1,1,3,4,1,2,3,1,3,4],
    #     [1,1,1,3,1,2,2,1,3,2],
    #     [1,1,1,4,1,5,2,4,3,3],
    #     [1,4,1,4,1,4,2,1,3,4],
    #     [3,1,1,3,1,3,2,1,3,3]
    #     ]
    # )
    
    # b = np.array(
    #             [ 
    #             [1],
    #             [2],
    #             [1],
    #             [2],
    #             [1],
    #             [2],
    #             [1],
    #             [2],
    #             [1],
    #             [2]
    #             ]
    #         ) 
    
    large = sparse.random(10, 10, density=0.30, data_rvs=np.ones)
    large_matrix = large.toarray()
    
    vector = sparse.random(10, 1, density=0.30, data_rvs=np.ones)
    large_vector = vector.toarray()
    
    Q, h = arnoldi_iteration(large_matrix, large_vector, 10, eps = 1e-12)
    
    print()
    
    print("This is Q")
    # Matrix Q represent ortonormal bases of krylov spaces 
    print(Q)
    print()
    plt.title('Ortonormal bases of krylov spaces')
    plt.imshow(Q)
    plt.colorbar()
    plt.show()
    
    print("--------------------")
    
    print("This is H")
    print()
    print(h)
    plt.title('Upper Hessemberg Matrix')
    plt.imshow(h)
    plt.colorbar()
    plt.show()
    
   
    u = linalg.eigvals(large_matrix)

    
    
    print("--------------------")
    
    print("This is u")
    # The eigenvalues in ascending order, each repeated according to its multiplicity.
    # U is the eigh find by method 
    eigh = list(u)

    # Eight find is the eigh of the arnoldi method 
    eight_find = list(h[0])
    
    s1 = sort(eigh)
    s2 = sort(eight_find).tolist()
    
    start1 = s1[0:5]
    start2 = s2[0:5]
    
    highest1 = s1[-1]
    highest2 = s2[-1]
    
    print(highest1)
    print(highest2)
    
    print(start1)
    print(start2)
    
    end1 = s1[:5]
    end2 = s2[:5]
    
    res = power_method(large_matrix, large_vector,iteration=10, eps = 1e-12)
    
    print(res['eigenvalue'])
    
    """
        Plot graph difference
    """
    
    plt.title('Eigenvalue difference between find by arnoldi and original')
    plt.plot(start1, '-o', label="eighenvalue find by arnoldi")
    #plt.plot(res['eigenvalue'], '-o', label="eighenvalue power method")
    plt.plot(start2, '-o', label="eighenvalue original")
    plt.legend(loc="upper right")
    plt.show()
    
    plt.title('Eigenvalue difference between find by arnoldi and original')
    plt.plot(end1, '-o', label="eighenvalue find by arnoldi")
    #plt.plot(res['eigenvalue'], '-o', label="eighenvalue power method")
    plt.plot(end2, '-o', label="eighenvalue original")
    plt.legend(loc="upper right")
    plt.show()
    
    
    """
        Generate 1 matrix large sparse 
        
        Generate 1 vector large sparse
        
        for test convergenza power method
        
        Il metodo delle potenze sembra essere applicabile a matrici di grnandi dimensioni 
    """
    
    # large = sparse.random(1000, 1000, density=0.20, data_rvs=np.ones)
    # large_matrix = large.toarray()
    
    # vector = sparse.random(1000, 1, density=0.20, data_rvs=np.ones)
    # large_vector = vector.toarray()
              
    # Recall power method 
    res = power_method(large_matrix, large_vector,iteration=100, eps = 1e-12)

    # Print verify
    print(res)
    
    """
        Plot convergenza matrici a grandi dimensioni
    """
    
    # plt.title('Convergence of eigenvalue by methof of power')
    # plt.plot(res['eigenvalue'])
    # plt.show()
        
    plt.title('Eigenvalue difference between find by arnoldi and original')
    plt.plot(highest2, '-o', label="eighenvalue find by arnoldi")
    plt.plot(res['eigenvalue'], '-o', label="eighenvalue power method")
    plt.plot(highest1, '-o', label="eighenvalue original")
    plt.legend(loc="upper right")
    plt.show()

        
 

    
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
    
    
    """
    #TODO: Verificare correttezza grafico 
    
    # eigenvalues = eigvals(A)

    # fig = plt.figure(figsize=(8, 8))
    # ax = fig.add_subplot(111)
    # ax.plot(np.real(eigenvalues), np.imag(eigenvalues), 'rx', markersize=1)
    
    # plt.show()
    
    # shifts = [0, 1, 2, 5, 10]
    # colors = ['r', 'k', 'y', 'g', 'm']

    # b = np.random.randn(10)

    # fig = plt.figure(figsize=(8, 8))
    # ax = fig.add_subplot(111)

    # for index, shift in enumerate(shifts):
    #     Ashift = A + shift * np.eye(10)
    #     residuals = []
    #     callback = lambda res: residuals.append(res)
    #     x, _ = gmres(Ashift, b, restart=None, callback=callback, callback_type='pr_norm')
    #     if len(residuals) > 50:
    #         residuals = residuals[:50] # Only plot the first 50 residuals
    #     ax.semilogy(residuals, colors[index] + '-x', markersize=2)
        
    # fig.legend(loc='lower center', fancybox=True, shadow=True, ncol=len(shifts), labels=[str(shift) for shift in shifts])
    
    # plt.show()
    
    """