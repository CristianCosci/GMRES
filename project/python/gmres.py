from typing import Tuple
from scipy import sparse
import numpy as np

from scipy.linalg import lstsq
from scipy.linalg import norm
from scipy import sparse
from scipy.sparse.linalg import gmres
from scipy.sparse.linalg import spsolve
from scipy.io import savemat


import matplotlib.pyplot as plt


def generate_data(dim:int, den:float, seed=911, randx0=False) -> Tuple[np.ndarray, np.array, np.array]:
    '''
    Generate initial data.

        returns:
            - matrix A
            - vector b
            - initial guess x0 (random or zeros)
    '''
    np.random.seed(seed)
    
    A = sparse.random(dim, dim, density=den , format="csr", dtype=np.float32)
    #A = A.transpose().dot(A) # ? wtf?
    b = np.random.random(dim)

    # savemat('test.mst', {"a": A, "b":b})

    if randx0:
        x0 = np.random.random(dim)
    else:
        x0 = np.zeros(dim)

    return A, b, x0


def gmres_scipy(A, b, x0, tollerance=1e-10, max_iter=10):
    '''
    Gmres scipy call.
    '''

    return gmres(A,b, x0=x0, tol=tollerance, maxiter=max_iter)


def customNorm(res):
    for i in range(1, len(res)):
        res[i] -= res[i-1]
    
    return res

def arnoldi_iteration_pd(A: np.ndarray, b: np.array, n: int, eps=1e-12, Q=None, H=None):

    if H is None:
        H = np.zeros((n+1,n))
    if Q is None:
        Q = np.zeros((A.shape[0],n+1))

    # Normalize the input vector
    # Q[:, 0] = b / norm(b, 2)   # Use it as the first Krylov vector
    for k in range(1, n+1):
        v = A.dot(Q[:, k-1])  # Generate a new candidate vector
        
        for j in range(k):  # Subtract the projections on previous vectors
            H[j,k-1] = Q[:,j].T.dot(v)  # TODO: Q @ v
            v = v - H[j,k-1] * Q[:,j]
        
        H[k, k-1] = norm(v, 2)
        
        if H[k, k-1] > eps:  # Add the produced vector to the list, unless
            Q[:,k] = v/H[k, k-1]
        else:  # If that happens, stop iterating.
            return Q, H

        # condizione dentro funzione, quindi -> break
    
    return Q, H


# def arnoldi_iteration(A, b, n: int):
#     """Computes a basis of the (n + 1)-Krylov subspace of A: the space
#     spanned by {b, Ab, ..., A^n b}.

#     Arguments
#       A: m × m array
#       b: initial vector (length m)
#       n: dimension of Krylov subspace, must be >= 1
    
#     Returns
#       Q: m x (n + 1) array, the columns are an orthonormal basis of the
#         Krylov subspace.
#       h: (n + 1) x n array, A on basis Q. It is upper Hessenberg.  
#     """
#     eps = 1e-12
#     h = np.zeros((n+1,n))
#     Q = np.zeros((A.shape[0],n+1))
#     # Normalize the input vector
#     Q[:,0] =b/np.linalg.norm(b,2)   # Use it as the first Krylov vector
#     for k in range(1,n+1):
#         v = np.dot(A,Q[:,k-1])  # Generate a new candidate vector
#         for j in range(k):  # Subtract the projections on previous vectors
#             h[j,k-1] = np.dot(Q[:,j].T, v)
#             v = v - h[j,k-1] * Q[:,j]
#         h[k,k-1] = np.linalg.norm(v,2)
#         if h[k,k-1] > eps:  # Add the produced vector to the list, unless
#             Q[:,k] = v/h[k,k-1]
#         else:  # If that happens, stop iterating.
#             return Q, h
#     return Q, h


# def gmres_pd(A, b, x0, tolerance=1e-10, max_iter=10, type = np.float64):
#     res = []
#     Q = np.ones((b.shape[0], max_iter + 1), dtype=type) #zeros o ones?
#     H = np.zeros((max_iter+1, max_iter), dtype=type)
#     v = np.zeros((1, 1), dtype=type)
#     r0 = b - A.dot(x0)
#     r_norm = norm(r0, 2)
#     # r0 = b - A.dot(x0)
#     # beta = norm(r0, 2)
#     e1 = np.zeros((max_iter+1, 1))
#     e1[0] = 1
#     beta = r_norm * e1
#     Q[:, 0] = r0 / r_norm
#     y = 0

#     for j in range(max_iter-1):
#         #print("Arrivo j: ", j)
#         # ARNOLDI
#         Q[:, j+1] = A.dot(Q[:, j])
#         for h in range(5): # WTF IS THIS? (come funziona l'ortogonalizzazione?)
#             for i in range(j):
#                 v = Q[:, i].transpose().dot(Q[:, j+1])
#                 Q[:, j+1] = Q[:, j+1] - Q[:,i]*v   # TODO: forse più efficiente
#                 H[i, j] = H[i, j] + v
        
#         H[j+1, j] = norm(Q[:, j+1], 2)

        
#         if abs(H[j+1, j]) > tolerance:
#             Q[:, j+1] = Q[:, j+1] / H[j+1, j]

#         e1 = np.zeros(j+2)
#         e1[0] = 1
#         e1 = e1.transpose()

#         #y = least_squares(lambda x: H[:j+2, :j+1].dot(x) - e1.dot(beta), ).x
#         # print("H: \n", H[:j+2, :j+1])
#         # print("B: \n", beta * e1)
#         y = lstsq(H[:j+2, :j+1],  e1.dot(beta))[0]
#         #y = lstsq(H,  e1)[0]
        
#         # y = nnls(H[:j+2, :j+1],  beta * e1)[0]

#         # y = lstsq(H[:j+2, :j+1] , e1.dot(beta)) # TODO: io ci spero
#         res.append(norm(H[:j+2, :j+1].dot(y) - e1.dot(beta), 2))
        
#         if res[-1] < tolerance:
#             print('stop due')
#             return Q[:, :j+1].dot(y)+x0, res

#     return Q[:, :j+1].dot(y)+x0, res

def gmres_pd(A, b, x0, tollerance=1e-10, max_iter=10, type = np.float64):
    res = []
    Q = np.zeros((b.shape[0], max_iter + 1), dtype=type) #zeros o ones?
    H = np.zeros((max_iter+1, max_iter), dtype=type) #Perchè ha questa dimensione?
    r0 = b - A.dot(x0)
    beta = norm(r0, 2)
    Q[:, 0] = r0 / beta
    y = 0

    for j in range(max_iter-1):
        # ARNOLDI
        Q[:, j+1] = A.dot(Q[:, j])
        for _ in range(1):
            for i in range(j):
                H[i, j] = Q[:, i].transpose().dot(Q[:, j+1])
                Q[:, j+1] = Q[:, j+1] - Q[:,i].dot(H[i,j])   # TODO: forse più efficiente
        
        H[j+1, j] = norm(Q[:, j+1], 2)

        
        if abs(H[j+1, j]) > tollerance:
            Q[:, j+1] = Q[:, j+1] / H[j+1, j]

        e1 = np.zeros(j+2)
        e1[0] = 1
        e1 = e1.transpose()

        #y = least_squares(lambda x: H[:j+2, :j+1].dot(x) - e1.dot(beta), ).x
        # print("H: \n", H[:j+2, :j+1])
        # print("B: \n", beta * e1)
        y = lstsq(H[:j+2, :j+1],  e1.dot(beta))[0]
        # y = nnls(H[:j+2, :j+1],  beta * e1)[0]

        # y = lstsq(H[:j+2, :j+1] , e1.dot(beta)) # TODO: io ci spero
        res.append(norm(H[:j+2, :j+1].dot(y) - e1.dot(beta), 2))
        
        if res[-1] < tollerance:
            print('stop due')
            return Q[:, :j+1].dot(y)+x0, res

    return Q[:, :j+1].dot(y)+x0, res


def main():

    # Generate Data
    A, b, x0 = generate_data(20, 0.5, randx0=False, seed=42) # TODO: provare con `randx0=True`
    # A = np.array([[3, 2, -1],
    #                 [2, -2, 4],
    #                 [-1, 0.5, -1]], dtype=np.float64)

    # b = np.array([1, -2, 0], dtype=np.float64)
    # x0 = np.zeros(b.shape, dtype=np.float64)

    print(A)
    #A = sparse.csr_matrix(A)
    #b = sparse.csr_matrix(b)

    # Call gmres scipy
    res_scipy, info = gmres_scipy(A, b, x0, max_iter=20)
    #print('res_scipy ', res_scipy)

    # Call gmres
    x, res = gmres_pd(A, b, x0, tollerance=1e-15 ,max_iter=20, type=np.float64)
    #print('x gmres nostro ', x)

    # Solve
    x_solve = spsolve(A, b)
    #print('x solve ', x_solve)

    #print('nostro vs vero ', np.min(abs(x_solve - x)))
    print('nostro vs vero ', np.max(abs(x_solve - x))/np.max(abs(x)))
    print()
    #print('nostro vs scipy ', np.min(abs(x - res_scipy)))
    print('nostro vs scipy ', np.max(abs(x - res_scipy))/np.max(abs(res_scipy)))
    print()
    #print('scipy vs vero ', np.min(abs(x_solve - res_scipy)))
    print('scipy vs vero ', np.max(abs(x_solve - res_scipy))/np.max(abs(res_scipy)))

    print(np.max(abs(x - x_solve)))
    print(np.max(abs(x - res_scipy)))
    print(np.max(abs(x_solve - res_scipy)))

    
    fig, ax = plt.subplots()
    ax.semilogy()
    ax.plot(res)
    plt.show()

    print(x_solve - x)
    # print(x_solve)
    print(np.max(x_solve - x))

if __name__ == "__main__":
    main()