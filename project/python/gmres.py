from typing import Tuple
from scipy import sparse
import numpy as np

from scipy.linalg import lstsq
from scipy.linalg import norm
from scipy import sparse
from scipy.sparse.linalg import gmres
from scipy.sparse.linalg import spsolve

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



def gmres_pd(A, b, x0, tolerance=1e-10, max_iter=10, type = np.float64):
    res = []
    Q = np.ones((b.shape[0], max_iter + 1), dtype=type) #zeros o ones?
    H = np.zeros((max_iter+1, max_iter), dtype=type)
    v = np.zeros((1, 1), dtype=type)
    r0 = b - A.dot(x0)
    beta = norm(r0, 2)
    Q[:, 0] = r0 / beta
    y = 0

    for j in range(max_iter-1):
        #print("Arrivo j: ", j)
        # ARNOLDI
        Q[:, j+1] = A.dot(Q[:, j])
        for h in range(5): # WTF IS THIS? (come funziona l'ortogonalizzazione?)
            for i in range(j):
                v = Q[:, i].transpose().dot(Q[:, j+1])
                Q[:, j+1] = Q[:, j+1] - Q[:,i]*v   # TODO: forse più efficiente
                H[i, j] = H[i, j] + v
        
        H[j+1, j] = norm(Q[:, j+1], 2)

        
        if abs(H[j+1, j]) > tolerance:
            Q[:, j+1] = Q[:, j+1] / H[j+1, j]

        e1 = np.zeros(j+2)
        e1[0] = 1
        e1 = e1.transpose()

        #y = least_squares(lambda x: H[:j+2, :j+1].dot(x) - e1.dot(beta), ).x
        # print("H: \n", H[:j+2, :j+1])
        # print("B: \n", beta * e1)
        y = lstsq(H[:j+2, :j+1],  e1.dot(beta))[0]
        #y = lstsq(H,  e1)[0]
        
        # y = nnls(H[:j+2, :j+1],  beta * e1)[0]

        # y = lstsq(H[:j+2, :j+1] , e1.dot(beta)) # TODO: io ci spero
        res.append(norm(H[:j+2, :j+1].dot(y) - e1.dot(beta), 2))
        
        if res[-1] < tolerance:
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


    #A = sparse.csr_matrix(A)
    #b = sparse.csr_matrix(b)

    # Call gmres scipy
    res_scipy, info = gmres_scipy(A, b, x0, max_iter=50)
    #print('res_scipy ', res_scipy)

    # Call gmres
    x, res = gmres_pd(A, b, x0, tollerance=1e-15 ,max_iter=150, type=np.float64)
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
    
    fig, ax = plt.subplots()
    ax.semilogy()
    ax.plot(res)
    plt.show()
    
    

if __name__ == "__main__":
    main()