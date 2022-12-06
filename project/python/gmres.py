from typing import Tuple
from scipy import sparse
import numpy as np

from scipy.linalg import lstsq
from scipy.linalg import norm
from scipy.sparse.linalg import gmres
from scipy.optimize import least_squares


def generate_data(dim:int, den:float, seed=911, randx0=False) -> Tuple[np.ndarray, np.array, np.array]:
    """
    
    Genera i dati iniziali per il avviare GMRES
    
    """
    
    np.random.seed(seed)
    
    A = sparse.random(dim, dim, density=den , format="csr", dtype=np.float32)
    b = np.random.random(dim)

    if randx0:
        x0 = np.random.random(dim)
    else:
        x0 = np.zeros(dim)

    return A, b, x0


def gmres_scipy(A, b, x0, tollerance=1e-5, max_iter=10):
    """"
    Invoca la funzione GMRES di SciPy con i dati passati
    """

    return gmres(A,b, x0=x0, tol=tollerance, maxiter=max_iter)



def gmres_pd(A, b, x0, tollerance=1e-5, max_iter=10):
    Q = np.zeros((b.shape[0], max_iter + 1), dtype=np.float64) #zeros o ones?
    H = np.zeros((max_iter+1, max_iter), dtype=np.float64) #Perchè ha questa dimensione?
    r0 = b - A.dot(x0)
    beta = norm(r0, 2)
    Q[:, 0] = r0 / beta
    y = 0

    for j in range(max_iter-1):
        # ARNOLDI
        Q[:, j+1] = A.dot(Q[:, j])
        for i in range(j):
            #  ???
            H[i, j] = Q[:, i].transpose().dot(Q[:, j+1])
            Q[:, j+1] = Q[:, j+1] - H[i,j] * (Q[:,i])   # TODO: forse più efficiente
        
        H[j+1, j] = norm(Q[:, j+1], 2)

        
        if abs(H[j+1, j]) > tollerance:
            Q[:, j+1] = Q[:, j+1] / H[j+1, j]

        e1 = np.zeros((j+2,1))
        e1[0] = 1
        # e1 = e1.transpose()
        
        # print(e1)
        # print(H[:j+2, :j+1])
        # print(H[:j+2, :j+1].shape)
        # print((beta * e1).shape)
        #y = least_squares(lambda x: H[:j+2, :j+1].dot(x) - e1.dot(beta), ).x
        print("H: \n", H[:j+2, :j+1])
        print("B: \n", beta * e1)
        y = lstsq(H[:j+2, :j+1],  beta * e1)[0]
        # y = lstsq(H,  beta * e1)[0]

        #print('Y DIO MAIALE ', y)
        # y = lstsq(H[:j+2, :j+1] , e1.dot(beta)) # TODO: io ci spero
        res = norm(H[:j+2, :j+1].dot(y) - e1.dot(beta), 2)
        if res < tollerance:
            return Q[:, :j+1].dot(y)+x0, res

    return Q[:, :j+1].dot(y)+x0, res


def main():

    #print("------------------------\n")
    # Generate Data
    A, b, x0 = generate_data(3, 0.5, randx0=False) # TODO: provare con `randx0=True`
    
    # Call gmres scipy
    res_scipy, info = gmres_scipy(A, b, x0)

    # print(res)
    # print(info)


    #print("-----------------------\n")

    # Call gmres porco dio
    x, res = gmres_pd(A, b, x0)

    print(np.min(res - res_scipy))
    print(np.max(res - res_scipy))
    #print(x)
    #print(res)
    # Compare the two GMRES 


if __name__ == "__main__":
    main()
