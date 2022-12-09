import numpy as np 
from numpy import linalg as LA


def arnoldi_iteration(A, b, n: int, eps = 1e-12):
    
    """
    Computes a basis of the (n + 1)-Krylov subspace of A: the space
    spanned by {b, Ab, ..., A^n b}.

    Arguments
      A: m × m array
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
     # Normalize the input vector
    q = b/np.linalg.norm(b,2)
    print(q)
    Q[:,0] =  q.transpose()  # Use it as the first Krylov vector
    print(Q)
    for k in range(1,n+1):
        print("------------------------------------------------")
        print("k",k)
        v = np.dot(A,Q[:,k-1])  # Generate a new candidate vector
        #print("v", v)
        for j in range(k):  # Subtract the projections on previous vectors
            h[j,k-1] = np.dot(Q[:,j].T, v)
            print(" h[j,k-1]: ", h[j,k-1])
            v = v - h[j,k-1] * Q[:,j]
        h[k,k-1] = np.linalg.norm(v,2)
        if h[k,k-1] > eps:  # Add the produced vector to the list, unless
            Q[:,k] = v/h[k,k-1]
        else:  # If that happens, stop iterating.
            return Q, h
    return Q, h


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
    
    Q, H = arnoldi_iteration(A, b, 9, eps = 1e-12)
    
    print()
    print("This is Q")
    print(Q)
    print()
    print("This is H")
    print(H)
    print()
    
    u,v = LA.eigh(A)
    
    print("This is u")
    print(u)
    print()
    print("This is v")
    print(v)

    
    
if __name__ == "__main__":
    main()

