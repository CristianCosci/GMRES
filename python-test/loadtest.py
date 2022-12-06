import scipy.io as sio
import mat73
import numpy as np
from scipy.linalg import lstsq
from scipy.sparse.linalg import gmres


def loadmatrix(path:str):
    try:
        data_dict = mat73.loadmat(path)
    except:
        print('Vecchio formato')
        data_dict = sio.loadmat(path)

    return data_dict

# data_dict = mat73.loadmat('mawi_201512020330.mat')
# data_dict = mat73.loadmat('webbase-2001.mat')
data_dict = loadmatrix('cage15.mat')

print('Matrice Caricata')

# print(type(data_dict))
# print(data_dict.keys())
# print(data_dict['Problem'].keys())
# exit()
A = data_dict['Problem']['A']
b = np.random.rand(A.shape[0])
x0 = np.zeros(A.shape[0])

# A = np.array([[1,0,0],[0,2,0],[0,0,3]])
# b = np.array([1, 4, 6])
# x0 = np.zeros(b.size)

print('Avvio GMRES')
res, info = gmres(A, b, x0=x0, tol=1e-05, restart=None)

print(res)
print(info)

# print(A)
# print(b)

# res = A.dot(b)

# print(res)
# print(res.shape)