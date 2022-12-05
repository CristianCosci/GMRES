# import scipy.io as sio
import mat73
import numpy as np

data_dict = mat73.loadmat('mawi_201512020330.mat')

print(type(data_dict))
print(data_dict.keys())
print(data_dict['Problem'].keys())

A = data_dict['Problem']['A']
b = np.random.rand(A.shape[0])

print(A)
print(b)

res = A.dot(b)

print(res)
print(res.shape)