from ctypes import *

so_file = r'./cfunctions/linalg_functions.so'

my_functions = CDLL(so_file)
my_functions.power.argtypes = [c_float]
my_functions.power.restype = c_float