import random
from numba import njit
from scipy.stats import mode
import numpy as np
from timeit import default_timer as timer  

def func2(arr, temp):
    for i in arr:
        temp[i] += 1
    return np.argmax(temp)

@njit
def func1(arr, temp):
    for i in arr:
        temp[i] += 1
    return np.argmax(temp)


a = []
for i in range(100000):
    a.append(random.randint(1,500))


_max = max(a) + 1
temp = np.array([0]*_max)
start = timer() 
func1(a, temp) 
print("with GPU:", timer()-start)     

_max = max(a) + 1
temp = np.array([0]*_max)
start = timer() 
func2(a, temp) 
print("without GPU:", timer()-start) 