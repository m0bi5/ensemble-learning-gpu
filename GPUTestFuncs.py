

# Timing
from timeit import default_timer as timer
import numpy as np
import numba as nb
from  numba import cuda, float32, int32, jit, njit, vectorize,prange
from math import exp, ceil


# CPU funcs
@njit(parallel= True)
def numba_sum(arr):    
    s = 0 
    for i in prange(arr.shape[0]):
        s += arr[i]
    return s#arr.sum()

@vectorize(['float32(float32[:])'], target='cuda')
def numba_sum_arrayS(arr):
    s = 0.0 
    l = arr.shape[0]
    for i in range(l):
        s =s + arr[i]
    #print (s)
    #s =arr.sum()
    print(s)
    return s
# GPU func Test
# Things to make Parrallel
# sum
#@vector
@cuda.jit#(('(float32[:],int32)'), device=True, inline=True)
def numba_sum_array(arr,s):
    #s = 0.0
    #tid = cuda.threadIdx.x
    #bid = cuda.blockIdx.x
    #bdim = cuda.blockDim.x
    #i = (bid * bdim) + tid
    i = cuda.grid(1)
    #if i < arr.shape[0]:
        #for j in range(arr.shape[0]):
    cuda.atomic.add(s,0,arr[i])# += arr[j]  
    #return s
       

# Array Multiplication
#@cuda.jit()

@vectorize(['float32(float32, float32)'], target='cuda')
def Numba_array_mult(A,B):
   #row = cuda.grid(1)
   #tx = cuda.threadIdx.row
   #bgp = cuda.gridDim.row  # blocks per grid

    #for i in range(bgp):
    return A * B

@vectorize(['float32(float32, float32)'], target='cuda')    
def numbaScalarMult(A, B):
    return A*B

@cuda.jit#([float32[:],float32[:],float32[:] ])
def Numba_array_mult_wKern(A,B,c):
    row = cuda.grid(1)
    tx = cuda.threadIdx.x
    bgp = cuda.gridDim.x  # blocks per grid
    height = c.shape[0]
    for i in range(row, height, bgp):
        c[i]= A[i]*B[i]
    #for i in range(bgp):
    

# Len()

# Scalar Mult and Addition
# Final Gini
# Final Gini
@vectorize(['float32 (float32, float32, int32)'], target='cuda')
def gini_score(score, size, num_sample):
    return (1 - score) * (size/num_sample)

# adding two  branches / Nodes 

# incrementers

# Random choise
def main():
    Test_array1 = np.arange(10, dtype=np.float32)
    print ("Test array 1: ", Test_array1.shape)
    
    Test_array2 = np.arange(10, dtype=np.float32)
    print ("Test array 2: ", Test_array2.shape)
    
    TestNdarry = np.arange(20, dtype=np.float32)
    print ("Test ndarray Before reshape: ", TestNdarry.shape)
    TestNdarry = np.reshape(TestNdarry,( int(TestNdarry.shape[0]/2), 2 ))
    print ("Test ndarray: ", TestNdarry.shape)
    
    s = np.zeros(1, dtype=np.float32)
    threadsperblock = 32
    blockspergrid = ceil(Test_array1.shape[0] / threadsperblock)
    numba_sum_array[blockspergrid, threadsperblock](Test_array1,s)
    
    print("Output from numba_sum_array: ", s[0])

    print("Numpy array sum: ",Test_array1.sum())
    print("CPU array sum: ", numba_sum_arrayS(Test_array1))
    
    output_array = np.zeros(Test_array1.shape, dtype=np.float32)
    #numba_sum.parallel_diagnostics(level=4)
    #mulArray = Numba_array_mult(Test_array1,Test_array2)
    Numba_array_mult_wKern[blockspergrid, threadsperblock](Test_array1, Test_array2, output_array)
    print("Output from Numba_array_mult: ", output_array)
    print("Numpy array sum: ",Test_array1 * Test_array2)
    Numba_array_mult(Test_array1,Test_array2)
    gi = gini_score(10, 5, 3)    
    print("gini score GPU: ", gi[0])
    print("Regular calc: ", (1.0 - 10) * (5/3) )
    #numba_sum_.parallel_diagnostics(level=4)


if __name__ == '__main__':
	main() 