#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timing
from timeit import default_timer as timer
import numpy as np
import numba as nb
from math import exp, ceil
from statistics import mode
from  numba import cuda, float32, int32, jit, njit, vectorize,prange
import random
# Things to make Parrallel
# sum

# Serial
#@vectorize(['float32 (float32[:])'], target='cuda')
#def numba_sum_arrayS(arr):
#   return arr.sum()  

# Parrallel
@cuda.jit
def numba_sum_arrayP(arr,s):
    i = cuda.grid(1)    
    cuda.atomic.add(s,0,arr[i])

# Serial
#@vectorize(['float32[:](float32[:], float32[:])'], target='cuda')
#def Numba_array_mult(A,B):
#    return A * B


# Array Multiplication Parrallel
@cuda.jit
def Numba_array_mult_wKern(in1,in2,out):
    row = cuda.grid(1)
    tx = cuda.threadIdx.x
    bgp = cuda.gridDim.x  # blocks per grid
    height = out.shape[0]
    for i in range(row, height, bgp):
        out[i]= in1[i] * in2[i]
# Len()

# Scalar Mult and Addition serial
@vectorize(['float32(float32, float32)'], target='cuda')    
def numbaScalarMult(A, B):
    return A*B

# Final Gini Serial
@vectorize(['float32 (float32, float32, int32)'], target='cuda')
def gini_score_v(score, size, num_sample):
    return (1 - score) * (size/num_sample)
    #return gini_score

# adding two  branches / Nodes 

# incrementers

# Random choise
@njit()
def numba_sum(arr):    
    s = 0 
    for i in prange(arr.shape[0]):
        s += arr[i]
    return s#arr.sum()
@njit
def numba_max(arr):
  return np.argmax(arr)

class DecisionTreeGPU(object):
    """
    Class to create decision tree model (CART)
    """
    def __init__(self, _max_depth, _min_splits, _gpu=False):
        self.max_depth = _max_depth
        self.min_splits = _min_splits
        self.gpu = _gpu

    def fit(self, _feature, _label):
        """
        :param _feature:
        :param _label:
        :return:
        """
        self.feature = _feature
        self.label = _label
        
        # Can this be paralllel?
        self.train_data = np.column_stack((self.feature,self.label))
        
        
        self.build_tree()
        return self

    #@vectorize(['float32(float32[:],float32[:])'])
    def compute_gini_similarity(self, groups, class_labels):
        """
        compute the gini index for the groups and class labels
        :param groups:
        :param class_labels:
        :return:
        """
        # Paralle
        num_sample = sum([len(group) for group in groups])  # Parallel
        gini_score = 0

        for group in groups:
            size = float(len(group))

            if size == 0:
                continue
            score = 0.0
            for label in class_labels:
                to_sum = (group[:,-1] == label)
                _sum = 0
                if self.gpu:
                    #print(type(to_sum))
                     #= numba_sum(,len(to_sum))
                    s = np.zeros(1, dtype=np.float32)
                    #arr= np.array(to_sum)
                    threadsperblock = 32
                    blockspergrid = ceil(np.array(to_sum).shape[0] / threadsperblock)                    
                    numba_sum_arrayP[blockspergrid, threadsperblock](np.array(to_sum),s)
                    _sum = s[0]
                else:   
                    _sum = to_sum.sum() # ruun on GPU
                proportion =  _sum / size
                score += numbaScalarMult(proportion,  proportion)[0]
            
            gini_score += gini_score_v(score,size,int32(num_sample))[0]   
        #print(gini_score)
        return gini_score

    def terminal_node(self, _group):
        """
        Function set terminal node as the most common class in the group to make prediction later on
        is an helper function used to mark the leaf node in the tree based on the early stop condition
        or actual stop condition which ever is meet early
        :param _group:
        :return:
        """
        #print(_group[:,-1])
        class_labels, count = np.unique(_group[:,-1], return_counts= True)  
        _max = None
        try:
            if self.gpu:
                _max = numba_max(np.array(count))   # Parallel
            else:
                _max = np.argmax(count)   # GPU    
        except:
            return None
        
        return class_labels[_max]

    def split(self, index, val, data):
        """
        split features into two groups based on their values
        :param index:
        :param val:
        :param data:
        :return:
        """
        data_left = np.array([]).reshape(0,self.train_data.shape[1])
        data_right = np.array([]).reshape(0, self.train_data.shape[1])

        for row in data:
            if row[index] <= val :
                data_left = np.vstack((data_left,row))

            else:#if row[index] >= val:
                data_right = np.vstack((data_right, row))

        return data_left, data_right

    def best_split(self, data):
        """
        find the best split information using the gini score
        :param data:
        :return best_split result dict:
        """
        class_labels = np.unique(data[:,-1])
        best_index = float('inf')
        best_val = float('inf')
        best_score = float('inf')
        best_groups = None
       
        for idx in range(data.shape[1]-1):
            
            for row in data:
                groups = self.split(idx, row[idx], data)
                gini_score = self.compute_gini_similarity(groups,class_labels)

                if gini_score < best_score:
                    best_index = idx
                    best_val = row[idx]
                    best_score = gini_score
                    best_groups = groups
        result = {}
        result['index'] = best_index
        result['val'] = best_val
        result['groups'] = best_groups
        return result


    def split_branch(self, node, depth):
        """
        recursively split the data and
        check for early stop argument based on self.max_depth and self.min_splits
        - check if left or right groups are empty is yess craete terminal node
        - check if we have reached max_depth early stop condition if yes create terminal node
        - Consider left node, check if the group is too small using min_split condition
            - if yes create terminal node
            - else continue to build the tree
        - same is done to the right side as well.
        else
        :param node:
        :param depth:
        :return:
        """
        left_node , right_node = node['groups']

        del(node['groups'])

        if not isinstance(left_node,np.ndarray) or not isinstance(right_node,np.ndarray):
            node['left'] = self.terminal_node(left_node + right_node) ## PArralel
            node['right'] = self.terminal_node(left_node + right_node)  ## Parallel
            return

        if depth >= self.max_depth:
            node['left'] = self.terminal_node(left_node)
            node['right'] = self.terminal_node(right_node)
            return

        if len(left_node) <= self.min_splits:
            node['left'] = self.terminal_node(left_node)
        else:
            node['left'] = self.best_split(left_node)
            self.split_branch(node['left'],depth + 1)


        if len(right_node) <= self.min_splits:
            node['right'] = self.terminal_node(right_node)
        else:
            node['right'] = self.best_split(right_node)
            self.split_branch(node['right'],depth + 1)

    def build_tree(self):
        """
        build tree recursively with help of split_branch function
         - Create a root node
         - call recursive split_branch to build the complete tree
        :return:
        """
        
        self.root = self.best_split(self.train_data)
        
        self.split_branch(self.root, 1)
        return self.root

    def _predict(self, node, row):
        """
        Recursively traverse through the tress to determine the
        class of unseen sample data point during prediction
        :param node:
        :param row:
        :return:
        """
        if row[node['index']] < node['val']:
            if isinstance(node['left'], dict):
                return self._predict(node['left'], row)
            else:
                return node['left']
        else:
            if isinstance(node['right'],dict):
                return self._predict(node['right'],row)
            else:
                return node['right']

    def predict(self, test_data):
        """
        predict the set of data point
        :param test_data:
        :return:
        """
        self.predicted_label = np.array([])
        for idx in test_data:
            self.predicted_label = np.append(self.predicted_label, self._predict(self.root,idx))

        return self.predicted_label

class BaggingGPU():
    #@vectorize([''])
    def __init__(self, n,  gpu=False, parallel=False):
        self.clfs = []
        self.n = n
        self.parallel = parallel
        for i in range(n):
            self.clfs.append(DecisionTreeGPU(_max_depth=2, _min_splits=6, _gpu=gpu))            
       
    def resample(self, data):
        return random.choices(data, k=len(data))
    
    def fit(self, x_train, y_train):
        dataset = list(zip(x_train, y_train))
        for i in range(self.n):
            resampled = self.resample(dataset)
            x_train, y_train = list(zip(*resampled))
            self.clfs[i].fit(x_train, y_train)
    
    def predict(self, x_test):
        y_preds = []
        for i in range(len(x_test)):
            row_pred = []
            for j in range(self.n):
                row_pred.append(self.clfs[j].predict([x_test[i]]))
            try:
                y_preds.append(mode(row_pred))
            except:
                y_preds.append(random.choice(row_pred))
        return y_preds

"""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from decision_tree import DecisionTree
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import random
import time
import pandas as pd
#from decision_tree import DecisionTree
def main():
     #Dataset
    wine = pd.read_csv("red_wine.csv")
    X = wine.drop('quality', axis=1)
    y = wine['quality']
    x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=0)
    x_train = x_train.values[:50]
    y_train = y_train.values[:50]
    x_test = x_test.values[:50]
    y_test = y_test.values[:50]
    
    
    # CPU
    base_clf = DecisionTreeGPU(_max_depth=2, _min_splits=8, _gpu=False)
    #Sequential GPU
    base_clf.fit(x_train, y_train)
    start_cpu_parallel = time.time_ns()
    base_clf.fit(x_train, y_train)
    y_pred = base_clf.predict(x_test)
    cpu_parallel_acc = accuracy_score(y_test, y_pred)
    end_cpu_parallel = time.time_ns()
    #print("Converting CPU data ")
    #x_train = cp.array(x_train)
    #y_train = cp.array(y_train)
    #x_test = cp.array(x_test)
    #y_test = cp.array(y_test)
    #print("Convert CPU data ")
    gpu_sequential = DecisionTreeGPU(_max_depth=2, _min_splits=8, _gpu=True)
    #Sequential GPU
    gpu_sequential.fit(x_train, y_train)
    start_gpu_sequential = time.time_ns()
    gpu_sequential.fit(x_train, y_train)
        
    y_pred = gpu_sequential.predict(x_test)
        #print(y_pred)
    gpu_sequential_acc = accuracy_score(y_test, y_pred)
    end_gpu_sequential = time.time_ns()
    print(            
        #"SequentialCPU: %fs, %f acc\n"%(end_cpu_sequential - start_cpu_sequential, cpu_sequential_acc),
        "ParallelCPU: %fs %f acc\n"%(end_cpu_parallel - start_cpu_parallel, cpu_parallel_acc),
        "SequentialGPU: %fs %f acc\n"%(end_gpu_sequential - start_gpu_sequential, gpu_sequential_acc),
        #"ParallelGPU: %fs %f acc\n"%(end_gpu_parallel - start_gpu_parallel, gpu_parallel_acc),
        '--------------------------------------------------------------------------------------------'
    )
    from ensemble import Bagging
    gpu_sequential = BaggingGPU(n=10, gpu=True, parallel=False)
    gpu_parallel = BaggingGPU(n=10, parallel=True, gpu=True)
    gpu_sequential.fit(x_train, y_train)
    y_pred = gpu_sequential.predict(x_test)
    gpu_parallel.fit(x_train, y_train)
    y_pred = gpu_parallel.predict(x_test)
   
if __name__ == '__main__':
	main() 
"""