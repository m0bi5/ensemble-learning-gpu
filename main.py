#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from decision_tree import DecisionTree
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from ensemble import Bagging
import numpy as np
import random
import time
import pandas as pd

def TestRunTree():
    #Dataset
    wine = pd.read_csv("red_wine.csv")
    X = wine.drop('quality', axis=1)
    y = wine['quality']
    x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=0)
    x_train = x_train.values[:500]
    y_train = y_train.values[:500]
    x_test = x_test.values[:500]
    y_test = y_test.values[:500]
    '''
    dataset = load_iris()  
    print (type(dataset))
    X = dataset.data
    y = dataset.target
    x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=0)
    '''
    #Classifiers
    base_clf = DecisionTree(_max_depth=2, _min_splits=8, _gpu=False)
    base_clf_parallel = DecisionTree(_max_depth=2, _min_splits=8, _gpu=True)
    cpu_sequential = Bagging(base_clf=base_clf, n=5)
    cpu_parallel = base_clf
    gpu_sequential = Bagging(base_clf=base_clf_parallel, n=5)
    gpu_parallel = base_clf_parallel
    #Use 10% of data, 20%, 30% .....
    #Helps determine speedup achieved vs dataset size
    for i in range(30, 101, 33):
        train_batch_len = int(i/100 * len(x_train))
        test_batch_len = int(i/100 * len(x_test))
        x_train, y_train = x_train[:train_batch_len], y_train[:train_batch_len]
        x_test, y_test = x_test[:test_batch_len], y_test[:test_batch_len]
        #Sequential CPU
        start_cpu_sequential = time.time_ns()
        cpu_sequential.fit(x_train, y_train)
        y_pred = cpu_sequential.predict(x_test)
        print(y_pred)
        cpu_sequential_acc = accuracy_score(y_test, y_pred)
        end_cpu_sequential = time.time_ns()
        #Parallel CPU
        start_cpu_parallel = time.time_ns()
        cpu_parallel.fit(x_train, y_train)
        y_pred = cpu_parallel.predict(x_test)
        cpu_parallel_acc = accuracy_score(y_test, y_pred)
        end_cpu_parallel = time.time_ns()
        #Sequential GPU
        start_gpu_sequential = time.time_ns()
        gpu_sequential.fit(x_train, y_train)
        
        y_pred = gpu_sequential.predict(x_test)
        print(y_pred)
        gpu_sequential_acc = accuracy_score(y_test, y_pred)
        end_gpu_sequential = time.time_ns()
        
        #Parallel GPU
        start_gpu_parallel = time.time_ns()
        gpu_parallel.fit(x_train, y_train)
        y_pred = gpu_parallel.predict(x_test)
        gpu_parallel_acc = accuracy_score(y_test, y_pred)
        end_gpu_parallel = time.time_ns()

        print(
            'Run Count ', i ,':\n',
            "Dataset Size:%f\n"%(train_batch_len),
            "SequentialCPU: %fs, %f acc\n"%(end_cpu_sequential - start_cpu_sequential, cpu_sequential_acc),
            "ParallelCPU: %fs %f acc\n"%(end_cpu_parallel - start_cpu_parallel, cpu_parallel_acc),
            "SequentialGPU: %fs %f acc\n"%(end_gpu_sequential - start_gpu_sequential, gpu_sequential_acc),
            "ParallelGPU: %fs %f acc\n"%(end_gpu_parallel - start_gpu_parallel, gpu_parallel_acc),
            '--------------------------------------------------------------------------------------------'
        )
    

def main():
    starTestRun = time.time_ns()
    TestRunTree()
    endTestRun = time.time_ns()
    print( "Test Run Time : %fs \n"%(endTestRun- starTestRun))

if __name__ == '__main__':
	main()