#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from decision_tree import DecisionTree
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from ensemble import Bagging
from ensemble_GPU import DecisionTreeGPU, BaggingGPU
import numpy as np
#from arboretum import RFClassifier
import random
import time
import pandas as pd

def TestRunTree():
    #Dataset
    titanic = pd.read_csv("titanic.csv")
    X = titanic.drop('Survived', axis=1)
    X = titanic.drop('Name', axis=1)
    y = titanic['Survived']
    x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=0)
    x_train = x_train.values
    y_train = y_train.values
    x_test = x_test.values
    y_test = y_test.values
    '''
    dataset = load_iris()  
    X = dataset.data
    y = dataset.target
    x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=0)
    '''
    train_og_length = len(x_train)
    test_og_length = len(x_test)
    #Use 20% of data, 20%, 40% .....
    #Helps determine speedup achieved vs dataset size
    for i in range(20, 101, 20):

        #Classifiers
        cpu_sequential = Bagging(n=10, parallel=False, gpu=False)
        cpu_parallel = Bagging(n=10, parallel=True, gpu=False)
        gpu_sequential = BaggingGPU(n=10, parallel=False, gpu=True)
        gpu_parallel = BaggingGPU(n=10,parallel=True, gpu=True)

        train_batch_len = int(i/100 * train_og_length)
        test_batch_len = int(i/100 * test_og_length)
        x_train, y_train = x_train[:train_batch_len], y_train[:train_batch_len]
        x_test, y_test = x_test[:test_batch_len], y_test[:test_batch_len]

        #Sequential CPU
        start_cpu_sequential = time.time_ns()
        cpu_sequential.fit(x_train, y_train)
        y_pred = cpu_sequential.predict(x_test)
        cpu_sequential_acc = accuracy_score(y_test, y_pred)
        end_cpu_sequential = time.time_ns()

        #Parallel CPU
        start_cpu_parallel = time.time_ns()
        cpu_parallel.fit(x_train, y_train)
        y_pred = cpu_parallel.predict(x_test)
        cpu_parallel_acc = accuracy_score(y_test, y_pred)
        end_cpu_parallel = time.time_ns()

        #Sequential GPU
        gpu_sequential.fit(x_train, y_train)  #Warmup GPU run, don't time it
        start_gpu_sequential = time.time_ns()
        gpu_sequential.fit(x_train, y_train)
        y_pred = gpu_sequential.predict(x_test)
        gpu_sequential_acc = accuracy_score(y_test, y_pred)
        end_gpu_sequential = time.time_ns()
        
        #Parallel GPU
        gpu_parallel.fit(x_train, y_train)  #Warmup GPU run, don't time it
        start_gpu_parallel = time.time_ns()
        gpu_parallel.fit(x_train, y_train)
        y_pred = gpu_parallel.predict(x_test)
        gpu_parallel_acc = accuracy_score(y_test, y_pred)
        end_gpu_parallel = time.time_ns()

        print(
            'Run Count ', i ,':\n',
            "Dataset Size:%d\n"%(train_batch_len),
            "SequentialCPU: %d ms, %f acc\n"%((end_cpu_sequential - start_cpu_sequential) // 1e6 , cpu_sequential_acc),
            "ParallelCPU: %d ms %f acc\n"%((end_cpu_parallel - start_cpu_parallel) // 1e6, cpu_parallel_acc),
            "SequentialGPU: %d ms %f acc\n"%((end_gpu_sequential - start_gpu_sequential) // 1e6, gpu_sequential_acc),
            "ParallelGPU: %d ms %f acc\n"%((end_gpu_parallel - start_gpu_parallel) // 1e6, gpu_parallel_acc),
            '--------------------------------------------------------------------------------------------'
        )
    

def main():
    starTestRun = time.time_ns()
    TestRunTree()
    endTestRun = time.time_ns()
    print( "Test Run Time : %fs \n"%(endTestRun- starTestRun))

if __name__ == '__main__':
	main()