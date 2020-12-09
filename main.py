#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from decision_tree import DecisionTree
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import random
import time

def TestRunTree():
    #Dataset
    dataset = load_iris()
    X = dataset.data
    y = dataset.target
    x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=0)
    classes = 3
    #Classifiers
    cpu_sequential = DecisionTree(_max_depth = 2, _min_splits = 5)
    cpu_parallel = DecisionTreeClassifier(max_depth=10)
    gpu_sequential = DecisionTreeClassifier(max_depth=10)
    gpu_parallel = DecisionTreeClassifier(max_depth=10)
    train_dataset = list(zip(x_train, y_train))
    test_dataset = list(zip(x_test, y_test))
    #Use 10% of data, 20%, 30% .....
    #Helps determine speedup achieved vs dataset size
    for i in range(10, 101, 10):
        train_batch_len = i/100 * len(train_dataset)
        test_batch_len = i/100 * len(test_dataset)
        #Generate i% subset of data
        train_batch = random.choices(train_dataset, k=int(train_batch_len))
        test_batch = random.choices(test_dataset, k=int(test_batch_len))
        x_train, y_train = zip(*train_batch)
        x_test, y_test = zip(*test_batch)
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
        start_gpu_sequential = time.time_ns()
        gpu_sequential.fit(x_train, y_train)
        
        y_pred = gpu_sequential.predict(x_test)
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