#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from tensorflow.keras.datasets import mnist
from tensorflow.keras import utils
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from timeit import default_timer as timer
#from trees import ForestCPU, ForestGPU
import numpy as np
import random

def TestRunTree():
    #Dataset
    (x_train, y_train), (x_test, y_test) = dataset = mnist.load_data(path='mnist.npz')
    classes = 10
    #Classifiers
    cpu_sequential = DecisionTreeClassifier(max_depth=10)
    cpu_parallel = DecisionTreeClassifier(max_depth=10)
    gpu_sequential = DecisionTreeClassifier(max_depth=10)
    gpu_parallel = DecisionTreeClassifier(max_depth=10)
    #Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_train = [i.flatten() for i in x_train]
    x_test = x_test.astype("float32") / 255
    x_test = [i.flatten() for i in x_test]
    #Convert class vectors to binary class matrices
    y_train = utils.to_categorical(y_train, classes)
    y_test = utils.to_categorical(y_test, classes)
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
        start_cpu_sequential = timer()
        cpu_sequential.fit(x_train, y_train)
        y_pred = cpu_sequential.predict(x_test)
        cpu_sequential_acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
        end_cpu_sequential = timer()
        #Parallel CPU
        start_cpu_parallel = timer()
        cpu_parallel.fit(x_train, y_train)
        y_pred = cpu_parallel.predict(x_test)
        cpu_parallel_acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
        end_cpu_parallel = timer()
        #Sequential GPU
        start_gpu_sequential = timer()
        gpu_sequential.fit(x_train, y_train)
        y_pred = gpu_sequential.predict(x_test)
        gpu_sequential_acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
        end_gpu_sequential = timer()
        #Parallel GPU
        start_gpu_parallel = timer()
        gpu_parallel.fit(x_train, y_train)
        y_pred = gpu_parallel.predict(x_test)
        gpu_parallel_acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
        end_gpu_parallel = timer()

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
    starTestRun = timer()
    TestRunTree()
    endTestRun = timer()
    print( "Test Run Time : %fs \n"%(endTestRun- starTestRun))

if __name__ == '__main__':
	main()