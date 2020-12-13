#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from timeit import default_timer as timer
import numpy as np
import numba as nb
from random import seed
from random import randrange
from decision_tree import DecisionTree
from statistics import mode
import random
import multiprocessing

def worker(clf, x_train, y_train):
    dataset = list(zip(x_train, y_train))
    resampled = self.resample(dataset)
    x_train, y_train = list(zip(*resampled))
    clf.fit(x_train, y_train)

@nb.njit
def numba_mode(arr):
    _max = max(arr)[0] + 1
    temp = np.array([0]*_max)
    for i in arr:
        temp[i[0]] += 1
    return np.argmax(temp)

class Bagging():
    def __init__(self, n, parallel=False, gpu=False):
        self.clfs = []
        self.n = n
        self.parallel = parallel
        for i in range(n):
            self.clfs.append(DecisionTree(_max_depth=2, _min_splits=6, _gpu=gpu))
    
    def resample(self, data):
        return random.choices(data, k=len(data))

    def fit(self, x_train, y_train):
        dataset = list(zip(x_train, y_train))
        if self.parallel == False:
            for i in range(self.n):
                resampled = self.resample(dataset)
                x_train, y_train = list(zip(*resampled))
                self.clfs[i].fit(x_train, y_train)
        else:
            #Create one process for each tree
            pool = multiprocessing.Pool(self.n)
            for i in range(self.n):
                self.clfs[i] = pool.apply_async(self.clfs[i].fit, (x_train, y_train))
            
            for i in range(self.n):
                self.clfs[i] = self.clfs[i].get()
            
            pool.close()
    
    def predict(self, x_test):
        y_preds = []
        for i in range(len(x_test)):
            row_pred = []
            for j in range(self.n):
                row_pred.append(self.clfs[j].predict([x_test[i]]))
            try:
                if self.gpu:
                    y_preds.append(numba_mode(row_pred))
                else:
                    y_preds.append(mode(row_pred))

            except:
                y_preds.append(random.choice(row_pred))
        return y_preds