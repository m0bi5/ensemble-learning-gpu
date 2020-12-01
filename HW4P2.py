#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ECE5984 SP20 HW4 Part 2 - Random Forest Classifier
Created on Thu Feb 20 17:41:33 2020
@author: ccody7
"""
import pandas as pd
import numpy as np
from sklearn import neighbors
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier

# 1. Load dataset and 2. Seperate target from feature values
def loadData(pathName, dataFile, target):
	# 1. Load Dataset 
    dataFrameTrain = pd.read_excel(pathName + dataFile, sheet_name='train')
    dataFrameTest = pd.read_excel(pathName + dataFile, sheet_name='test')
    # 2. Seperate Targets from features
    trainX = dataFrameTrain.drop([target], axis=1)
    trainY = dataFrameTrain[target]
    testX = dataFrameTest.drop([target], axis=1)
    testY = dataFrameTest[target]
    return trainX, trainY, testX, testY
# 3. Nomalize the dataset approapraiatly 
def NormalizeData(data):
    return( preprocessing.scale(data))
 
# 7. Missclassification rate of each classifier 
def misclassificationRateOut(Alg, Xtest, ytest,ypred):
    #print ( Alg , ' :Number of mislabeled points out of a total of r00 points : ', numMiss)
    print("%s: Number of mislabeled points out of a total %d points : %d"% (Alg, Xtest.shape[0], (ytest!= ypred).sum()))
    confusionMatrixOut(ytest,ypred)
# 8. Print Confussion Matrix
def confusionMatrixOut(ytest,ypred):
    confusion = confusion_matrix(ytest,ypred)
    print(confusion)
# 2. Random Foest Classifier
def RandomForest(Xtrain, ytrain, Xtest, ytest):
    # Normalize     
    Xtrain = NormalizeData(Xtrain)
    Xtest = NormalizeData(Xtest)
    #clf = RandomForestClassifier(n_estimators=15)
    clf = RandomForestClassifier( n_estimators=15, max_depth=None, min_samples_split=2, max_features='auto')
    clf.fit(Xtrain, ytrain)
    
    ypred = clf.predict(Xtest)
    misclassificationRateOut('RandomForest', Xtest, ytest,ypred)
    
    
def main():
    pathName = "C:\\Users\\Cody\\OneDrive\\Documents\\VT\\ECE5984\\hw\\DATASETS\\"
    shuttleFile = 'shuttle.xlsx'
    shuttleTarget = "class"
    wifiFile= 'wifi.xlsx'
    wifiTarget = "level"
    
    # Shuttledata 
    trainX, trainY, testX, testY = loadData(pathName, shuttleFile, shuttleTarget)
    RandomForest(trainX, trainY, testX, testY)
    #wifiData
    trainX, trainY, testX, testY = loadData(pathName, wifiFile, wifiTarget)
    RandomForest(trainX, trainY, testX, testY)

if __name__ == '__main__':
	main()