#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
ECE5984 SP20 HW3 Part 2 - wifi signal strength
Created on 
@author: ccody7
"""
from sklearn import neighbors
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def loadData(pathName):
    dataFrameTrain = pd.read_excel(pathName + 'wifi.xlsx', sheet_name='train')
    dataFrameTest = pd.read_excel(pathName + 'wifi.xlsx', sheet_name='test')
    trainX = dataFrameTrain.drop(["level"], axis=1)
    trainY = dataFrameTrain.level
    testX = dataFrameTest.drop(["level"], axis=1)
    testY = dataFrameTest.level
    return trainX, trainY, testX, testY

def KNeighborsClassifier(trainX, trainY,  k, weight,met):
    clf= neighbors.KNeighborsClassifier(n_neighbors= k, weights=weight, metric=met)
    clf.fit(trainX,trainY)
    return clf 

def ErrorRate(predict, actual): 
    
    num_test = len(predict)    
    confusion = confusion_matrix(actual, predict)    
    err_rate= (num_test - confusion.trace())/confusion.sum()   
    return err_rate, confusion
    
def Test(clf, testX):
    results = clf.predict(testX)
    return results    

def plotErrors(kValue,errorsUW,errorsWE, errorsWM ):
    plt.plot(kValue,errorsUW, label= 'Unweighted')
    plt.plot(kValue,errorsWE, label= 'Weighted Eucledian disstance')
    plt.plot(kValue,errorsWM, label= 'Weighted Manhattan disstance')
    plt.xlabel("Number of Neighbors K")
    plt.ylabel("Misclassification Error")
    plt.legend()
    plt.show()

def PrettyPrint(kValue,errors,confusions):
    
    print ('k =',kValue[0] ,' Error Rate = ' ,errors[0], 'confusion Matrix = ' )
    print(confusions[0])
    
    print ('k =',kValue[4] ,' Error Rate = ' ,errors[4], 'confusion Matrix = ' )
    print(confusions[4])
    
    print ('k =',kValue[24] ,' Error Rate = ' ,errors[24], 'confusion Matrix = ' )
    print(confusions[24])
    

def main():
    pathName = "C:\\Users\\Cody\\OneDrive\\Documents\\VT\\ECE5984\\hw\\DATASETS\\"
    trainX, trainY, testX, testY = loadData(pathName)
    kValue = range(1,26)
    error_rateUW=[]
    confusion_matsUW=[]
    error_rateWE=[]
    confusion_matsWE=[]
    error_rateWM=[]
    confusion_matsWM=[]
    
    unweighted = 'uniform'
    weighted = 'distance'
    metricEu = 'euclidean'
    metricM = 'manhattan'
    for k in kValue:
        clfUW = KNeighborsClassifier(trainX, trainY,  k, unweighted, metricEu)
        neighbors
        clfWE = KNeighborsClassifier(trainX, trainY,  k, weighted, metricEu)
        clfWM = KNeighborsClassifier(trainX, trainY,  k, weighted, metricM)
        
        predictedUW = Test(clfUW, testX)
        predictedWE = Test(clfWE, testX)                  
        predictedWM = Test(clfWM, testX) 
        
        err_rateUW, confusionUW = ErrorRate(predictedUW, testY)
        err_rateWE, confusionWE = ErrorRate(predictedWE, testY)
        err_rateWM, confusionWM = ErrorRate(predictedWM, testY)
        
        error_rateUW.append(err_rateUW)
        confusion_matsUW.append(confusionUW)
        
        error_rateWE.append(err_rateWE)
        confusion_matsWE.append(confusionWE)
        
        error_rateWM.append(err_rateWM)
        confusion_matsWM.append(confusionWM)        
       
    plotErrors(kValue,error_rateUW,error_rateWE,error_rateWM )
    
    print ("Unweighted: ")
    PrettyPrint(kValue,error_rateUW,confusion_matsUW)
    
    print ("Weighted Eucledian distance: ")
    PrettyPrint(kValue,error_rateWE,confusion_matsWE)
    
    print ("Weighted Manhattan distance: ")
    PrettyPrint(kValue,error_rateWM,confusion_matsWM)
   
    
if __name__ == '__main__':
	main()
