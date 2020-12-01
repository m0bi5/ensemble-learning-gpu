#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ECE5984 SP20 HW5  - 
Created on Thu Feb 20 17:41:33 2020
@author: ccody7
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier


# 2. Nomalize the dataset approapraiatly 
def NormalizeData(data):
    return( preprocessing.scale(data))
# 1. Load dataset and 2. Seperate target from feature values 3. Split Train and test Data
def loadData(pathName, dataFile, sheetName, IDNAME, TargetNAME):
	# 1. Load Dataset 
    dataFrame = pd.read_excel(pathName + dataFile, sheet_name=sheetName)
    # . Seperate Targets from features
    DataX = dataFrame.drop([IDNAME,TargetNAME], axis=1)
    DataY = dataFrame[TargetNAME]   
    # 2. Normalize Data
    DataX = NormalizeData(DataX)
    # 3. Split Train and test Data
    trainX,testX, trainY, testY = train_test_split(DataX, DataY, test_size = 0.3, random_state = 0)   
    
    return trainX, trainY, testX, testY

# 4.A -Train an MLPClassifier on the training data 
def Classifiers(trainX, trainY, testX, testY):
    regpenalty= 0.001
    active  = ['tanh','relu','identity']
    hiddenLayers=np.array([1,2,3])    
    HiddenNodes = np.arange(1, 21)  

    results = pd.DataFrame(columns =['MissclassificationRate', 'AUROC','Activation', 'Nodes', 'Layers'])

    for act in active:            
        for n_Layers in hiddenLayers:     
            for n_nodes in HiddenNodes:
                clf= MLPClassifier(hidden_layer_sizes=(n_Layers, n_nodes), activation=act, solver='adam', alpha=regpenalty, early_stopping=True, validation_fraction=0.42)
                clf.fit(trainX,trainY)
                ypred= clf.predict(testX)                
                MR,AUROC  = Measurements(testY,ypred)
                results=results.append({'MissclassificationRate': MR, 'AUROC':AUROC,'Activation': act, 'Nodes': n_nodes, 'Layers':n_Layers}, ignore_index=True)               
                   
    return results
    
#4.B- Measure the performance on the test set using two different measures: AUROC and misclassification rate.
def Measurements(ytest,ypred):    
    MR= misclassificationRateOut(ytest,ypred)     
    AUROC = Curves(ytest,ypred)    
    return(MR,AUROC)

#  Missclassification rate of each classifier 
def misclassificationRateOut(ytest,ypred):    
    #print("%s: Number of mislabeled points out of a total %d points : %d"% (Alg, Xtest.shape[0], (ytest!= ypred).sum()))
    MR = ((ytest!= ypred).sum()/ ytest.shape[0])    
    
    return(MR)   

# AUROC Curves
def Curves(ytest,ypred):
    fpr, tpr, thresholds = metrics.roc_curve(ytest,ypred)
    auroc= metrics.roc_auc_score(ytest,ypred)    
    return (auroc)


# 5. Build tables for the 10 best model architectures by AUROC and the 10 best Model Architectures by misclassification rate
def BuildTables(measures):
   #print ('The %s performance is %f' %(Algs, measures))
    BestClassification = measures.sort_values(by ='MissclassificationRate', ascending=True)
    BestAUROC =  measures.sort_values(by ='AUROC', ascending=False)
     
    print(BestClassification.head(10))
    print(BestAUROC.head(10))
    measures.to_csv('test.csv')

def main():
    pathName = "C:\\Users\\Cody\\OneDrive\\Documents\\VT\\ECE5984\\hw\\DATASETS\\"
    filename ='ccpp.xlsx'
    sheetName = 'allBin'
    IDNAME = 'ID'
    TargetNAME='TG'

    trainX, trainY, testX, testY = loadData(pathName, filename, sheetName, IDNAME, TargetNAME)
    
    results= Classifiers(trainX, trainY, testX, testY)
   
    BuildTables(results)
if __name__ == '__main__':
	main()