#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Clustering
"""
from sklearn import neighbors
import pandas as pd
import numpy as np
import math



def distance(inA, inB, distSelector, mu, co):
    a = np.asarray(inA)
    b = np.asarray(inB)
    
    if(distSelector == 'Manhattan'):
        #<<< your code here >>> # Manhattan
        answer = np.sum(np.fabs(np.subtract(a,b)))
        # manhattan is the sum of the absoute diffrences 
        return answer
    elif(distSelector == 'Mahalanobis'):
        #<<< your code here >>> # Mahalanobis
       
        amu = np.subtract(a, mu)
        bmu =  np.subtract(b, mu)
        inn = np.array(amu- bmu)
        
        inCov= np.linalg.inv(co)
        left = np.dot(inn.T,inCov)
        answer= np.sqrt(np.dot(left, inn))
       
        # Mahalanobis
        return answer
    else: #default is Euclidean
        answer = math.sqrt(np.dot(a-b, a-b))
        #print (answer)
        return answer
        # Eucl is the square root of the sum of the squares of the differences

def makeTable(minRow, minDist, minTarget):
    print("Min at", minRow)
    print(minDist, minTarget)
    
def Training(trainX, trainY, newX ):
    
    minDist = 100000
    count = 0
    mu = trainX.mean() # for Mahalanobis
    co = trainX.cov() # for Mahalanobis
   
    for row in trainX.iterrows(): # NOTE! this is slow and only for use on small ADS
        samp = row[1]
        #dist = distance(np.transpose(np.asarray(newX)), samp, "Euclidean", mu, co)
        #dist = distance(np.transpose(np.asarray(newX)), samp, "Manhattan", mu, co)
        dist = distance(np.transpose(np.asarray(newX)), samp, "Mahalanobis", mu, co)
                
        
        if (dist < minDist): # find the smallest distance (most similar)
            minDist = dist
            minRow = row
            minTarget = trainY[count]
        count = count + 1
        
    makeTable(minRow, minDist, minTarget)
  
 # Can Be cchanged to accomidate diffrent datadset 
def getData():
    pathName = "C:\\Users\\Cody\\OneDrive\\Documents\\VT\\ECE5984\\hw\\DATASETS\\"

    dataFrame = pd.read_excel(pathName + 'iris.xlsx', sheet_name='data')
    trainX = dataFrame.drop(["species"], axis=1)
    trainY = dataFrame.species
    return trainX, trainY
  
def main():
    newX = [5.5, 3.0, 4.4, 1.4]
    newX = [5.15, 3.25, 2.9, 0.9]
    newX = [6.7, 3.1, 5.15, 1.95]
    
    #pathName = "C:\\Data\\"
    trainX, trainY =  getData()
    
    Training(trainX, trainY, newX )
    #plot(trainX,trainY)
    

   
    
if __name__ == '__main__':
	main()
