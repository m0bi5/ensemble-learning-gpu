#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from timeit import default_timer as timer
import numpy as np
#import numba as nb
from random import seed
from random import randrange

class BaggingTreeClassifier(object):
    """
    Class to create a bagging treee classifier
    """
    def __init__(self, _max_depth, _min_size, _sample_size, _n_trees):
        """
        :param :
        :param :
        """
        self.max_depth = _max_depth
        self.min_size = _min_size
        self.sample_size = _sample_size
        self.n_trees = _n_trees

    def subsample(self, dataset):
        """
        :param :
        :param :
        :return:
        """   
        sample = list()
        n_nample = round(len(dataset) * self.sample_size) 
        while len(sample) < self.sample_size:
            index = randrange(len(dataset))
            sample.append(dataset[index])
        return np.array(sample)

    def fit(self, train ):
        """
        :param : 
        
        """

        trees = list()#np.empty(shape = 1)
        for t in range(self.n_trees):
            sample = self.subsample(train)
            tree = self.build_tree(train)
            trees.append(tree)
        return np.array(trees)
            

    def gini_index(self, groups ,classes):
        """
        :param :
        :param :
        :return :   
        """
        n_instances = float (sum(len(group) for group in groups))
        gini = 0.0
        for group in groups:
            size = float(len(group))

            if size == 0 :
                continue
            score = 0.0
            for class_val in classes: 
                p = [row[-1] for row in group].count(class_val) / size
                score += p * p
            
            gini += (1.0 - score) * (size / n_instances)  
        return gini


    def test_split(self, index, value, dataset):
        """
        :param :
        :param :
        :return:
        """
        left = list()
        right = list()
        for row in dataset:
            if row[index] < value:
                left.append(row)
            else:
                right.append(row)
        return np.array(left), np.array(right)

    def get_split(self, dataset):
        """
        :param : 
        :return:
        """
        #print (dataset.shape)
        class_values = np.array(dataset[:, -1])
        #print(class_values.shape)
        b_index, b_value, b_score, b_groups = 999, 999, 999, None
        for index in range(dataset.shape[1]-1):
            for row in dataset:
                groups = self.test_split(index, row[index], dataset)
                gini = self.gini_index(groups, class_values)
                if gini < b_score:
                    b_index, b_value, b_score, b_groups = index, row[index], gini, groups
        return {'index':b_index, 'value':b_value, 'groups':b_groups}

    def to_terminal_Node(self, group):
        """
        :param : 
        :return :
        """
        out = [row[-1] for row in group]
        return max(set(out), key = out.count)
    
    def split(self,  node, depth):
        """
        :param : 
        
        """
        left, right = node['groups']
        # free up node groups
        del (node['groups'])
        #print(left.size, right.size)
        # Check if there are no splits 
        if not left.size == 0 or not right.size == 0:
            node['left'] = node['right'] = self.to_terminal_Node(np.vstack((left , right)))
            return
        #  Check max depth
        if depth >= self.max_depth:
            node['left'], node['right'] = self.to_terminal_Node(left), self.to_terminal_Node(right)             

        # Left branch
        if len(left) <= self.min_size:
            node['left'] = self.to_terminal_Node(left)

        # Right Branch
        if len(right) <= self.min_size:
            node['right'] = self.to_terminal_Node(right)
        
        # The rest recursively             
        else:
            node['right'] = self.get_split(right)
            self.split(node['right'], depth)


    def build_tree(self, train):
        """
        :param :
        :param :
        :return: root
        """ 
        root = self.get_split(train)
        self.split(root,1)
        return root

    def tree_predict(self, node, row):
        if row[node['index']] < node['value']:
            if isinstance(node['left'], dict):
                return self.tree_predict(node['left'], row)
            else:
                return node['left']
        else: 
            if isinstance(node['right'], dict):
                return self.tree_predict(node['right'], row)
            else:
                return node['right']

    def bag_predict(self, trees,row):
        """
        :param :
        :return: tree_predictions
        """
        tree_predictions = [self.tree_predict(tree, row) for tree in trees ] 
        return max(set(tree_predictions), key = tree_predictions.count)

    def predict(self, trees, x_test):
        
        predictions = [self.bag_predict(trees, row) for row in x_test]
        return predictions

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def main():
    dataset = load_iris()  
    print (type(dataset))
    X = dataset.data
    y = dataset.target
    x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=0)
    #train , test =  train_test_split(dataset, test_size=0.2)
    print (x_train.shape, y_train.shape)
    #GPU 
    model = BaggingTreeClassifier(5, 2, 4, 5)
    s = timer()
    trees = model.fit(np.column_stack((x_train,  y_train)))
    e =  timer()
    print("Tree GPU Time: {0:1.6f}s ".format(e- s))
    print(accuracy_score(
        y_true=y_test,
        y_pred=model.predict(trees, x_test)
    ))
    
    #CPU
    #model = DecisionTree(2,5,False)
    #s =  timer()
    #model.fit(x_train, y_train)
    #e =  timer()
    #print("Tree CPU Time: {0:1.6f}s ".format(e- s))
    #print(accuracy_score(
    #    y_true=y_test,
    #    y_pred=model.predict(x_test)
    #))


if __name__ == '__main__':
	main()

