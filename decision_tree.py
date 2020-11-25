"""This module includes methods for training and predicting using decision trees."""
import numpy as np
import pandas as pd
import scipy.stats 
from scipy.stats import norm
class Node: 
    def __init__(self,key): 
        self.left = None
        self.right = None
        self.label = None 
        self.test=None
        self.leaf= False

#def tree_split(arr, max_gain):



def calculate_information_gain(data, labels):
    """
    Computes the information gain on label probability for each feature in data

    :param data: d x n matrix of d features and n examples
    :type data: ndarray
    :param labels: n x 1 vector of class labels for n examples
    :type labels: array
    :return: d x 1 vector of information gain for each feature (H(y) - H(y|x_d))
    :rtype: array
    """
    all_labels = np.unique(labels)
    num_classes = len(all_labels)

    class_count = np.zeros(num_classes)

    d, n = data.shape
    
    full_entropy = 0
    for c in range(num_classes):
        class_count[c] = np.sum(labels == all_labels[c])
        if class_count[c] > 0:
            class_prob = class_count[c] / n
            full_entropy -= class_prob * np.log(class_prob)
           

    #print("Full entropy is %d\n" % full_entropy)
    
    gain = full_entropy * np.ones(d)
    
    # we use a matrix dot product to sum to make it more compatible with sparse matrices
    num_x = data.dot(np.ones(n))
    prob_x = num_x / n
    prob_not_x = 1 - prob_x
    
    for c in range(num_classes):
        # print("Computing contribution of class %d." % c)
        num_y = np.sum(labels == all_labels[c])
        # this next line sums across the rows of data, multiplied by the
        # indicator of whether each column's label is c. It counts the number
        # of times each feature is on among examples with label c.
        # We again use the dot product for sparse-matrix compatibility
        data_with_label = data[:, labels == all_labels[c]]
        num_y_and_x = data_with_label.dot(np.ones(data_with_label.shape[1]))
        
        # Prevents Python from outputting a divide-by-zero warning
        with np.errstate(invalid='ignore'):
            prob_y_given_x = num_y_and_x / (num_x + 1e-8)
        prob_y_given_x[num_x == 0] = 0
       
        nonzero_entries = prob_y_given_x > 0
        if np.any(nonzero_entries):
            with np.errstate(invalid='ignore', divide='ignore'):
                cond_entropy = - np.multiply(np.multiply(prob_x, prob_y_given_x), np.log(prob_y_given_x))
            gain[nonzero_entries] -= cond_entropy[nonzero_entries]

        # The next lines compute the probability of y being c given x = 0 by
        # subtracting the quantities we've already counted
        # num_y - num_y_and_x is the number of examples with label y that
        # don't have each feature, and n - num_x is the number of examples
        # that don't have each feature
        with np.errstate(invalid='ignore'):
            prob_y_given_not_x = (num_y - num_y_and_x) / ((n - num_x) + 1e-8)
        prob_y_given_not_x[n - num_x == 0] = 0
          
        
        
        nonzero_entries = prob_y_given_not_x > 0
        if np.any(nonzero_entries):
            with np.errstate(invalid='ignore', divide='ignore'):
                cond_entropy = - np.multiply(np.multiply(prob_not_x, prob_y_given_not_x), np.log(prob_y_given_not_x))
            gain[nonzero_entries] -= cond_entropy[nonzero_entries]
     #   print(gain)
    #print (gain)
    return gain


def decision_tree_train(train_data, train_labels, params):
    """Train a decision tree to classify data using the entropy decision criterion.

    :param train_data: d x n numpy matrix (ndarray) of d binary features for n examples
    :type train_data: ndarray
    :param train_labels: length n numpy vector with integer labels
    :type train_labels: array_like
    :param params: learning algorithm parameter dictionary. Must include a 'max_depth' value
    :type params: dict
    :return: dictionary encoding the learned decision tree
    :rtype: dict
    """
    max_depth = params['max_depth']

    labels = np.unique(train_labels)
    num_classes = labels.size

    model = recursive_tree_train(train_data, train_labels, depth=0, max_depth=max_depth, num_classes=num_classes)
    return model


def recursive_tree_train(data, labels, depth, max_depth, num_classes):
    """Helper function to recursively build a decision tree by splitting the data by a feature.

    :param data: d x n numpy matrix (ndarray) of d binary features for n examples
    :type data: ndarray
    :param labels: length n numpy array with integer labels
    :type labels: array_like
    :param depth: current depth of the decision tree node being constructed
    :type depth: int
    :param max_depth: maximum depth to expand the decision tree to
    :type max_depth: int
    :param num_classes: number of classes in the classification problem
    :type num_classes: int
    :return: dictionary encoding the learned decision tree node
    :rtype: dict
    """
    # TODO: INSERT YOUR CODE FOR LEARNING THE DECISION TREE STRUCTURE HERE
    d, n = data.shape
    classes = np.unique(labels)
    #num_classes = classes.size
    # If Data is all in one class or the max depth is reached 
    #node = {}
    #node['classes']= classes
    
    node = Node
    if (num_classes <2 or  max_depth<= depth):
        #print(num_classes)
        #print(depth)
        #node.prection = most common class in D
        #node['test']=None
        #node['left']=None
        #node['right'] = None
        #node['label']=classes
        #node ['leaf']= True
        #node.right=None
        #node.left= None
        #node.test=None
        #print(classes)
        node.label=classes
        node.leaf = True
        #print (node.label)
        return node
    # rule = bestDecisionRule(D)
    rule = calculate_information_gain(data, labels)
    #print(rule)
    #print (rule.size)
    #print(d)
    # This is the feature column with the max gain that will serve as the split
    max_gain = np.argmax(rule)
    #print(max_gain)
    left_index=np.where(data[max_gain,:]==0)[0]
    right_index=np.where(data[max_gain,:]!=0)[0]
    #print(data[max_gain,:]==0)
   
    
    # dataLeft = {(x,y) from D where rule(D) is true }
    # dataLeft = data, labels, depth, max_depth, num_classes
    labelLeft = labels[left_index]
    dataLeft = data[:,left_index]
    num_classesLeft=np.unique(labelLeft).size
    # dataRight {(x,y) from D where rule(D) is false }
    labelRight= labels[right_index]    
    #print (dataLeft.shape)
    dataRight = data[:,right_index]   
    #print(dataRight.shape)
    num_classesRight=np.unique(labelRight).size
    #node['label']=None
    #node['test']= max_gain
    #node['leaf'] = False
    #node['left'] =  recursive_tree_train(dataLeft, labelLeft, depth+1, max_depth, num_classesLeft)
    #node['right'] =  recursive_tree_train(dataRight, labelRight, depth+1, max_depth, num_classesRight)
    
    

    node.test=max_gain
    node.left =  recursive_tree_train(dataLeft, labelLeft, depth+1, max_depth, num_classesLeft)
    node.right =  recursive_tree_train(dataRight, labelRight, depth+1, max_depth, num_classesRight)
    
    return node


def decision_tree_predict(data, model):
    """Predict most likely label given computed decision tree in model.

    :param data: d x n ndarray of d binary features for n examples.
    :type data: ndarray
    :param model: learned decision tree model
    :type model: dict
    :return: length n numpy array of the predicted class labels
    :rtype: array_like
    """
    # TODO: INSERT YOUR CODE FOR COMPUTING THE DECISION TREE PREDICTIONS HERE
    d, n = data.shape
    labels= np.empty(n)
    #print(list(model))
    #print(model['right'])
    #print(model['left'])
    #print(model['label'])
    #print(model['test'])
 
     
    #print(len(model['label']))
    # test for leaf
    if model.leaf:
        if model.label.size<1: #terminating node leaf
           
            #labels.fill(model['label'][0])
            #print(model['classes'])
            return labels
        else: # max depth leaf
            labels.fill(np.argmax(model.label))  
            return labels
    #ttest if max depth position occured
    # Test for root 
    #if model['label'] == None:
        #rightLabel= decision_tree_predict(data, model['right'])
        
   
    if model.leaf:
        max_gain = model.test
        left_index=np.where(data[max_gain,:]==0)[0]
        right_index=np.where(data[max_gain,:]!=0)[0]
        #print(np.where(data[max_gain,:]!=0)[0])
        #print(np.where(data[max_gain,:]==0)[0])
        dataLeft = data[:,left_index]
        dataRight = data[:,right_index]   
    
      
        if  model.right == None:
            return decision_tree_predict(dataLeft, model.left)
        
        if   model.left== None:
            return decision_tree_predict(dataRight, model.right)

        
        rightLabel= decision_tree_predict(dataRight, model.right)
        leftLabel = decision_tree_predict(dataLeft, model.left)
        labels= np.append( leftLabel,rightLabel)
    
#check for the root node
    #if model.label== None and model.leaf== False and model.test ==None:
        #print(model)
        #return labels#leftLabel = decision_tree_predict(data, model['left'])

    return labels
