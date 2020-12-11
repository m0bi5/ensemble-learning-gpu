from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import numpy as np

def LoadData():
    dataset = load_wine()
    X = dataset.data
    y = dataset.target
    x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=0)
    classes = np.unique(y)