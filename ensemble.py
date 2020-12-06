from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
import time
digits = load_iris()
X = digits.data
y = digits.target
x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=0)

model = DecisionTree(_max_depth = 2, _min_splits = 5)

s = time.time_ns()
model.fit(x_train, y_train)

print(accuracy_score(
    y_true=y_test,
    y_pred=model.predict(x_test)
))
e = time.time_ns()
print(e-s)
 