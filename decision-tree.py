import numpy as np
from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt
from sklearn import datasets, metrics
from sklearn import cross_validation
from sklearn import tree
import time 
from sklearn.utils import shuffle

mnist = fetch_mldata("MNIST original")
print(mnist.data.shape)
print(np.unique(mnist.target))
X, y = np.float32(mnist.data[:70000]), np.float32(mnist.target[:70000])
X, y = shuffle(X,y)
X_train, y_train = np.float32(X[:15000]), np.float32(y[:15000])
X_test, y_test = np.float32(X[60000:]), np.float32(y[60000:])
start = int(round(time.time() * 1000))
DT=tree.DecisionTreeClassifier(random_state=1)
DT.fit(X_train,y_train)
predicted=DT.predict(X_test)
expected=y_test
print(cross_validation.cross_val_score(DT, X_train,y_train, cv=5))
end = int(round(time.time() * 1000))
print("--Finished in ", (end-start), "ms--------------")
print("Classification report  %s:\n%s\n"
     % (DT, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
print("Accuracy is:",round(metrics.accuracy_score(expected,predicted)*100,2))
print("Test error is:",100-round(metrics.accuracy_score(expected,predicted)*100,2))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))