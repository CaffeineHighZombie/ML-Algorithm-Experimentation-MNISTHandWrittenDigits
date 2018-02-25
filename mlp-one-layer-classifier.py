import numpy as np
from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt
from sklearn import datasets, metrics
from sklearn import cross_validation
from sklearn.neural_network import MLPClassifier
import time 
from sklearn.utils import shuffle

mnist = fetch_mldata("MNIST original")
print(mnist.data.shape)
print(np.unique(mnist.target))
X, y = np.float32(mnist.data[:70000]), np.float32(mnist.target[:70000])
X, y = shuffle(X,y)
X_train, y_train = np.float32(X[:15000]), np.float32(y[:15000])
X_test, y_test = np.float32(X[60000:]), np.float32(y[60000:])
print (mnist.data.shape)
start = int(round(time.time() * 1000))
clf = MLPClassifier(solver='sgd', activation='tanh',alpha=1e-5,hidden_layer_sizes=(210), random_state=1)
clf.fit(X_train,y_train)
end = int(round(time.time() * 1000))
print("--Finished in ", (end-start), "ms--------------")
predicted=clf.predict(X_test)
expected=y_test
print(cross_validation.cross_val_score(clf, X_train,y_train, cv=5))
print("Classification report for One-hidden layer classifier %s:\n%s\n"
     % (clf, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
print("Accuracy is:",round(metrics.accuracy_score(expected,predicted)*100,2))
print("Test error is:",100-round(metrics.accuracy_score(expected,predicted)*100,2))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
