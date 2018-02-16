import numpy as np
from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn import cross_validation
import time 
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression

## LibLinear Solver and 10 iteration
mnist = fetch_mldata("MNIST original")
print(mnist.data.shape)
print(np.unique(mnist.target))
X, y = np.float32(mnist.data[:70000]), np.float32(mnist.target[:70000])
X, y = shuffle(X,y)
X_train, y_train = np.float32(X[:15000]), np.float32(y[:15000])
X_test, y_test = np.float32(X[60000:]), np.float32(y[60000:])
start = int(round(time.time() * 1000))
LR = LogisticRegression(solver='liblinear',max_iter=10)
LR.fit(X_train,y_train)
end = int(round(time.time() * 1000))
print("--Finished in ", (end-start), "ms--------------")
predicted=LR.predict(X_test)
expected=y_test
print(cross_validation.cross_val_score(LR, X_train,y_train, cv=5))
print("Classification report %s:\n%s\n"
     % (LR, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
print("Accuracy is:",round(metrics.accuracy_score(expected,predicted)*100,2))
print("Test error is:",100-round(metrics.accuracy_score(expected,predicted)*100,2))

## LibLinear Solver and 20 iteration
mnist = fetch_mldata("MNIST original")
print(mnist.data.shape)
print(np.unique(mnist.target))
X, y = np.float32(mnist.data[:70000]), np.float32(mnist.target[:70000])
X, y = shuffle(X,y)
X_train, y_train = np.float32(X[:15000]), np.float32(y[:15000])
X_test, y_test = np.float32(X[60000:]), np.float32(y[60000:])
start = int(round(time.time() * 1000))
LR = LogisticRegression(solver='liblinear',max_iter=20)
LR.fit(X_train,y_train)
end = int(round(time.time() * 1000))
print("--Finished in ", (end-start), "ms--------------")
predicted=LR.predict(X_test)
expected=y_test
print(cross_validation.cross_val_score(LR, X_train,y_train, cv=5))
print("Classification report %s:\n%s\n"
     % (LR, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
print("Accuracy is:",round(metrics.accuracy_score(expected,predicted)*100,2))
print("Test error is:",100-round(metrics.accuracy_score(expected,predicted)*100,2))

## newton-cg Solver and 10 iteration
mnist = fetch_mldata("MNIST original")
print(mnist.data.shape)
print(np.unique(mnist.target))
X, y = np.float32(mnist.data[:70000]), np.float32(mnist.target[:70000])
X, y = shuffle(X,y)
X_train, y_train = np.float32(X[:15000]), np.float32(y[:15000])
X_test, y_test = np.float32(X[60000:]), np.float32(y[60000:])
start = int(round(time.time() * 1000))
LR = LogisticRegression(solver='newton-cg',max_iter=10)
LR.fit(X_train,y_train)
end = int(round(time.time() * 1000))
print("--Finished in ", (end-start), "ms--------------")
predicted=LR.predict(X_test)
expected=y_test
print(cross_validation.cross_val_score(LR, X_train,y_train, cv=5))
print("Classification report %s:\n%s\n"
     % (LR, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
print("Accuracy is:",round(metrics.accuracy_score(expected,predicted)*100,2))
print("Test error is:",100-round(metrics.accuracy_score(expected,predicted)*100,2))

## newton-cg Solver and 20 iteration
mnist = fetch_mldata("MNIST original")
print(mnist.data.shape)
print(np.unique(mnist.target))
X, y = np.float32(mnist.data[:70000]), np.float32(mnist.target[:70000])
X, y = shuffle(X,y)
X_train, y_train = np.float32(X[:15000]), np.float32(y[:15000])
X_test, y_test = np.float32(X[60000:]), np.float32(y[60000:])
start = int(round(time.time() * 1000))
LR = LogisticRegression(solver='newton-cg',max_iter=20)
LR.fit(X_train,y_train)
end = int(round(time.time() * 1000))
print("--Finished in ", (end-start), "ms--------------")
predicted=LR.predict(X_test)
expected=y_test
print(cross_validation.cross_val_score(LR, X_train,y_train, cv=5))
print("Classification report %s:\n%s\n"
     % (LR, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
print("Accuracy is:",round(metrics.accuracy_score(expected,predicted)*100,2))
print("Test error is:",100-round(metrics.accuracy_score(expected,predicted)*100,2))