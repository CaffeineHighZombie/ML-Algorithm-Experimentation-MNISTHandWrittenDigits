import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt
from sklearn import datasets, metrics
from sklearn import cross_validation
import time 
from sklearn import neighbors
from sklearn.utils import shuffle

tf.logging.set_verbosity(tf.logging.ERROR)
mnist = fetch_mldata("MNIST original")
print(mnist.data.shape)
print(np.unique(mnist.target))
X, y = np.float32(mnist.data[:70000]), np.float32(mnist.target[:70000])
X, y = shuffle(X,y)
X_train, y_train = np.float32(X[:15000]), np.float32(y[:15000])
X_test, y_test = np.float32(X[60000:]), np.float32(y[60000:])
start = int(round(time.time() * 1000))
k = np.arange(5,100,10)
acc = []
for i in k:
    clf = neighbors.KNeighborsClassifier(n_neighbors=i)
    clf.fit(X_train, y_train)
    end = int(round(time.time() * 1000))
    print("KNN fitting finished in ", (end-start), "ms")
    print("--------- Cross validation accuracy--------")
    scores = cross_validation.cross_val_score(clf, X_train,y_train, cv=5)
    print('cross validation with k value', i)
    print ('Cross validation', scores)
    acc.append(scores.mean())
    print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
