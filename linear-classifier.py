from sklearn.datasets import load_digits
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split
from tensorflow.contrib import learn
import tensorflow as tf
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn import metrics
from sklearn.utils import shuffle
import time
m_data = tf.contrib.learn.datasets.mnist.load_mnist()
train_set = m_data.train
train_images = train_set.images
train_labels = train_set.labels.astype(np.int32)
val_set = m_data.validation
val_images = val_set.images
val_labels = val_set.labels.astype(np.int32)
test_set = m_data.test
test_images = test_set.images
test_labels = test_set.labels.astype(np.int32)
image_set = tf.contrib.layers.real_valued_column('', dimension=784)
cls = learn.LinearClassifier(feature_columns=[image_set], n_classes=10)
start = int(round(time.time() * 1000))
cls.fit(train_images, train_labels , steps=2000)
end = int(round(time.time() * 1000))
print("--NN fitting finished in ", (end-start), "ms--------------")
val_pred = list(cls.predict(val_images, as_iterable=True))
score = metrics.accuracy_score(val_labels, val_pred)
print ('Test Accuracy', score)
Recall = metrics.recall_score(val_labels, val_pred, average='weighted')
print('Test Recall', Recall)
Precision = metrics.precision_score(val_labels, val_pred, average='weighted')
print('Test Precision', Precision)
F1 = metrics.f1_score(val_labels, val_pred, average='weighted')
print('F1', F1)
matrix = metrics.confusion_matrix(val_labels, val_pred)
print('Confusion Matrix', matrix)
print("Classification report for Linear classifier %s:\n%s\n"
     % (cls, metrics.classification_report(val_labels, val_pred)))