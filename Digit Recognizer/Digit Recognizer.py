# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 11:12:32 2017

@author: newton
"""

import pandas as pd 
import scipy as sp
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn import metrics
from sklearn.metrics import confusion_matrix 
import csv 
import matplotlib.pyplot as plt
import itertools

labels = [0,1,2,3,4,5,6,7,8,9]

#def plot_confusion_matrix(cm, title='Confusion Matrix', cmap = plt.cm.binary):  
#    plt.imshow(cm, interpolation='nearest', cmap=cmap)  
#    plt.title(title)  
#    plt.colorbar()  
#    xlocations = np.array(range(len(labels)))  
#    plt.xticks(xlocations, labels, rotation=90)  
#    plt.yticks(xlocations, labels)  
#    plt.ylabel('True label')  
#    plt.xlabel('Predicted label')  

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
trains = pd.read_csv("train.csv")
tests = pd.read_csv("test.csv")

Y = trains['label'] 

del trains['label']

X_datasarr = trains.as_matrix()
X_norm = X_datasarr > 0
X = X_norm.astype(int) 


X_train,X_test,y_train,y_test = cross_validation.train_test_split(X, Y, test_size=0.3, random_state=0)

unique, counts = np.unique(y_test, return_counts=True)
print np.asarray((unique, counts)).T

X_des_datasarr = tests.as_matrix()
X_des_norm = X_des_datasarr > 0
X_des = X_des_norm.astype(int) 

results = []
# 最小叶子结点的参数取值
sample_leaf_options = list(range(1, 50, 3))
# 决策树个数参数取值
n_estimators_options = list(range(1, 10, 5))
for leaf_size in sample_leaf_options:
    for n_estimators_size in n_estimators_options:
        
        rfc = RandomForestClassifier(min_samples_leaf=leaf_size, n_estimators=n_estimators_size, random_state=50)
        rfc.fit(X_train,y_train)

        y_pred_class = rfc.predict(X_test)
        results.append((leaf_size, n_estimators_size, (y_test == y_pred_class).mean()))        

print(max(results, key=lambda x: x[2]))

rfc = RandomForestClassifier(min_samples_leaf=6, n_estimators=6, random_state=50)
rfc.fit(X_train,y_train)
cm = confusion_matrix(y_test, y_pred_class)
plot_confusion_matrix(cm,labels, title='Normalized confusion matrix') 

        
#print metrics.accuracy_score(y_test, y_pred_class)
#print rfc.score(X_test, y_test)
#print rfc.classes_
#print y_test[0]
#print rfc.predict(X_test[0])
#print rfc.predict_proba(X_test[0])

Y_des = rfc.predict(X_des)

#Data is not binary and pos_label is not specified
#precision, recall, pr_thresholds = precision_recall_curve(y_test, y_pred_class)
#print precision,recall,pr_thresholds

headers = ['ImageId','Label']

with open('digit_submission.csv','w') as f:
    f_csv = csv.writer(f)
    f_csv.writerow(headers)
    rowid = 1
    for y in Y_des:
        row = [rowid,y]
        rowid += 1
        f_csv.writerow(row)


