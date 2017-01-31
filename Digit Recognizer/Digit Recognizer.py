# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 11:12:32 2017

@author: newton
"""

import pandas as pd 
import scipy as sp
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import csv 

trains = pd.read_csv("train.csv")
tests = pd.read_csv("test.csv")

print trains.head()

labels = trains['label'] 

del trains['label']

X_train_datasarr = trains.as_matrix()
#X_train_datasarr = np.array(trains)

X_train_norm = X_train_datasarr > 0
X_train = X_train_norm.astype(int) 

X_test_datasarr = tests.as_matrix()
X_test_norm = X_test_datasarr > 0
X_test = X_test_norm.astype(int) 

rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train,labels)

Y_test = rfc.predict(X_test)

print X_train[:10],len(Y_test)
headers = ['ImageId','Label']

with open('digit_submission.csv','w') as f:
    f_csv = csv.writer(f)
    f_csv.writerow(headers)
    rowid = 1
    for y in Y_test:
        row = [rowid,y]
        rowid += 1
        f_csv.writerow(row)


