# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 11:12:32 2017

@author: newton
"""

import pandas as pd 
import scipy as sp
from sklearn.ensemble import RandomForestClassifier
import csv 

num_clusters = 3
seed = 2
sp.random.seed(seed) 

datas = pd.read_csv("train.csv")
tests = pd.read_csv("test.csv")

labels = datas['label'] 

del datas['label']

rfc = RandomForestClassifier(n_estimators=100)

rfc.fit(datas.as_matrix(),labels)
y_results = rfc.predict(tests.as_matrix())

print y,len(y_results)
headers = ['ImageId','Label']

with open('digit_submission.csv','w') as f:
    f_csv = csv.writer(f)
    f_csv.writerow(headers)
    rowid = 1
    for y in y_results:
        row = [rowid,y]
        rowid += 1
        f_csv.writerow(row)
