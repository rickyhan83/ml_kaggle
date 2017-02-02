import pandas as pd 
import numpy as np

import matplotlib.pyplot as plt

trains = pd.read_csv("train.csv")

Y = trains['label'] 

del trains['label']

X_datasarr = trains.as_matrix()
X_norm = X_datasarr > 0
X = X_norm.astype(int) 

ones = []
for i in range(len(Y)):
    if Y[i] == 4:
        ones.append(i)
fig = plt.figure()
    
for index in range(25):
    #fig, ax = plt.subplots()
    ax = fig.add_subplot(5,5,index+1)
    data = X[ones[index]]
    
    digit = np.zeros((28,28),dtype=int)
    xlist = []
    ylist = []
    
            
    col = 28
    row = 28
    for i in range(row):
        for j in range(col):    
            if data.item(i * 28 + j)==1:
                digit [i][j] = data.item(i * 28 + j)
    
    data = np.rot90(digit,k=3)
                
    for i in range(row):
        for j in range(col):
            if data.item(i * 28 + j)==1:
                xlist.append(i)
                ylist.append(j)
            
    ax.set_xlim([0,28])
    ax.set_ylim([0,28])
    plt.scatter(xlist,ylist, marker = 'x')    



    