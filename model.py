# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 07:40:18 2020

@author: Suraz Bhatarai
"""


import numpy as np
from sklearn import preprocessing, model_selection, neighbors
import pandas as pd
import pickle
#import ctypes
#print (ctypes.sizeof(ctypes.c_voidp))
#import platform
#print (platform.architecture()) 
df = pd.read_csv('breast-cancer-dataset.data')
df.replace('?', -9999, inplace=True)
df.drop(['id'], 1, inplace=True) #id nachaiena vaeko le hataideko .outlier hunxa dataset ma so
print(df.head())
X = np.array(df.drop(['class'], 1)) #class is used as label(output) and all others are features representing X
y = np.array(df['class'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2) #test-size = 20%
clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)

#accuracy = clf.score(X_test, y_test)
#print(accuracy)
# Saving model to disk
pickle.dump(clf, open('model.pkl','wb'))


pickle_in = open('model.pkl','rb')
model = pickle.load(pickle_in)
#example_measures = np.array([[4, 2, 1, 1, 1, 2, 3, 2, 1],[4,2,1,2,2,2,3,2,1]]) #testing datas here
example_measures = np.array([[4, 2, 1, 1, 1, 2, 3, 2, 1]]) #testing datas here

example_measures = example_measures.reshape(len(example_measures), -1)
prediction = model.predict(example_measures)
print(prediction)