#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 07:04:41 2021

@author: alexlange
"""

import numpy as np
import pandas as pd
import seaborn as sns
import ds6103 as dc
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report
#%%
data = pd.read_csv("/Users/alexlange/Downloads/train.csv")
test = pd.read_csv("/Users/alexlange/Downloads/test.csv")
#%%
def cleanDfIncome(row, colname): # colname can be 'rincome', 'income' etc
  thisamt = row[colname]
  if (thisamt == "neutral or dissatisfied"): return float(0)
  if (thisamt == "satisfied"): return float(1)
data['satisfaction'] = data.apply(cleanDfIncome, colname='satisfaction', axis=1)
#%%
clf1 = LogisticRegression()
clf2 = LinearSVC()
clf3 = SVC(kernel="linear")
clf4 = SVC()
clf5 = DecisionTreeClassifier()
clf6 = KNeighborsClassifier(n_neighbors=3) 
classifiers = [clf1,clf2] # use even numbers to avoid issue for now
# classifiers.append(clf3)
classifiers.append(clf4)
classifiers.append(clf5)
# classifiers.append(clf6)
# You can try adding clf3 and clf6, but KNN takes a long time to render.
xadmit = data[['Inflight wifi service', 'Food and drink', 'Seat comfort', 'Inflight entertainment', 'Inflight service']]
yadmit = data['satisfaction']
xadmit2 = data[['Inflight wifi service', 'Food and drink']]  # just two features so that we can plot 2-D easily
# yadmit = dfadmit['admit']
#%% about 10 min
# Fit the classifiers
for c in classifiers:
    c.fit(xadmit2,yadmit)

# Plot the classifiers
dc.plot_classifiers(xadmit2.values, yadmit.values, classifiers)
#%% about 5 min
X_train, X_test, y_train, y_test = train_test_split(xadmit, yadmit)
svc = SVC()
svc.fit(X_train,y_train)
#%%
print(f'svc train score:  {svc.score(X_train,y_train)}')
print(f'svc test score:  {svc.score(X_test,y_test)}')
#%%
print(confusion_matrix(y_test, svc.predict(X_test)))
print(classification_report(y_test, svc.predict(X_test)))



