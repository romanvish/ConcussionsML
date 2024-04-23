### In this script, we will hyperparameter tune our decision tree regression model to get a higher accuracy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import NearestCentroid
from sklearn.model_selection import train_test_split
from sklearn import metrics
from numpy import arange
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score



data = pd.read_csv('out3.csv')
X = data[['Games affected by season',
          'Number of Concussions Total',
          'Position_L',
          'Position_P',
          'Position_QB',
          'Position_RB/TE',
          'Position_WR/S']]
y = data[['Multiple Concussions in a Season?']]
y = np.array(y)
y = y.ravel()

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.1,random_state =64)

model = NearestCentroid(metric='euclidean',shrink_threshold=3.59)
'''cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid = dict()
grid['shrink_threshold'] = arange(0, 10, 0.01)
grid['metric'] = ['euclidean', 'manhattan']
search = GridSearchCV(model, grid, scoring='accuracy', cv=cv, n_jobs=-1)
results = search.fit(X, y)
print('Mean Accuracy: %.3f' % results.best_score_)
print('Config: %s' % results.best_params_)'''
model.fit(X_train,y_train)
prediction=model.predict(X_test)
print('Accuracy :'+str(accuracy_score(y_test,prediction)))