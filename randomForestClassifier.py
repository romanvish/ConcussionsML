### In this script, we will hyperparameter tune our random forest regression model to get a higher accuracy

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import warnings
import numpy as np
 
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
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
model=RandomForestClassifier(bootstrap=True,
                             max_depth=10,
                             max_features='sqrt',
                             min_samples_leaf=2,
                             min_samples_split=10,
                             n_estimators=200)

'''n_estimators = [100, 200, 300, 400, 500]
max_features = ['auto', 'sqrt']
max_depth = [10,50,110]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

tuning_model=GridSearchCV(model,param_grid=random_grid,scoring='neg_mean_squared_error',cv=3,verbose=3)
tuning_model.fit(X,y)
print(tuning_model.best_params_)'''

model.fit(X_train,y_train)
prediction=model.predict(X_test)

print('Accuracy :'+str(accuracy_score(y_test,prediction)))
print('Confusion Matrix:') 
print(metrics.confusion_matrix(y_test, prediction))
print('Classification Report:')
print(metrics.classification_report(y_test, prediction))
print("Importances")
print(model.feature_importances_)
matrix = metrics.confusion_matrix(y_test, prediction)
matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

# Build the plot
plt.figure(figsize=(10,5))
sns.set(font_scale=.7)
sns.heatmap(matrix, annot=True, annot_kws={'size':7},
            cmap=plt.cm.Blues, linewidths=0.2)

# Add labels to the plot
class_names = ['Low Reinjury risk', 'High Reinjury risk']
tick_marks = np.arange(len(class_names))
tick_marks2 = tick_marks + 0.5
plt.xticks(tick_marks, class_names, rotation=25)
plt.yticks(tick_marks2, class_names, rotation=0)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix for Random Forest Model')
plt.show()