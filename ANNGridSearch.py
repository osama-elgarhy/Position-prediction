# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 13:39:37 2022

@author: oselga
"""

# multi-class classification with Keras
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



# Importing the dataset

dataset = pd.read_csv('data_col25.csv') 
#dataset = pd.read_csv('data_RTK_Field_25.csv') 


# the length of X and y is set according to the dataset 
#X = dataset.iloc[:, 0:5].values
#y = dataset.iloc[:, 5].values
#X = dataset.iloc[:, 0:10].values
#y = dataset.iloc[:, 10].values
X = dataset.iloc[:, 0:25].values
y = dataset.iloc[:, 25].values






# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X=sc.fit_transform(X)


###########
# upsampling

from sklearn import datasets
from sklearn.utils import shuffle

i_class0 = np.where(y == 0)[0]
i_class1 = np.where(y == 1)[0]

s_class0 = len(i_class0); 
#print(); print("s_class0: ", s_class0)
s_class1 = len(i_class1); print(); print("s_class1: ", s_class1)
i_class1_upsampled = np.random.choice(i_class1, size=s_class0, replace=True)
z=np.hstack((y[i_class1_upsampled], y[i_class0]))
xu=np.vstack((X[i_class1_upsampled,:], X[i_class0,:]))
y=z
X=xu
X, y = shuffle(X, y, random_state=0)
#print();

###########


#####################
# SMOTE (other upsmapling methods)
# uncomment a method from below and comment the above "upsampling" code
"""
#from imblearn.over_sampling import SMOTE
#sm = SMOTE(random_state=42)

#from imblearn.over_sampling import BorderlineSMOTE
#sm = BorderlineSMOTE(random_state=42) 

#from imblearn.under_sampling import NeighbourhoodCleaningRule
#sm = NeighbourhoodCleaningRule( n_neighbors=3, threshold_cleaning=0.) 

#from imblearn.over_sampling import ADASYN
#sm = ADASYN(random_state=42)

#from imblearn.over_sampling import SVMSMOTE
#sm = SVMSMOTE(random_state=42)

#from imblearn.under_sampling import TomekLinks
#sm = TomekLinks()

#from imblearn.under_sampling import CondensedNearestNeighbour
#sm = CondensedNearestNeighbour(n_neighbors=1)

#from imblearn.under_sampling import NearMiss
#sm = NearMiss(version=1, n_neighbors=3)


X, y  = sm.fit_resample(X, y)
"""
####################


from sklearn.model_selection import GridSearchCV

# define baseline model
def baseline_model(opti):
    classifier = Sequential()
    #classifier.add(Dense(units = 3, kernel_initializer = 'uniform', activation = 'relu', input_dim = 5))
    #classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 10))
    classifier.add(Dense(units = 13, kernel_initializer = 'uniform', activation = 'relu', input_dim = 25))
    #input_dim = 3
    #input_dim = 2 #for speed
    classifier.add(Dense(units = 2, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = opti, loss = 'binary_crossentropy', metrics = ['accuracy'])
    #classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['Precision'])
    return classifier
"""
estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=5, verbose=0)
kfold = KFold(n_splits=10, shuffle=True)
results = cross_val_score(estimator, X, y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
"""

classifier = KerasClassifier(build_fn = baseline_model)
parameters = {'batch_size': [25,5],
              'epochs': [500,1000,2000],
              'opti': [ 'adam', 'rmsprop']}

"""
# write as many parameters as needed to try (more parameters = more computing time)
parameters = {'batch_size': [25, 10, 5],
              'epochs': [500,1000,2000],
              'opti': [ 'Adamax', 'SGD', 'adam', 'rmsprop']}

"""

grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_searc = grid_search.fit(X, y)
best_parameters = grid_searc.best_params_
best_accuracy = grid_searc.best_score_

print(best_accuracy)
print(best_parameters)
cvrs=grid_searc.cv_results_
print(cvrs["mean_test_score"])
print("rank=",cvrs["rank_test_score"])
print("params=",cvrs["params"])

#import dill                            #pip install dill --user
#filename = 'globalsave.pkl'
#dill.dump_session(filename)


