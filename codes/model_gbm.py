#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 10:22:34 2019

@author: ankyriakide
"""
# https://machinelearningmastery.com/stochastic-gradient-boosting-xgboost-scikit-learn-python/
import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
#from matplotlib import pyplot
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef

# load data
data = pd.read_csv('modellingdata.csv')
dataset = data.values   #ignore headings
# split data into X and y and define features
X = dataset[:,3:]
y = dataset[:,1]
features = data.columns[3:]

# Parameter Selection 
# encode string class values as integers, i.e. let Rumour=1 NonRumour=0
label_encoded_y = LabelEncoder().fit_transform(y)

# Split to Train and Test Set 
X_train, X_test, y_train, y_test = train_test_split(X, label_encoded_y, test_size=0.3, random_state=123)

# Set the weights to re-balance the 0/1
w0=1          #non rumour weight 
w1=4023/2402  #rumour weight, penalises more the rumour faults. Note that if this is set to 1 you turn on default

sample_weights= np.zeros(len(y_train))
sample_weights[y_train== 0] = w0
sample_weights[y_train == 1] = w1

# Parameter selection via grid search - to know best hyperparams
model = GradientBoostingClassifier()
subsample = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
learning_rate = [0.05, 0.1, 0.15, 0.2, 0.25]
param_grid = dict(subsample=subsample, learning_rate=learning_rate)  #dictionary containing the values available for the hyperparameters
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=123)
grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=None, cv=kfold)
grid_result = grid_search.fit(X_train, y_train, sample_weights)

# Summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))      # brings back the average of the 10-fold CV, and returns the best of the minimised loss function  
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
	print("%f (%f) with: %r" % (mean, stdev, param))   # a list of the parameters and the corresponding means and std of the errors of the loss function 

for mean, stdev, param in zip(means, stds, params):
	print("%f, %f, %r" % (mean, stdev, param))   # a list of the parameters and the corresponding means and std of the errors of the loss function 



# Test Results
sample_weights= np.zeros(len(y_train))
sample_weights[y_train == 0] = w0
sample_weights[y_train == 1] = w1
# this is for final model, once hyperparam tunning is done
model_final=GradientBoostingClassifier(learning_rate=0.1, subsample=0.6)
model_result=model_final.fit(X_train, y_train, sample_weight=sample_weights)

y_true = y_test
y_pred = model_result.predict(X_test)
conf_mat = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = conf_mat.ravel()

# metrics
tnr = tn/(tn+fp)*100
fpr = fp/(fp+tn)*100
fnr = fn/(fn+tp)*100
tpr = tp/(tp+fn)*100

ppv = tp/(tp+fp)*100 #precision
acc = (tp+tn)/(tp+fn+tn+fp)*100  #accuracy 
f1 = (2*tp)/((2*tp)+fp+fn)*100  #f1

#mcc1 = (tp*tn)-(fp*fn)
#mcc2 = ((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))**0.5
#mcc = (mcc1/mcc2)*100  #matthews correlation coefficient

sample_weights_test= np.zeros(len(y_test))
sample_weights_test[y_test == 0] = w0
sample_weights_test[y_test == 1] = w1

mcc = matthews_corrcoef(y_true, y_pred,sample_weights_test)

print(round(tn), round(fp), round(fn), round(tp))
print(round(tnr), round(fpr), round(fnr), round(tpr))
print(round(ppv),round(acc),round(f1),round(mcc))


#confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Non Rumour','Rumour'],
                      title='Confusion matrix')
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print("True Negatives: ",tn)
print("False Positives: ",fp)
print("False Negatives: ",fn)
print("True Positives: ",tp)
plt.savefig("confmatrix_2label.png")


# determines the importance of each feature to the classification result
feats = {} # a dict to hold feature_name: feature_importance
for feature, importance in zip(features, model_result.feature_importances_):
    feats[feature] = importance #add the name/value pair 

feats


########

# Score M&A
# load data
data_ma = pd.read_csv('scoring_data.csv')
dataset = data_ma.values   #ignore headings
# split data into X and y and define features
X_ma = dataset[:,2:]
#X_ma = X_ma[~np.isnan(X_ma).any(axis=1)]

ma_results = model_final.predict(X_ma)
