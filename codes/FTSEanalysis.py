#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 23:14:55 2019

@author: ankyriakide
"""
#FTSE analysis

import numpy as np
import pandas as pd

from pydoc import help
from scipy.stats.stats import pearsonr

import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests


### 2Label 
#FTSE100 Closing Price VS Proportion
data1 = pd.read_csv('FTSEcloseprice_FN_2label.csv')

#Granger Causality Test   https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.grangercausalitytests.html
granger_test_result_1 = grangercausalitytests(data1, maxlag=7, verbose=True)
#Pearson Correlation Coefficient  https://stackoverflow.com/questions/3949226/calculating-pearson-correlation-and-significance-in-python
X1 = data1.iloc[:,0].values
Y1 = data1.iloc[:,1].values
pearsonr(X1,Y1)


#FTSE Volume VS Proportion
data2 = pd.read_csv('FTSE%volume_FN_2label.csv')

#Granger Causality Test   https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.grangercausalitytests.html
granger_test_result_2 = grangercausalitytests(data2, maxlag=7, verbose=True)
#Pearson Correlation Coefficient  https://stackoverflow.com/questions/3949226/calculating-pearson-correlation-and-significance-in-python
X2 = data1.iloc[:,0].values
Y2 = data1.iloc[:,1].values
pearsonr(X2,Y2)



### 3Label 
#FTSE100 Closing Price VS Proportion
data11 = pd.read_csv('FTSEcloseprice_FN_3label.csv')

#Granger Causality Test   https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.grangercausalitytests.html
granger_test_result_11 = grangercausalitytests(data11, maxlag=7, verbose=True)
#Pearson Correlation Coefficient  https://stackoverflow.com/questions/3949226/calculating-pearson-correlation-and-significance-in-python
X11 = data1.iloc[:,0].values
Y11 = data1.iloc[:,1].values
pearsonr(X11,Y11)


#FTSE Volume VS Proportion
data22 = pd.read_csv('FTSE%volume_FN_3label.csv')

#Granger Causality Test   https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.grangercausalitytests.html
granger_test_result_22 = grangercausalitytests(data22, maxlag=7, verbose=True)
#Pearson Correlation Coefficient  https://stackoverflow.com/questions/3949226/calculating-pearson-correlation-and-significance-in-python
X22 = data1.iloc[:,0].values
Y22 = data1.iloc[:,1].values
pearsonr(X22,Y22)

