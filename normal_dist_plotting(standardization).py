# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 14:20:56 2022

@author: bilal
"""

import numpy as np
import matplotlib.pyplot as plt
#from scipy import stats
import seaborn as sns

#mu, sigma = 13.5, 3.7
x = np.random.normal(13.5, 3.7, 100)
y = np.random.normal(5, 1.5, 100)



plt.scatter(x,y)
plt.xlabel('X data')
plt.ylabel('Y data')
plt.title('raw features')
plt.xlim([-3, 22])
plt.ylim([-3, 9]) 


def standardize(lst):
    lst = (lst - np.mean(lst)) / np.std(lst)
    return lst

xnew = standardize(x)
ynew = standardize(y)


plt.scatter(xnew,ynew)
plt.xlabel('X data')
plt.ylabel('Y data')
plt.title('standardized features ')
plt.xlim([-3, 22])
plt.ylim([-3, 9]) 




sns.distplot(x)
plt.title('raw data')


sns.distplot(xnew)
plt.title('standardized data')