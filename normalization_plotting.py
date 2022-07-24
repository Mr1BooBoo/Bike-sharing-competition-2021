# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 12:57:37 2022

@author: bilal
"""
import numpy as np
import matplotlib.pyplot as plt

x = np.random.uniform(0.1, 5,14)
y = np.random.uniform(0.1, 5,15)

#add an outlier 
x = np.append(x, 9)

############################################################################################################

def normalize(lst):
    lst = (lst - min(lst)) / (max(lst) - min(lst))
    return lst

xnorm = normalize(x)
ynorm = normalize(y)


plt.scatter(x, y)
plt.title('raw data')
plt.xlim([-0.2, max(x)+0.2])
plt.ylim([-0.2, max(x)+0.2]) 

plt.scatter(xnorm, ynorm)
plt.title('normalized data')
plt.xlim([-0.2, max(x)+0.2])
plt.ylim([-0.2, max(x)+0.2]) 

############################################################################################################




