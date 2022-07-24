# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 12:12:42 2022

@author: bilal
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split


path = "C:/Users/bilal/OneDrive/Desktop/Code/Bike Sharing Demand/"
train_data = pd.read_csv(os.path.join(path, "train.csv"))
test_data = pd.read_csv(os.path.join(path, "test.csv"))

train_y = train_data['count']
train_data.drop(['count'],axis=1,inplace=True)

train_data['temp'].describe()
train_data['windspeed'].describe()


plt.hist(train_data.windspeed)
plt.ylabel('number of rented bikes')
plt.xlabel('wind speed')
plt.show()


def standarize(df,column):
    standarized = df[column] = (df[column]-df[column].mean()) / df[column].std()
    return df[column].mean(),df[column].std()

def normalize(df, column):
    normalized= df[column] = (df[column]-df[column].min())/(df[column].max()-df[column].min())
    return df[column].min(), df[column].max()

def destandardize(df,column, mean, std):
    destandardized = df[column] = df[column] * std + mean
    return None

def denormalize(df, column, xmin, xmax):
    denormalized= df[column] = df[column] * (xmax - xmin) + xmin
    return None


#rescale training data
standarize(train_data,'temp')
standarize(train_data,'atemp')
normalize(train_data,'windspeed')
normalize(train_data,'humidity')

#rescale testing data
standarize(test_data,'temp')
standarize(test_data,'atemp')
normalize(test_data,'windspeed')
normalize(test_data,'humidity')


#split year,month,day,hour of the training data
train_data['datetime'] = train_data.datetime.apply(pd.to_datetime)

train_data['year'] = train_data.datetime.apply(lambda x:x.year)
train_data['month'] = train_data.datetime.apply(lambda x:x.month)
train_data['day'] = train_data.datetime.apply(lambda x:x.day)
train_data['hour'] = train_data.datetime.apply(lambda x:x.hour)


##split year,month,day,hour of the testing data
test_data['datetime'] = test_data.datetime.apply(pd.to_datetime)

test_data['year'] = test_data.datetime.apply(lambda x:x.year)
test_data['month'] = test_data.datetime.apply(lambda x:x.month)
test_data['day'] = test_data.datetime.apply(lambda x:x.day)
test_data['hour'] = test_data.datetime.apply(lambda x:x.hour)

#drop irrelevant columns
train_data.drop(['datetime','atemp','casual','registered'],axis=1,inplace=True)
test_data.drop(['datetime','atemp'],axis=1,inplace=True)



###################################   Linear regression model ####################################
x_train,x_test,y_train,y_test = train_test_split(train_data,train_y,test_size=0.25)

model = keras.Sequential()
model.add(keras.layers.Dense(5000,activation='relu',input_shape=[11,]))
model.add(keras.layers.Dropout(0.7))
model.add(keras.layers.Dense(1000,activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(1))


model.compile(optimizer='Adam',
              loss='mae',
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=100)
model.summary()

plt.plot(history.history['loss'])



################################## Random Forests model #####################################
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
#Splitting the dataset into the Training set and Test set
x_train,x_test,y_train,y_test = train_test_split(train_data,train_y,test_size=.2, random_state=2)

#iter thru the models to find the number of trees with best MAE 
MAE_list =[]
print('finding the best number of trees.')
for i in range(10,550,10):
    regressor = RandomForestRegressor(n_estimators = i, random_state = 0)
    regressor.fit(x_train,y_train)
    y_predicted = regressor.predict(x_test)
    MAE_list.append(metrics.mean_absolute_error(y_test,y_predicted))
    print("({}% completed)".format(100*i//550))


plt.figure(figsize=(10,6))
plt.plot(range(10,550,10),MAE_list,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('MAE vs. number of estimators(trees)')
plt.xlabel('Trees')
plt.ylabel('MAE')
plt.show()


regressor = RandomForestRegressor(n_estimators = 115, random_state = 0)
regressor.fit(x_train,y_train)
y_predicted = regressor.predict(x_test)


metrics.mean_absolute_error(y_test, y_predicted)
#metrics.mean_squared_error(y_test, y_predicted)
# Store scores of this model in score_table








