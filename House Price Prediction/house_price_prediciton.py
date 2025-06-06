# -*- coding: utf-8 -*-
"""House Price prediciton.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Ac0dQHJiKV8ztrc8KbZHNLINOtcfWVdN

Importing Dependencies
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import sklearn.datasets
from xgboost import XGBRegressor
from sklearn import metrics

"""Import the Dataset"""

house_price_dataset=sklearn.datasets.fetch_california_housing()

print(house_price_dataset)

#adding data array to dataframe
house_price_dataframe=pd.DataFrame(house_price_dataset.data, columns=house_price_dataset.feature_names)

house_price_dataframe.head()

#adding target array to data frame
house_price_dataframe['price']=pd.DataFrame(house_price_dataset.target)

#printing 1st five instances of the data
house_price_dataframe.head()

# checking the number of rows and columns in the dataframe
house_price_dataframe.shape

#check for missing values in each colume
house_price_dataframe.isnull().sum()

#getting the statistical insights of the data
house_price_dataframe.describe()

corelation = house_price_dataframe.corr()

"""corelation is the dataframe containing corelation vlaues being plotted

cbar : adds tthe color bar on the right side helps to interpret the datavalues based on colors

square : ensures each cell is in square shape

fmt : to show the decimal points upto 1 decimal number and for whole number, with 1 decimal point

annot : enables each cell to display its value

annot_kws : size of the value inside the square

cmap : color of the map
"""

#understanding the corelation between the various featues in dataset
plt.figure(figsize=(10,10))
sns.heatmap(corelation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':8}, cmap='Blues')

"""Splitting the data and target"""

x=house_price_dataframe.drop(['price'],axis =1) #for dropping row axis=0 ; for dropping column axis=1
y=house_price_dataframe['price']

print(x)
print(y)

"""Splitting the data into trainging data and test data"""

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=1)

#checking the nubmer of instances in train and test split
print(x.shape,x_train.shape,x_test.shape)

"""Model Training

XGBoost Regressor - ensemble model
"""

model = XGBRegressor()

#training the model with x_train
model.fit(x_train, y_train)

"""Evaluation"""

#accuracy on prediction on training dataset
training_data_prediction = model.predict(x_train)

print(training_data_prediction)

#R square error
score_1=metrics.r2_score(y_train,training_data_prediction)

#mean absolute error
score_2=metrics.mean_absolute_error(y_train,training_data_prediction)

print("R square value : ",score_1)      #the value should be near to 0 then the model is predicting well, lesser the value higher the model accuracy

print("Mean absolute error : ",score_2)

"""Visualize the acutal and pedicted prices"""

plt.scatter(y_train,training_data_prediction) #scatter(x_axis, y_axis)
plt.xlabel("Actual Price")
plt.ylabel("Predicted price")
plt.title("Actual price vs predicted price")
plt.show()

"""Prediction on Test data"""

test_data_prediction = model.predict(x_test)

#R square error
score_1=metrics.r2_score(y_test,test_data_prediction)

#mean absolute error
score_2=metrics.mean_absolute_error(y_test,test_data_prediction)

print("R square error for test data : ",score_1)

print("Mean absolute error for test data : ",score_2)

"""Predicting model"""

input_data=(8.3252, 41.0, 6.984127, 1.023810, 322.0, 2.555556, 37.88, -122.23)

input_data_as_numpy_array=np.asarray(input_data)
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
prediction=model.predict(input_data_reshaped)

print("House price : ",prediction)