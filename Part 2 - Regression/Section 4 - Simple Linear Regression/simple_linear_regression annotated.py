# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
# [:,] means we take all rows.  [:-1] means we take all columns except for 
# last column, i.e. except the "Salary" column.  
# X is the FEATURES matrix of INDEPENDENT variables.

y = dataset.iloc[:, 1].values
# [:, 1] means we take all rows from the dataset and only the last column, i.e.
# including the "Salary" column.  
# y is a VECTOR of the DEPENDENT variables.

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
# Imports from sklearn.cross_validation library the train_test_split library

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)
# X_train = training part of matrix of features. 
# X_test = training part of matrix of features.
# y_test takes an array of X values, array of y values.  1/3 means one third of
# the dataset will be test data and 2/3 will be training.  Random_state = 0 
# keeps my results consistent with the instructors.

# Fitting Simple Linerar Regression to the Training Set
from sklearn.linear_model import LinearRegression
# Imports the LinearRegression class

regressor = LinearRegression()
# Creates instance of the LinearRegression() class and assigns it to the 
# variable, "regressor".

regressor.fit(X_train, y_train)
# Fits the regressor instance to training data, X_train & y_train.  Doing so
# makes it learn how to predict y values based on the X values.

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""
