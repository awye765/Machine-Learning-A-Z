# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# =============================================================================
# Importing the dataset
# =============================================================================

dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
# INDEPENDENT VARIABLES = R&D Index, Administration, Marketing Spend and State.
# DEPENDENT VARIABLE = PROFIT.  [:,] takes ALL ROWS and [, :-1] takes ALL 
# COLUMNS EXCEPT the last column, i.e. Profit.
#
# Note cannot preview this data subset in the variable explorer as it is 
# type "object".  This is because it contains multiple data types, i.e. both
# floats (i.e. the financial data) AND strings (i.e. the State categories, e.g.
# "California" or "New York).

y = dataset.iloc[:, 4].values
# [:,] takes ALL ROWS and [4] takes the final column, i.e. the DEPENDENT
# VARIABLE, which is the PROFIT.

# =============================================================================
# Encoding the INDEPENDENT categorical data
# =============================================================================

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
# Creates instance of the LabelEncoder() class and assigns to varibale 
# "labelencoder"

X[:, 3] = labelencoder.fit_transform(X[:, 3])
# X[:, 3] identifies the column we want to encode, i.e. the column containing 
# the independent categorical variables (the State names).  
#
# The fit_transform method of the labelencoder instance is applied to that 
# column.  Finally, the X[:, 3] value is then updated with those transformed 
# values, i.e. so that the STRING values for each State are replaced with a 
# NUMERICAL representation, as follows:
# 0 = California
# 1 = Florida
# 2 = New Tork

# =============================================================================
# ONE HOT ENCODING the INDEPENDENT categorical data
# =============================================================================

onehotencoder = OneHotEncoder(categorical_features = [3])
# Creates instance of the OneHoteEncoder class with the parameter 
# categorical_features set to index 3.  This specifies that the column at index
# 0, i.e. the State column, is the one we want to OneHotEncode.  
#
# Applying one hot encoding we create 3 new BOOLEAN columns for each category, 
# i.e. each State.  Only one column can take the value 1 for each sample, hence
# the term ONE hot encoding.  
#
# Visual example here: https://www.quora.com/What-is-one-hot-encoding-and-when-is-it-used-in-data-science

X = onehotencoder.fit_transform(X).toarray()
# Transforms X by fitting onehotencoder object to array.  

# =============================================================================
# Avoiding the Dummy Variable Trap
# =============================================================================

X = X[:, 1:]

# =============================================================================
# Splitting the dataset into the Training set and Test set
# =============================================================================

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# =============================================================================
# Multiple Linear Regression
# =============================================================================

# Multiple linear regression is where there are MULTIPLE INDEPENDENT X 
# variables to be analysed to understand the DEPENDENT variable Y, e.g.
# y = b0 + b1*x1 + b2*x2 + b3*x3 + b4*x4

# =============================================================================
# Dummy Variables
# =============================================================================

# A variable that takes the value of either 0 or 1 to indicate the presence or
# absence of some categorical effect, e.g. smoker / non-smoker.  In essence
# dummy variables are proxies or numeric stand-ins for QUALITATIVE facts in
# a regression model.

# =============================================================================
# Expand numpy array
# =============================================================================

# np.set_printoptions(threshold='nan')
# np.set_printoptions(threshold=np.inf)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)