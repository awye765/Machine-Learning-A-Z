# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# Splitting the dataset into the Training set and Test set
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

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""