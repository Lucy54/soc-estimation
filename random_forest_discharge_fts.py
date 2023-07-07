"""
Discharge file
"""

# Call regressor
# Import required libraries
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import sklearn
from sklearn.neural_network import MLPRegressor

# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score

from SoC_Estimator import SoC_Estimator

### 2. Read data; basic data checks
data = SoC_Estimator("Oxford Data/discharge.csv")
data, starts, ends = data.read_discharge_data()
print("test.shape", data.shape)
print("PRINTING TEST -----------------")
print(data)

print("TEST TRANSPOSE")
print(data.describe().transpose())

print("STARTS: ", starts)
print("ENDS: ", ends)

### 3. Create Arrays for Features and Response Variable
# target_col = 'SoC_calc'
target_col = ['SoC_calc']
predictors = ["Voltage","Current","Temperature","Charging_Cycle"]
# Normalize
data[predictors] = data[predictors]/data[predictors].max()
print("TEST NORMAL")
print(data.describe().transpose())

print(data)

"""
Random forest regressor
"""
# random forest for feature importance on a regression problem
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot
# define dataset
X = data[predictors].values
y = data[target_col].values
print("PRINT X: ", X)
print("PRINT Y: ", y)
# define the model
model = RandomForestRegressor()
# fit the model
model.fit(X, y.ravel())
# get importance
importance = model.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
 print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
plt.xlabel('Feature')
plt.ylabel('Percentage')
pyplot.show()