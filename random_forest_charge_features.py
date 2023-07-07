# random forest for feature importance on a regression problem
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot




"""
Old charge curve
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
battery_585 = SoC_Estimator("cp3473x7xv-3/LG_HG2_Original_Dataset_McMasterUniversity_Jan_2020/25degC/549_Charge.csv")
battery_585_data, starts, ends = battery_585.read_data()
print("test.shape", battery_585_data.shape)
print("PRINTING TEST -----------------")
print(battery_585_data)

print("TEST TRANSPOSE")
print(battery_585_data.describe().transpose())

print("STARTS: ", starts)
print("ENDS: ", ends)
# print("CYCLE_STARTS")
# print(battery_585_data.cycle_starts)
# print("CYCLE ENDS")
# print(battery_585_data.cycle_ends)


### 3. Create Arrays for Features and Response Variable
# target_col = 'SoC_calc'
target_col = ['SoC_calc']
predictors = ["Voltage","Current","Temperature","Charging_Cycle"]
# Normalize
battery_585_data[predictors] = battery_585_data[predictors]/battery_585_data[predictors].max()
print("TEST NORMAL")
print(battery_585_data.describe().transpose())

print(battery_585_data)

print("CYCLE NUM COLUMN: _______________________________: ", battery_585_data['Charging_Cycle'])

"""
Random forest regressor
"""
# define dataset
X = battery_585_data[predictors].values
y = battery_585_data[target_col].values
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