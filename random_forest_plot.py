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
#1 Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#2 Importing the dataset
X = battery_585_data[predictors].values
y = battery_585_data[target_col].values
# Splitting the dataset into the Training set and Test set
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 28)

#3 Fitting the Random Forest Regression Model to the dataset
# Create RF regressor here
from sklearn.ensemble import RandomForestRegressor 
# Initializing the Random Forest Regression model with 10 decision trees
model = RandomForestRegressor(n_estimators = 10, random_state = 0)

# Fitting the Random Forest Regression model to the data
model.fit(x_train, y_train)

# Predicting the target values of the test set
y_pred = model.predict(x_test)

# RMSE (Root Mean Square Error)
rmse = float(format(np.sqrt(mean_squared_error(y_test, y_pred)), '.3f'))
print("\nRMSE: ", rmse)


#Put 10 for the n_estimators argument. n_estimators mean the number #of trees in the forest.
regressor = RandomForestRegressor(n_estimators=10, random_state=0)
regressor.fit(X,y.ravel())
#4 Visualising the Regression results (for higher resolution and #smoother curve)
# X_grid = np.arange(min(X), max(X), 0.01)
# X_grid = X_grid.reshape((len(X_grid), 1))
# plt.scatter(X, y, color = 'red')
# plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
# plt.title('Check It (Random Forest Regression Model)')
# plt.xlabel('Position level')
# plt.ylabel('Salary')
# plt.show()