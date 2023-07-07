""""
Reference: https://www.pluralsight.com/guides/machine-learning-neural-networks-scikit-learn
"""

### 1. Load req libraries and modules
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
battery_585 = SoC_Estimator("cp3473x7xv-3/LG_HG2_Original_Dataset_McMasterUniversity_Jan_2020/0degC/585_Dis_0p5C.csv")
battery_585_data = battery_585.read_data()
print("test.shape", battery_585_data.shape)
print("PRINTING TEST -----------------")
print(battery_585_data)

print("TEST TRANSPOSE")
print(battery_585_data.describe().transpose())



### 3. Create Arrays for Features and Response Variable
# target_col = 'SoC_calc'
target_col = ['SoC_calc']
predictors = ["Voltage","Current","Temperature"]
# Normalize
battery_585_data[predictors] = battery_585_data[predictors]/battery_585_data[predictors].max()
print("TEST NORMAL")
print(battery_585_data.describe().transpose())

print(battery_585_data)

### 4. Creating training and test sets
X = battery_585_data[predictors].values
Y = battery_585_data[target_col].values
print("PRINTING X: ", X)
print("PRINTING Y: ", Y)

print("PRINTING Y.RAVEL(), dimensions: ", np.shape(Y.ravel()), Y.ravel())

X_train, X_test, Y_train, Y_test = train_test_split(X, Y.ravel(), test_size=0.30, random_state=40)
print("PRINTING X_train.shape")
print(X_train[0], Y_train[0])
print("PRINTING X_test.shape")
print(X_test.shape)

### 5. Building, Predicting, and Evaluating the Neural Network Model
from sklearn.neural_network import MLPClassifier

mlp = MLPRegressor(hidden_layer_sizes=(8,8), activation='tanh', solver='lbfgs', max_iter=500)
mlp.fit(X_train,Y_train)
model_score = mlp.score(X_test,Y_test)
print("MODEL SCORE: ", model_score)