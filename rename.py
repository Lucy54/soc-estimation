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
y = battery_585_data[target_col].values

X_train, X_test, Y_train, Y_test = train_test_split(X, y.ravel(), test_size=0.30, random_state=40)
print("PRINTING X_train.shape")
print(X_train[0], Y_train[0])
print("PRINTING X_test.shape")
print(X_test.shape)

### 5. Building, Predicting, and Evaluating the Neural Network Model
from sklearn.neural_network import MLPClassifier

# Simulate train / test / validation sets
# X, y = make_regression(n_samples=1000)
X_train, X_hold, y_train, y_hold = train_test_split(X, y.ravel(), train_size=.6)
X_valid, X_test, y_valid, y_test = train_test_split(X_hold, y_hold, train_size=.5)

# using 'adam' solver (instead of "lbfgs") b/c can only use partial solvers for batches
reg = MLPRegressor(hidden_layer_sizes=(8,8), activation='tanh', solver='adam', max_iter=500)
batch_size, train_loss_, valid_loss_ = 50, [], []

for _ in range(150):
    for b in range(batch_size, len(y_train), batch_size):
        X_batch, y_batch = X_train[b-batch_size:b], y_train[b-batch_size:b]
        reg.partial_fit(X_batch, y_batch)
        train_loss_.append(reg.loss_)
        valid_loss_.append(mean_squared_error(y_valid, reg.predict(X_valid) / 2))

plt.plot(range(len(train_loss_)), train_loss_, label="train loss")
plt.plot(range(len(train_loss_)), valid_loss_, label="validation loss")
plt.legend()

plt.show()