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

#print("CYCLE NUM COLUMN: _______________________________: ", data['Charging_Cycle'])

### 4. Creating training and test sets
X = data[predictors].values
y = data[target_col].values

X_train, X_test, Y_train, Y_test = train_test_split(X, y.ravel(), test_size=0.30, random_state=40)
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


# Simulate train / test / validation sets
# X, y = make_regression(n_samples=1000)
X_train, X_hold, y_train, y_hold = train_test_split(X, y.ravel(), train_size=.6)
X_valid, X_test, y_valid, y_test = train_test_split(X_hold, y_hold, train_size=.5)

# using 'adam' solver (instead of "lbfgs") b/c can only use partial solvers for batches
reg = MLPRegressor(hidden_layer_sizes=(8,8), activation='tanh', solver='adam', max_iter=500)
batch_size, train_loss_, valid_loss_ = 50, [], []

# for _ in range(150):
#     print("LOOP ITERATION", _)
#     for b in range(batch_size, len(y_train), batch_size):
#         X_batch, y_batch = X_train[b-batch_size:b], y_train[b-batch_size:b]
#         reg.partial_fit(X_batch, y_batch)
#         train_loss_.append(reg.loss_)
#         valid_loss_.append(mean_squared_error(y_valid, reg.predict(X_valid) / 2))

# plt.plot(range(len(train_loss_)), train_loss_, label="train loss")
# plt.plot(range(len(train_loss_)), valid_loss_, label="validation loss")
# plt.legend()

# plt.show()

model_scores = []

for _ in range(150):
    print("LOOP ITERATION", _)
    iteration_scores = []
    for b in range(batch_size, len(y_train), batch_size):
        X_batch, y_batch = X_train[b-batch_size:b], y_train[b-batch_size:b]
        reg.partial_fit(X_batch, y_batch)
        model_score = reg.score(X_batch,y_batch)
        iteration_scores.append(model_score)
        train_loss_.append(reg.loss_)
        valid_loss_.append(mean_squared_error(y_valid, reg.predict(X_valid) / 2))
    mean_curve_iteration = np.mean(iteration_scores, axis=0)
    model_scores.append(mean_curve_iteration)

# plt.plot(range(len(train_loss_)), train_loss_, label="train loss")
# plt.plot(range(len(train_loss_)), valid_loss_, label="validation loss")
# plt.legend()

## SUBPLOT 1: Training & Validation
plt.subplot(1, 2, 1)  # row 1, column 2, count 1
plt.plot(range(len(train_loss_)), train_loss_, label="train loss")
plt.plot(range(len(train_loss_)), valid_loss_, label="validation loss")
plt.title('Training and Validation Losses')
plt.xlabel('Steps')
plt.ylabel('Loss')

## SUBPLOT 2: Model Score
plt.subplot(1, 2, 2)  # row 1, column 2, count 1
plt.plot(range(150), model_scores, label="model score")
plt.title('Model Scores')
plt.xlabel('Batch')
plt.ylabel('Model Score')

print("MEAN MODEL SCORE: ", np.mean(model_scores))

# space between the plots
plt.tight_layout(4)

plt.show()