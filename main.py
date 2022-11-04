import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import math
import pandas as pd

# %%
# open weatherHistory.csv and use that data to create a histogram
open('weatherHistory.csv', 'r')
data = np.genfromtxt('weatherHistory.csv', delimiter=',')

# %%
# Y (Apparent Temperature (C)) on X (Temperature (C))
# X is column 3
# Y is column 4
x = data[:, 3]
y = data[:, 4]
# clean x and y of NaN values
x = x[~np.isnan(x)]
y = y[~np.isnan(y)]

# %%
# perform a single linear regression of
# and plot the data and the regression line
model = LinearRegression()
model.fit(x.reshape(-1, 1), y.reshape(-1, 1))
y1 = model.predict(x.reshape(-1, 1))
plt.scatter(x, y, s=8, color='blue')
plt.plot(x, y1, color='red')
plt.xlabel('Temperature (C)')
plt.ylabel('Apparent Temperature (C)')
plt.show()

# %%
# get score of the model
score = model.score(x.reshape(-1, 1), y.reshape(-1, 1))
print('score = ', score)
# get equation of the line
print('y = ', model.coef_[0][0], 'x + ', model.intercept_[0])