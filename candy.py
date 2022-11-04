#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import math
import pandas as pd

#%%
# open candy-data.csv
open('candy-data.csv', 'r')
data = np.genfromtxt('candy-data.csv', delimiter=',')

#%%
# extract the chocolate column
chocolate = data[:, 1]
# extract the sugar percent column
price_percentile = data[:, 11]

# clean chocolate and price_percentile of NaN values
chocolate = chocolate[~np.isnan(chocolate)]
price_percentile = price_percentile[~np.isnan(price_percentile)]
# multiply price_percentile by 100 to get a percentage
price_percentile = price_percentile * 100
# combine chocolate and price_percentile into a single array
X = np.column_stack((price_percentile, chocolate))
# sort X by price_percentile
X = X[X[:, 0].argsort()]
# extract the price_percentile column
price_percentile = X[:, 0]
# extract the chocolate column
chocolate = X[:, 1]

#%%
# perform a logistic regression of chocolate on sugar_percent
model = LogisticRegression(solver='liblinear')
model.fit(price_percentile.reshape(-1, 1), chocolate)
chocolate1 = model.predict(price_percentile.reshape(-1, 1))
# get the probability that a candy is chocolate
# chocolate1 = chocolate1[:, 1]

#%%
# plot the data
plt.scatter(price_percentile, chocolate, s=8, color='blue')
plt.plot(price_percentile, chocolate1, color='red')
plt.xlabel('Price Percentile')
plt.ylabel('Chocolate')
plt.show()

#%%
# get score of the model
score = model.score(price_percentile.reshape(-1, 1), chocolate)
print('score = ', score)
# get equation of the line
print('y = ', model.coef_[0][0], 'x + ', model.intercept_[0])
