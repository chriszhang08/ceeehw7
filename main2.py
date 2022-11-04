# Step1: Import package
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

#%%
# Step2: Get Data
data = pd.read_csv('weatherHistory.csv')
x = data[['Temperature (C)', 'Apparent Temperature (C)']] # Volume is x1 and Weight is x2
y = data['Humidity'] # CO2 dataset is y

#%%
# Step3: Do linear Regression
# Create an instance of a linear regression model and fit it to the data with the fit() function:
model = LinearRegression()
model.fit(x, y)

#%%
# The following section will get results by interpreting the created instance:

# Obtain the coefficient of determination by calling the model with the score() function, then print the coefficient:
r_sq = model.score(x, y)
print('coefficient of determination:', r_sq)

# Print the Intercept:
print('intercept:', model.intercept_)
a = model.intercept_

# Print the Slope:
print('slope:', model.coef_)
b1 = model.coef_[0]
b2 = model.coef_[1]

#%%
# Predict a Response and print it:
y_pred = model.predict(x)
print('Predicted response:', y_pred, sep='\n')

# Print the formula of regression multiple regression equation
print(f'Formula of multiple regression equation: Humidity = {a} + {b1} Temperature + {b2} App Temperature')

#%%
# # Step 4: Plot scatter points and regression line
x1_pred = np.linspace(min(data['Temperature (C)']), max(data['Temperature (C)']), 100)  # range of x1('Volume') values
x2_pred = np.linspace(min(data['Apparent Temperature (C)']), max(data['Apparent Temperature (C)']), 100)  # range of x2('Weight') values
#%%
xx1_pred, xx2_pred = np.meshgrid(x1_pred, x2_pred) # Generate the mesh
model_viz = np.array([xx1_pred.flatten(), xx2_pred.flatten()]).T
#%%
predicted = model.predict(model_viz) # y values ('CO2') for visualization

#%%
plt.style.use('default')
fig = plt.figure(figsize=(12, 12))
ax1 = fig.add_subplot(132, projection='3d')
# ax2 = fig.add_subplot(132, projection='3d')
# ax3 = fig.add_subplot(133, projection='3d')

axes = [ax1]

for ax in axes:
    ax.plot(x['Temperature (C)'], x['Apparent Temperature (C)'], y, color='k', zorder=15, linestyle='none', marker='o', alpha=0.5)
    ax.scatter(xx1_pred.flatten(), xx2_pred.flatten(), s=20, edgecolor='#70b3f0')
    ax.set_xlabel('Temperature (C)', fontsize=12)
    ax.set_ylabel('Apparent Temperature (C)', fontsize=12)
    ax.set_zlabel('Humidity', fontsize=12)
    ax.locator_params(nbins=4, axis='x')
    ax.locator_params(nbins=5, axis='x')

ax1.view_init(elev=27, azim=112)
# ax2.view_init(elev=16, azim=-51)
# ax3.view_init(elev=60, azim=165)

fig.suptitle('$R^2 = %.2f$' % (model.score(x, y)), fontsize=20)
fig.tight_layout()

plt.show()
