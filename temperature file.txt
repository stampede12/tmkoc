import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('temperatures.csv')
print(df)

print(df.head()) #printing first 5 rows

#input data
x = df['YEAR']

#output data
y = df['ANNUAL']

#plot figure
plt.figure(figsize=(8,5))
plt.title('Temperature Plot of INDIA')
plt.xlabel('Year')
plt.ylabel('Annual Average Temperature')
plt.scatter(x, y)
plt.show()

#1D array to 2D array
x = x.values
x = x.reshape(117, 1)
print(x.shape)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression() #obj created
regressor.fit(x, y) #Model Trained

print("Slope(m):",regressor.coef_)
print("C:",regressor.intercept_)

print("Predicted Temperature->",regressor.predict([[2024]]))  #yearly prediction
predicted= regressor.predict(x)

#Mean Absolute error
#import numpy as np
#print("Mean Absolute Error->",np.mean(abs(y - predicted)))

from sklearn.metrics import mean_absolute_error
print("Mean Absolute Error:",mean_absolute_error(y, predicted))

from sklearn.metrics import mean_squared_error
print("Mean Sqaured Error:",mean_squared_error(y, predicted))

from sklearn.metrics import r2_score
print("R2 Score:",r2_score(y, predicted))


#Regression Plot
plt.figure(figsize=(8,5))
plt.title('Temperature Plot of INDIA')
plt.xlabel('Year')
plt.ylabel('Annual Average Temperature')
plt.scatter(x, y, label = 'actual', color = 'r')
plt.plot(x, predicted, label = 'predicted', color = 'g')
plt.legend()
plt.show()

#Without implementing Linear Regression
sns.regplot(x='YEAR', y='ACTUAL', data=df)
plt.show()