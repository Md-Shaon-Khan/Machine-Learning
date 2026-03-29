import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

np.random.seed(0)  
X = np.linspace(0, 10, 30).reshape(-1, 1) 
y = np.sin(X).ravel() + 0.1 * np.random.randn(30)  

degree = 15  
model = make_pipeline(PolynomialFeatures(degree), LinearRegression())  


model.fit(X, y)  
y_pred = model.predict(X)  

plt.scatter(X, y, color='blue', label='Actual data') 
plt.plot(X, y_pred, color='red', label='Predicted (Overfit)')  
plt.title('Overfitting Example')
plt.legend()
plt.show()
