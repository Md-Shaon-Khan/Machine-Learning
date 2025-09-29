import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Non-linear data
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = np.sin(X).ravel()  # sine wave

# Linear regression (simple model)
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# Plot
plt.scatter(X, y, color='blue', label='Actual data')
plt.plot(X, y_pred, color='red', label='Predicted (Underfit)')
plt.title('Underfitting Example')
plt.legend()
plt.show()