import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

np.random.seed(0)
X = np.linspace(0, 10, 30).reshape(-1, 1)
y = np.sin(X).ravel() + 0.1 * np.random.randn(30)

model_under = LinearRegression()
model_under.fit(X, y)
y_under = model_under.predict(X)

model_over = make_pipeline(PolynomialFeatures(15), LinearRegression())
model_over.fit(X, y)
y_over = model_over.predict(X)

model_best = make_pipeline(PolynomialFeatures(3), LinearRegression())
model_best.fit(X, y)
y_best = model_best.predict(X)

plt.scatter(X, y, color='blue', label='Actual data')
plt.plot(X, y_under, color='red', label='Underfit')
plt.plot(X, y_over, color='green', label='Overfit')
plt.plot(X, y_best, color='orange', label='Best fit')
plt.title('Underfitting vs Overfitting vs Best Fitting')
plt.legend()
plt.show()
