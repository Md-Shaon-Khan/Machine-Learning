from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Load dataset
iris = load_iris()

# Logistic Regression
l_scores = cross_val_score(LogisticRegression(), iris.data, iris.target)
l_avg = np.average(l_scores)

# Decision Tree
d_scores = cross_val_score(DecisionTreeClassifier(), iris.data, iris.target)
d_avg = np.average(d_scores)

# Support Vector Machine (SVM)
s_scores = cross_val_score(SVC(), iris.data, iris.target)
s_avg = np.average(s_scores)

# Random Forest
r_scores = cross_val_score(RandomForestClassifier(n_estimators=40), iris.data, iris.target)
r_avg = np.average(r_scores)

# Optional: print averages
print("Logistic Regression Average Accuracy:", l_avg)
print("Decision Tree Average Accuracy:", d_avg)
print("SVM Average Accuracy:", s_avg)
print("Random Forest Average Accuracy:", r_avg)