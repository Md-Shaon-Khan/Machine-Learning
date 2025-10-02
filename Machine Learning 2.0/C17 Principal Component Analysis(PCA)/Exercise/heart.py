import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

# Load dataset
df = pd.read_csv('F:\Machine Learning\Machine Learning 2.0\C17 Principal Component Analysis(PCA)\Exercise\heartEX.csv')

# Remove outliers
df1 = df[df.Cholesterol <= (df.Cholesterol.mean() + 3*df.Cholesterol.std())]
df2 = df1[df1.Oldpeak <= (df1.Oldpeak.mean() + 3*df1.Oldpeak.std())]
df3 = df2[df2.RestingBP <= (df2.RestingBP.mean() + 3*df2.RestingBP.std())]

# Encode categorical columns (manual mapping)
df4 = df3.copy()
df4.ExerciseAngina.replace({'N': 0, 'Y': 1}, inplace=True)
df4.ST_Slope.replace({'Down': 1, 'Flat': 2, 'Up': 3}, inplace=True)
df4.RestingECG.replace({'Normal': 1, 'ST': 2, 'LVH': 3}, inplace=True)

# One-hot encoding
df5 = pd.get_dummies(df4, drop_first=True)

# Features and target
X = df5.drop("HeartDisease", axis='columns')
y = df5.HeartDisease

# Standard scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=30)

# Random Forest without PCA
model_rf = RandomForestClassifier()
model_rf.fit(X_train, y_train)
print("RandomForest Accuracy without PCA:", model_rf.score(X_test, y_test))

# PCA
pca = PCA(0.95)
X_pca = pca.fit_transform(X_scaled)

# Train-test split with PCA
X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=30)

# Random Forest with PCA
model_rf_pca = RandomForestClassifier()
model_rf_pca.fit(X_train_pca, y_train)
print("RandomForest Accuracy with PCA:", model_rf_pca.score(X_test_pca, y_test))

# PCA Visualization
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=y, cmap='bwr', alpha=0.7)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Visualization of Heart Disease Data")
plt.colorbar(label="HeartDisease (0 = No, 1 = Yes)")
plt.show()