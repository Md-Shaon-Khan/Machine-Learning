import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

data = pd.read_csv('F:\Machine Learning\Decision Tree\Churn_Modelling.csv')
data = data.drop(columns=['RowNumber', 'CustomerId', 'Surname'], axis=1)

le_gender = LabelEncoder()
data['Gender'] = le_gender.fit_transform(data['Gender'])

ohe_geography = OneHotEncoder(drop='first')
geo_encoded = ohe_geography.fit_transform(data[['Geography']]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=ohe_geography.get_feature_names_out(['Geography']))
data = pd.concat([data.drop(columns=['Geography']), geo_encoded_df], axis=1)

X = data.drop(columns=['Exited'], axis=1)
y = data[['Exited']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=['Actual Negative', 'Actual Positive'], 
                     columns=['Predicted Negative', 'Predicted Positive'])

fig = px.imshow(cm_df, 
                text_auto=True, 
                color_continuous_scale="Viridis", 
                title="Confusion Matrix")

fig.update_layout(
    title={'text': "Confusion Matrix", 'x': 0.5, 'xanchor': 'center'},
    xaxis_title="Predicted Label",
    yaxis_title="True Label",
    coloraxis_showscale=True
)
fig.show()
