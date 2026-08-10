import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('custom_multi_year_diabetes_dataset.csv')
print(df.head())
print(df.describe())
print(df['Diabetes_Status'].value_counts())
print(df.isnull().sum())
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
sns.boxplot(x=df['BMI'], ax=axes[0,0])
sns.boxplot(x=df['Systolic_BP'], ax=axes[0,1])
sns.boxplot(x=df['Diastolic_BP'], ax=axes[1,0])
sns.boxplot(x=df['HbA1c_Level'], ax=axes[1,1])

plt.tight_layout()
plt.show()
df['BMI'] = df['BMI'].fillna(df['BMI'].median())
df['Systolic_BP'] = df['Systolic_BP'].fillna(df['Systolic_BP'].median())
df['Diastolic_BP'] = df['Diastolic_BP'].fillna(df['Diastolic_BP'].median())
df['HbA1c_Level'] = df['HbA1c_Level'].fillna(df['HbA1c_Level'].median())

print(df.isnull().sum())
print(df['Gender'].unique())
print(df['Source_Year'].unique())

le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
print(df['Gender'].unique())

df = df.drop('Source_Year', axis=1)

X = df.drop('Diabetes_Status', axis=1)
y = df['Diabetes_Status']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(X_train.shape)
print(X_test.shape)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

print(y_train.value_counts())
print(y_train_smote.value_counts() if hasattr(y_train_smote, 'value_counts') else pd.Series(y_train_smote).value_counts())

models = {
    'Logistic Regression': LogisticRegression(),
    'KNN': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'SVC': SVC(),
    'Naive Bayes': GaussianNB()
}
for name, model in models.items():
    model.fit(X_train_smote, y_train_smote)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")

dt_model = DecisionTreeClassifier()
dt_model.fit(X_train_smote, y_train_smote)
y_pred_dt = dt_model.predict(X_test_scaled)
print(classification_report(y_test, y_pred_dt))

import joblib
final_model = DecisionTreeClassifier()
final_model.fit(X_train_smote, y_train_smote)

joblib.dump(final_model, 'diabetes_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le, 'label_encoder.pkl')
joblib.dump(X.columns.tolist(), 'columns.pkl')
print("Saved successfully!")
print(X.columns.tolist())




