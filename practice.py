import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("student_performance_dataset.csv")
print(df.shape)
print(df.info())
print(df.columns)
print(df.head())
print(df.describe())
print(df['parental_education'].mode())
df['parental_education'] = df['parental_education'].fillna('High School')
print(df.isnull().sum())
print(df.duplicated().sum())
numeric_cols = df.select_dtypes(include =['int64', 'float64']).columns
print(numeric_cols)
for col in numeric_cols:
    plt.figure(figsize=(6,4))
    sns.histplot(df[col], kde = True, bins=20)
    plt.title(col)
    plt.show()
sns.boxplot(x=df["study_time_hours"])
plt.show()
sns.countplot(x=df["gender"])
plt.show()










