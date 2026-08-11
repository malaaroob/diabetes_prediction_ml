# Diabetes Prediction using Machine Learning

A machine learning project that predicts diabetes status using clinical biomarkers, deployed as an interactive Streamlit web app.
## Overview

This project covers the full ML pipeline — from raw data to a working, deployed application:
- Exploratory Data Analysis (EDA)
- Data cleaning (missing values, outlier detection)
- Feature engineering (encoding categorical variables, feature selection)
- Handling class imbalance using SMOTE
- Training and comparing 5 classification models
- Deploying the best model with Streamlit

## Dataset

**Source:** Real Diabetes Lab Results & Biomarkers (Anti-Pima) Kaggle (License: CC0 Public Domain)

The dataset contains 20,111 records with clinical features: Age, Gender, BMI, Systolic BP, Diastolic BP, and HbA1c Level, with a target variable Diabetes_Status (0 = No Diabetes, 1 = Prediabetes, 2 = Diabetes).

## Process

**1. EDA & Cleaning**
- Identified missing values in BMI, Systolic_BP, Diastolic_BP, and HbA1c_Level (all under 10%)
- Used boxplots to detect outliers found significant outliers in all numeric columns
- Filled missing values using median (instead of mean) since it's robust to outliers

**2. Feature Engineering**
- Encoded Gender using LabelEncoder
- Dropped Source_Year not medically relevant to diabetes prediction

**3. Handling Class Imbalance**
- Discovered severe class imbalance: Class 0 (~86%), Class 2 (~12%), Class 1 (~2.4%)
- Initial models achieved high accuracy (~91%) but completely failed to detect Class 1 (Prediabetes) 0% recall
- Applied SMOTE to the training data to balance the classes
- Result: accuracy dropped (~83%) but Class 1 recall improved meaningfully a trade-off that matters more in a medical context, where missing a prediabetes case is worse than a lower overall accuracy score

**4. Model Comparison**
Trained and evaluated 5 classification models:
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Support Vector Classifier (SVC)
- Naive Bayes

**Final model:** Decision Tree (trained on SMOTE-balanced data) chosen for the best balance between overall accuracy and minority class detection.

## Tech Stack
- Python, pandas, NumPy
- scikit-learn (models, preprocessing, metrics)
- imbalanced-learn (SMOTE)
- Streamlit (deployment)
- joblib (model persistence)

This was a learning project built to practice the complete ML workflow from data cleaning to deployment with a focus on understanding why each step matters, not just implementing it.