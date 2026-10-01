# Heart Disease Risk Prediction using Machine Learning

## Overview

This project is a machine learning application designed to predict heart disease risk based on clinical and diagnostic input features.

The project covers the complete workflow from preparing the input data and scaling features to training a Support Vector Machine (SVM) model and deploying it as an interactive Flask web application.

The application provides a simple web interface where users can enter the required clinical parameters and receive the model's predicted class.

> Disclaimer: This project is created for educational and machine learning demonstration purposes only. It is not a medical diagnostic tool and should not be used to make healthcare decisions.

---

## Project Features

- Data preprocessing and feature preparation
- Feature scaling using a saved scaler
- Support Vector Machine (SVM) classification
- Model serialization using Joblib
- Flask backend for model deployment
- HTML/CSS frontend
- Input validation through form constraints
- Prediction result display
- Responsive web interface
- Separate saved model, scaler, and feature-column files

---

## Machine Learning Model

The project uses a Support Vector Machine (SVM) classifier.

The model receives 13 input features:

1. Age
2. Sex
3. Chest Pain Type (cp)
4. Resting Blood Pressure (trestbps)
5. Cholesterol (chol)
6. Fasting Blood Sugar (fbs)
7. Resting ECG (restecg)
8. Maximum Heart Rate Achieved (thalach)
9. Exercise-Induced Angina (exang)
10. ST Depression (oldpeak)
11. Slope
12. Number of Major Vessels (ca)
13. Thalassemia-related feature (thal)

The target variable represents the prediction class used by the trained model.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Flask
- HTML
- CSS

---

## Project Workflow

```text
Input Data
    ↓
Data Preprocessing
    ↓
Feature Selection
    ↓
Feature Scaling
    ↓
SVM Model Training
    ↓
Model Evaluation
    ↓
Model Serialization
    ↓
Flask Deployment
    ↓
Web Interface
    ↓
Prediction
