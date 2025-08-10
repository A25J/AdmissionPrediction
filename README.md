Admission Prediction using Regression Models
-----------------------------------------------

Overview:
---------
This project implements a machine learning pipeline to predict a student’s admission acceptance probability based on their academic profile and test scores.
It compares multiple regression algorithms using cross-validation and evaluation metrics, then optionally saves the best model to disk.

The dataset used is Admission_Predict.csv, which contains features such as GRE Score, TOEFL Score, CGPA, etc.
-------------------------------------------------

Features:
---------
  1- Data Analysis

    Display dataset info, statistics, and head

    Plot histograms of all numerical features

    Generate correlation heatmap

  2- Machine Learning Models Tested

    Random Forest Regressor
    XGBoost Regressor
    Decision Tree Regressor (CART)
    Support Vector Regressor (SVM)
    K-Nearest Neighbors Regressor (KNN)
    Multiple Linear Regression (MLR)

  3- Evaluation

    Cross-validation (Repeated K-Fold, 10 splits × 3 repeats)
    Metrics: RMSE, R² Score, Standard Deviation
    Optional train/test evaluation with RMSE, MAE, R²

  4- Model Saving

    Best-performing regressor can be saved as .sav using pickle

Metrics Used:
-------------
  1- RMSE (Root Mean Squared Error): Measures prediction error magnitude.

  2- R² Score: Measures how well the model fits the data.

  3- MAE (Mean Absolute Error): Average absolute difference between predictions and actual values.
