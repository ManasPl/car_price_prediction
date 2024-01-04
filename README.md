# Car Price Prediction

## Overview

This project focuses on predicting the selling price of cars based on various features using machine learning techniques. The dataset includes information such as car name, year of manufacture, selling price, kilometers driven, fuel type, seller type, transmission type, owner details, mileage, engine details, maximum power, torque, and the number of seats.

## Backend

The backend of this project is implemented using Flask, a web framework for Python. Flask provides a simple and lightweight way to build web applications, making it suitable for serving the machine learning model and handling predictions.

## Exploratory Data Analysis (EDA)

In the Jupyter notebook, an exploratory data analysis (EDA) is performed on the dataset. The EDA includes:

- Changing data types for better analysis.
- Feature selection based on correlation thresholds.
- One-hot encoding for categorical columns.
- Feature selection for categorical columns using Lasso Regression.
- Standard scaling of numerical features.
- Test-train split for model evaluation.

### Data Visualization

#### Bar Plot

A bar plot is utilized to visualize the distribution of certain categorical features, providing insights into the dataset's composition.

#### Scatter Plot

Scatter plots are used to explore relationships between numerical variables, aiding in identifying patterns and potential correlations.

## Model Training

For the model training phase, the following steps are taken:

- Polynomial features are introduced to capture non-linear relationships.
- A comparison is made between various regression models, including Linear Regression, Ridge Regression, Decision Tree Regressor, XGBoost Regressor, and Support Vector Regressor (SVR).
- A grid search is performed to find the optimal hyperparameters for each model.

## Model Comparison

The models are compared based on their performance metrics, and the best-performing model is identified. The comparison includes evaluating models on metrics such as mean squared error, mean absolute error, and R-squared.

### Selected Models:

- **XGBoost Regressor**
  - Hyperparameters: {'n_estimators': [100, 200]}

- **Decision Tree Regressor**
  - Hyperparameters: {'max_depth': [None, 5, 10]}

- **Ridge Regression**
  - Hyperparameters: {'alpha': [0.1, 0.5, 1.0]}

## Model Deployment

The best-performing model is saved using the `pickle` library for later deployment. The serialized model can be loaded and used for real-time predictions once the Flask web application is deployed.

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/ManasPl/car_price_pred.git
