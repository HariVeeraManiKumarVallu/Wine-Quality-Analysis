Wine Quality Analysis


This project analyzes the White Wine Quality Dataset using various machine learning classification and regression models. The goal is to predict the quality of wine based on physicochemical attributes.

1. Dataset Overview
The dataset contains 4898 samples with 12 features describing various chemical properties of the wine, such as:

Fixed Acidity
Volatile Acidity
Citric Acid
Residual Sugar
Chlorides
Free Sulfur Dioxide
Total Sulfur Dioxide
Density
pH
Sulphates
Alcohol
The target variable is Quality, which ranges from 3 to 9.

Dataset Distribution

The distribution of wine quality is highly imbalanced:

Quality 6: 44.88%
Quality 5: 29.75%
Quality 7: 17.97%
Others: < 5%

Visualization:
See quality_distribution.png for the count plot of the wine quality distribution.

Data Preprocessing

Key Steps:
Splitting Data:
Training: 80%
Testing: 20%
Stratified to maintain the distribution of target labels.

Feature Scaling:

Used StandardScaler to normalize features for models sensitive to feature scaling (e.g., SVM, Neural Networks).

Exploratory Data Analysis (EDA)

Correlation Heatmap:

A heatmap (correlation_heatmap.png) highlights correlations between features:

Alcohol and Quality show a strong positive correlation.
Density and Quality show a negative correlation.

Classification Models

We applied the following classification models to predict wine quality:

Model	Mean F1 Score	Std Dev
Random Forest	0.634	0.015
SVM	0.449	0.013
Gradient Boosting	0.571	0.011
Logistic Regression	0.360	0.016
K-Nearest Neighbors	0.524	0.012
Neural Network (MLP)	0.545	0.011
Naive Bayes	0.432	0.015

Best Model: Random Forest

Optimized Hyperparameters using GridSearchCV:
n_estimators: 200
max_depth: None
min_samples_split: 5
min_samples_leaf: 1

Test Set Performance:

Accuracy: 68%
Matthews Correlation Coefficient (MCC): 0.507
AUC-ROC Score: 0.849

See the confusion_matrix.png for detailed 

classification performance.

Regression Models

For predicting the wine quality as a continuous variable, we used the following regressors:

Model	RMSE	R² Score
Linear Regression	0.754	0.275
Polynomial Regression (Deg 2)	0.745	0.292
Random Forest Regressor	0.643	0.472
Support Vector Regressor	0.698	0.379
K-Nearest Neighbors Regressor	0.728	0.323
Neural Network Regressor	0.701	0.373
Decision Tree Regressor	0.888	-0.008

Best Regressor: Random Forest Regressor with:

RMSE: 0.643
R² Score: 0.472

Resampling Techniques for Imbalanced Data

Since the dataset is imbalanced, we applied:

SMOTE (Synthetic Minority Oversampling Technique)
RandomOverSampler (Fallback in case SMOTE fails)
After applying SMOTE, the class distribution was balanced:

See the confusion_matrix_smote.png for performance metrics on the resampled dataset.

Performance After Resampling:

F1 Score: 66%
AUC-ROC Score: 0.847

Feature Importance
Random Forest provides feature importance scores, helping us identify key predictors:

Alcohol was the most important feature.
See feature_importance.png for a complete bar chart.

Outputs and Visualizations

The project generates several outputs:

EDA Visualizations:
quality_distribution.png
correlation_heatmap.png
Model Performance:
confusion_matrix.png
confusion_matrix_smote.png
feature_importance.png


Requirements:

imblearn==0.11.0
matplotlib==3.8.0
numpy==1.24.0
pandas==2.1.0
scikit-learn==1.3.0
seaborn==0.12.2

Conclusion:
This project demonstrates:

Effective application of machine learning techniques for classification and regression.
Addressing data imbalance using resampling techniques.
Feature importance analysis to interpret model predictions.
Feel free to fork, contribute, and experiment with other models!