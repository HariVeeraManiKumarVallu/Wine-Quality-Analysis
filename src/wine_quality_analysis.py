import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, label_binarize, PolynomialFeatures
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, matthews_corrcoef
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from imblearn.over_sampling import SMOTE, RandomOverSampler
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Set base directory for the project
base_dir = r'C:\Users\manik\Downloads\Wine Quality Analysis'
output_dir = base_dir  # Save figures here

# Path to the CSV file
csv_file_path = os.path.join(base_dir, 'src', 'winequality-white.csv')

# Load the dataset
data = pd.read_csv(csv_file_path, sep=';')

# Inspect the dataset
print(data.describe())
print(data['quality'].value_counts(normalize=True))

# Distribution of wine quality
plt.figure(figsize=(10, 6))
sns.countplot(x='quality', data=data)
plt.title('Distribution of Wine Quality')
plt.savefig(os.path.join(output_dir, 'quality_distribution.png'))
plt.close()
print("Saved: quality_distribution.png")

# Correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
plt.close()
print("Saved: correlation_heatmap.png")

# Split features and target
X = data.drop('quality', axis=1)
y = data['quality']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define classifiers
classifiers = {
    'Random Forest': RandomForestClassifier(random_state=42, class_weight='balanced'),
    'SVM': SVC(random_state=42, class_weight='balanced', probability=True),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Neural Network': MLPClassifier(random_state=42, max_iter=1000),
    'Naive Bayes': GaussianNB()
}

# Define regressors
regressors = {
    'Linear Regression': LinearRegression(),
    'Polynomial Regression': make_pipeline(PolynomialFeatures(degree=2), LinearRegression()),
    'Random Forest Regressor': RandomForestRegressor(random_state=42),
    'SVR': SVR(),
    'K-Nearest Neighbors Regressor': KNeighborsRegressor(),
    'Neural Network Regressor': MLPRegressor(random_state=42, max_iter=1000),
    'Decision Tree Regressor': DecisionTreeRegressor(random_state=42)
}

# Define cross-validation strategy
cv = RepeatedKFold(n_splits=3, n_repeats=10, random_state=42)

# Evaluate classifiers
def evaluate_classifier(classifier, X, y):
    scores = cross_val_score(classifier, X, y, scoring='f1_weighted', cv=cv)
    return scores.mean(), scores.std()

# Evaluate regressors
def evaluate_regressor(regressor, X, y):
    mse_scores = cross_val_score(regressor, X, y, scoring='neg_mean_squared_error', cv=cv)
    r2_scores = cross_val_score(regressor, X, y, scoring='r2', cv=cv)
    return np.sqrt(-mse_scores.mean()), r2_scores.mean()

# Evaluate all classifiers
print("\nClassifier Performance:")
for name, clf in classifiers.items():
    mean, std = evaluate_classifier(clf, X_train_scaled, y_train)
    print(f"{name} - Mean F1: {mean:.3f} (+/- {std:.3f})")

# Evaluate all regressors
print("\nRegressor Performance:")
for name, reg in regressors.items():
    rmse, r2 = evaluate_regressor(reg, X_train_scaled, y_train)
    print(f"{name} - RMSE: {rmse:.3f}, R2: {r2:.3f}")

# Tune Random Forest hyperparameters using GridSearchCV
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(estimator=classifiers['Random Forest'], param_grid=param_grid, 
                           scoring='f1_weighted', cv=cv, n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)
best_classifier = grid_search.best_estimator_
print("\nBest Random Forest parameters:", grid_search.best_params_)

# Train on whole training set and evaluate on test set
best_classifier.fit(X_train_scaled, y_train)
y_pred = best_classifier.predict(X_test_scaled)
print("\nTest set performance (Classification):")
print(classification_report(y_test, y_pred, zero_division=1))

# Additional classification metrics
mcc = matthews_corrcoef(y_test, y_pred)
print(f"Matthews Correlation Coefficient: {mcc:.3f}")

# AUC-ROC calculation
y_test_binary = label_binarize(y_test, classes=np.unique(y))
y_score = best_classifier.predict_proba(X_test_scaled)
auc_roc = roc_auc_score(y_test_binary, y_score, multi_class='ovr', average='weighted')
print(f"AUC-ROC Score: {auc_roc:.3f}")

# Regression metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("\nTest set performance (Regression):")
print(f"Mean Squared Error: {mse:.3f}")
print(f"Root Mean Squared Error: {rmse:.3f}")
print(f"Mean Absolute Error: {mae:.3f}")
print(f"R-squared Score: {r2:.3f}")

# Feature importance plot
importances = best_classifier.feature_importances_
features = X.columns
plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=features)
plt.title('Feature Importance in Random Forest Model')
plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
plt.close()
print("Saved: feature_importance.png")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
plt.close()
print("Saved: confusion_matrix.png")

# Apply resampling techniques
# Try SMOTE first, if it fails, use RandomOverSampler
try:
    smote = SMOTE(random_state=42, k_neighbors=min(5, min(Counter(y_train).values()) - 1))
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    resampling_method = "SMOTE"
except ValueError:
    ros = RandomOverSampler(random_state=42)
    X_train_resampled, y_train_resampled = ros.fit_resample(X_train_scaled, y_train)
    resampling_method = "RandomOverSampler"

print(f"\nClass distribution after {resampling_method}:")
print(pd.Series(y_train_resampled).value_counts(normalize=True))

# Train on resampled dataset
pipeline = Pipeline([
    ('classifier', RandomForestClassifier(random_state=42, **grid_search.best_params_))
])

pipeline.fit(X_train_resampled, y_train_resampled)
y_pred_resampled = pipeline.predict(X_test_scaled)
print(f"\nTest set performance (model trained on {resampling_method} data):")
print(classification_report(y_test, y_pred_resampled, zero_division=1))

# Additional metrics for resampled model
mcc_resampled = matthews_corrcoef(y_test, y_pred_resampled)
print(f"Matthews Correlation Coefficient: {mcc_resampled:.3f}")

y_score_resampled = pipeline.predict_proba(X_test_scaled)
auc_roc_resampled = roc_auc_score(y_test_binary, y_score_resampled, multi_class='ovr', average='weighted')
print(f"AUC-ROC Score: {auc_roc_resampled:.3f}")

# Regression metrics for resampled model
mse_resampled = mean_squared_error(y_test, y_pred_resampled)
rmse_resampled = np.sqrt(mse_resampled)
mae_resampled = mean_absolute_error(y_test, y_pred_resampled)
r2_resampled = r2_score(y_test, y_pred_resampled)
print("\nTest set performance (Regression metrics for resampled model):")
print(f"Mean Squared Error: {mse_resampled:.3f}")
print(f"Root Mean Squared Error: {rmse_resampled:.3f}")
print(f"Mean Absolute Error: {mae_resampled:.3f}")
print(f"R-squared Score: {r2_resampled:.3f}")

# Confusion matrix for resampled model
cm_resampled = confusion_matrix(y_test, y_pred_resampled)
plt.figure(figsize=(10, 8))
sns.heatmap(cm_resampled, annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix ({resampling_method})')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig(os.path.join(output_dir, f'confusion_matrix_{resampling_method.lower()}.png'))
plt.close()
print(f"Saved: confusion_matrix_{resampling_method.lower()}.png")