# ==========================================
# Task 10 - Complex Relationships
# ==========================================

import warnings
warnings.filterwarnings("ignore")

# Data Handling
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt

# Train-Test Split & Hyperparameter Tuning
from sklearn.model_selection import train_test_split, GridSearchCV

# Encoding
from sklearn.preprocessing import LabelEncoder

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Evaluation Metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Partial Dependence Plot
from sklearn.inspection import PartialDependenceDisplay

# Save Model
import joblib

print("=" * 60)
print("      TASK 10 - COMPLEX RELATIONSHIPS")
print("=" * 60)

# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv("Titanic-Dataset.csv")

print("\nFirst 5 Rows")
print(df.head())

print("\nDataset Shape:", df.shape)

print("\nMissing Values")
print(df.isnull().sum())

# ==========================================
# Data Preprocessing
# ==========================================

df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1, inplace=True)

df["Age"].fillna(df["Age"].median(), inplace=True)
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

le = LabelEncoder()

df["Sex"] = le.fit_transform(df["Sex"])
df["Embarked"] = le.fit_transform(df["Embarked"])

print("\nDataset After Preprocessing")
print(df.head())

# ==========================================
# Train-Test Split
# ==========================================

X = df.drop("Survived", axis=1)
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# ==========================================
# Logistic Regression Model
# ==========================================

lr_model = LogisticRegression(max_iter=1000)

lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)

lr_accuracy = accuracy_score(y_test, y_pred_lr)
lr_precision = precision_score(y_test, y_pred_lr)
lr_recall = recall_score(y_test, y_pred_lr)
lr_f1 = f1_score(y_test, y_pred_lr)

print("\n===== Logistic Regression =====")
print(f"Accuracy : {lr_accuracy:.4f}")
print(f"Precision: {lr_precision:.4f}")
print(f"Recall   : {lr_recall:.4f}")
print(f"F1 Score : {lr_f1:.4f}")

# ==========================================
# Random Forest Model
# ==========================================

rf_model = RandomForestClassifier(random_state=42)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, y_pred_rf)
rf_precision = precision_score(y_test, y_pred_rf)
rf_recall = recall_score(y_test, y_pred_rf)
rf_f1 = f1_score(y_test, y_pred_rf)

print("\n===== Random Forest =====")
print(f"Accuracy : {rf_accuracy:.4f}")
print(f"Precision: {rf_precision:.4f}")
print(f"Recall   : {rf_recall:.4f}")
print(f"F1 Score : {rf_f1:.4f}")

# ==========================================
# Hyperparameter Tuning
# ==========================================

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

print("\nBest Parameters")
print(grid_search.best_params_)

# ==========================================
# Model Evaluation
# ==========================================

y_pred_best = best_model.predict(X_test)

best_accuracy = accuracy_score(y_test, y_pred_best)
best_precision = precision_score(y_test, y_pred_best)
best_recall = recall_score(y_test, y_pred_best)
best_f1 = f1_score(y_test, y_pred_best)

print("\n===== Tuned Random Forest =====")
print(f"Accuracy : {best_accuracy:.4f}")
print(f"Precision: {best_precision:.4f}")
print(f"Recall   : {best_recall:.4f}")
print(f"F1 Score : {best_f1:.4f}")

print("\n========== Model Comparison ==========")
print(f"{'Metric':<12}{'Logistic Regression':<22}{'Random Forest'}")
print(f"{'Accuracy':<12}{lr_accuracy:<22.4f}{best_accuracy:.4f}")
print(f"{'Precision':<12}{lr_precision:<22.4f}{best_precision:.4f}")
print(f"{'Recall':<12}{lr_recall:<22.4f}{best_recall:.4f}")
print(f"{'F1 Score':<12}{lr_f1:<22.4f}{best_f1:.4f}")

# ==========================================
# Hyperparameter Tuning
# ==========================================

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

print("\nBest Parameters")
print(grid_search.best_params_)

# ==========================================
# Evaluate Tuned Model
# ==========================================

y_pred_best = best_model.predict(X_test)

best_accuracy = accuracy_score(y_test, y_pred_best)
best_precision = precision_score(y_test, y_pred_best)
best_recall = recall_score(y_test, y_pred_best)
best_f1 = f1_score(y_test, y_pred_best)

print("\n===== Tuned Random Forest =====")
print(f"Accuracy : {best_accuracy:.4f}")
print(f"Precision: {best_precision:.4f}")
print(f"Recall   : {best_recall:.4f}")
print(f"F1 Score : {best_f1:.4f}")

print("\n========== Model Comparison ==========")
print(f"{'Metric':<12}{'Logistic Regression':<22}{'Tuned Random Forest'}")
print(f"{'Accuracy':<12}{lr_accuracy:<22.4f}{best_accuracy:.4f}")
print(f"{'Precision':<12}{lr_precision:<22.4f}{best_precision:.4f}")
print(f"{'Recall':<12}{lr_recall:<22.4f}{best_recall:.4f}")
print(f"{'F1 Score':<12}{lr_f1:<22.4f}{best_f1:.4f}")

# ==========================================
# Save Model
# ==========================================

joblib.dump(best_model, "model.pkl")

print("\nModel saved successfully!")

# ==========================================
# Save Metrics
# ==========================================

with open("metrics.txt", "w") as file:
    file.write("Logistic Regression\n")
    file.write(f"Accuracy : {lr_accuracy:.4f}\n")
    file.write(f"Precision: {lr_precision:.4f}\n")
    file.write(f"Recall   : {lr_recall:.4f}\n")
    file.write(f"F1 Score : {lr_f1:.4f}\n\n")

    file.write("Tuned Random Forest\n")
    file.write(f"Accuracy : {best_accuracy:.4f}\n")
    file.write(f"Precision: {best_precision:.4f}\n")
    file.write(f"Recall   : {best_recall:.4f}\n")
    file.write(f"F1 Score : {best_f1:.4f}\n")

print("Metrics saved successfully!")

# ==========================================
# Save Predictions
# ==========================================

predictions = X_test.copy()
predictions["Actual"] = y_test.values
predictions["Predicted"] = y_pred_best

predictions.to_csv("predictions.csv", index=False)

print("Predictions saved successfully!")

# ==========================================
# Feature Importance Plot
# ==========================================

importance = best_model.feature_importances_
features = X.columns

plt.figure(figsize=(8, 5))
plt.bar(features, importance)
plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()

print("Feature Importance Plot saved successfully!")
# ==========================================
# Partial Dependence Plot
# ==========================================

PartialDependenceDisplay.from_estimator(
    best_model,
    X_train,
    features=[0, 1]
)

plt.tight_layout()
plt.savefig("partial_dependence.png")
plt.show()

print("Partial Dependence Plot saved successfully!")

# ==========================================
# Save Experiment Log
# ==========================================

with open("experiment_log.txt", "w") as file:
    file.write("Task 10 - Complex Relationships\n")
    file.write("=" * 40 + "\n\n")
    file.write(f"Logistic Regression Accuracy : {lr_accuracy:.4f}\n")
    file.write(f"Random Forest Accuracy       : {rf_accuracy:.4f}\n")
    file.write(f"Tuned Random Forest Accuracy : {best_accuracy:.4f}\n\n")
    file.write("Best Parameters:\n")
    file.write(str(grid_search.best_params_))
    
print("Experiment log saved successfully!")
