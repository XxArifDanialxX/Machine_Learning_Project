import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_csv("D:/UIA KICT/Sem 4/Machine Learning/Dataset/Crop_recommendation.csv")

# Encode categorical labels
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["label"])

X = df.drop(columns=["label"])
y = df["label"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define parameter grid for hyperparameter tuning
param_dist = {
    'num_leaves': [20, 31, 50, 70],
    'max_depth': [6, 8, 10, -1],
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'min_data_in_leaf': [20, 50, 100],
    'feature_fraction': [0.6, 0.7, 0.8],
    'bagging_fraction': [0.6, 0.7, 0.8],
    'bagging_freq': [0, 5, 10],
    'lambda_l1': [0, 0.1, 0.5],
    'lambda_l2': [0, 0.1, 0.5]
}

# Initialize the classifier
lgb_clf = lgb.LGBMClassifier(objective='multiclass', metric='multi_logloss', n_jobs=-1, verbose=-1)

# Set up RandomizedSearchCV
random_search = RandomizedSearchCV(
    lgb_clf,
    param_distributions=param_dist,
    n_iter=20,
    scoring='accuracy',
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

# Run hyperparameter tuning
print("Starting hyperparameter tuning for LightGBM...")
random_search.fit(X_train, y_train)

# Get best estimator
best_lgb_model = random_search.best_estimator_

# Predictions
y_pred = best_lgb_model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"\nBest LightGBM Model Accuracy: {accuracy:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("LightGBM Confusion Matrix after Hyperparameter Tuning")
plt.show()

# Feature Importance Plot
importance = best_lgb_model.feature_importances_
features = X.columns
plt.figure(figsize=(10, 6))
sns.barplot(x=importance, y=features)
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("LightGBM Feature Importance after Hyperparameter Tuning")
plt.show()