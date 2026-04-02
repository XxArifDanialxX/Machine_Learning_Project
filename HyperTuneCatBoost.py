import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier
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
    'iterations': [250, 500, 750, 1000],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'depth': [4, 6, 8, 10],
    'l2_leaf_reg': [1, 3, 5, 7, 9],
    'border_count': [32, 64, 128, 254],
    'bagging_temperature': [0, 1, 2, 4],
    'random_strength': [1, 2, 3, 4]
}

# Initialize CatBoostClassifier
cbc = CatBoostClassifier(verbose=0, thread_count=-1)

# Setup RandomizedSearchCV
random_search = RandomizedSearchCV(
    cbc,
    param_distributions=param_dist,
    n_iter=20,
    scoring='accuracy',
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

# Fit model with hyperparameter tuning
print("Starting hyperparameter tuning...")
random_search.fit(X_train, y_train)

# Best estimator
best_model = random_search.best_estimator_

# Predictions with best model
y_pred = best_model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"\nBest Model Accuracy: {accuracy:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix after Hyperparameter Tuning")
plt.show()

# Feature Importance Plot
importance = best_model.feature_importances_
features = X.columns
plt.figure(figsize=(10, 6))
sns.barplot(x=importance, y=features)
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Feature Importance after Hyperparameter Tuning")
plt.show()