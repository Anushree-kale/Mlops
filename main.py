# Iris Dataset - Model Training
# Uses scikit-learn's built-in Iris dataset (same as Kaggle's iris.csv)

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ── 1. Load Data ──────────────────────────────────────────────────────────────
# Option A: Load from sklearn (identical to Kaggle's iris.csv)
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name="species")

# Option B: Load from Kaggle CSV (uncomment if you have the file)
# df = pd.read_csv("iris.csv")
# X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]d]
# y = df["species"].map({"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2})

print("Dataset shape:", X.shape)
print("\nFirst 5 rows:")
print(X.head())
print("\nClass distribution:")
print(y.value_counts())

# ── 2. Train / Test Split ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")

# ── 3. Feature Scaling ────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── 4. Train Model ────────────────────────────────────────────────────────────
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)
print("\nModel training complete!")

# ── 5. Evaluate ───────────────────────────────────────────────────────────────
y_pred = model.predict(X_test_scaled)

print(f"\nAccuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ── 6. Feature Importance ─────────────────────────────────────────────────────
print("\nFeature Importances:")
for name, importance in zip(iris.feature_names, model.feature_importances_):
    print(f"  {name:<30} {importance:.4f}")

# ── 7. Predict on New Sample ──────────────────────────────────────────────────
sample = np.array([[5.1, 3.5, 1.4, 0.2]])  # Known Setosa
sample_scaled = scaler.transform(sample)
prediction = model.predict(sample_scaled)
print(f"\nSample prediction: {iris.target_names[prediction[0]]}")