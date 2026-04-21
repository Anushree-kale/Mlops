# ── 1. Imports ─────────────────────────────────────────────
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ── 2. Load Dataset ────────────────────────────────────────
# Make sure iris.csv is in same folder
df = pd.read_csv("iris.csv")

# Features and target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Convert labels if needed
if y.dtype == 'object':
    y = y.astype('category').cat.codes

print("Dataset loaded:", X.shape)

# ── 3. Train-Test Split ────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 4. Scaling ─────────────────────────────────────────────
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ── 5. Define Experiments ─────────────────────────────────
experiments = [
    {"n_estimators": 50, "max_depth": 3},
    {"n_estimators": 100, "max_depth": 5},
    {"n_estimators": 150, "max_depth": None},
    {"n_estimators": 200, "max_depth": 10},
]

# ── 6. Run Experiments + Versioning ───────────────────────
results = []

for i, params in enumerate(experiments):

    print(f"\nRunning Experiment {i+1}: {params}")

    model = RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        random_state=42
    )

    model.fit(X_train, y_train)

    #  Save model (versioning)
    model_name = f"model_v{i+1}.pkl"
    joblib.dump(model, model_name)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {acc:.4f}")

    # Store results
    results.append({
        "version": f"v{i+1}",
        "model_file": model_name,
        "n_estimators": params["n_estimators"],
        "max_depth": params["max_depth"],
        "accuracy": acc,
        "timestamp": datetime.now()
    })

# ── 7. Save Experiment Logs ───────────────────────────────
results_df = pd.DataFrame(results)

results_df.to_csv("experiment_runs.csv", index=False)

print("\n All experiments saved!")

# ── 8. Best Model ─────────────────────────────────────────
best = results_df.sort_values(by="accuracy", ascending=False).iloc[0]

print("\n Best Model:")
print(best)