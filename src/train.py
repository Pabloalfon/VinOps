#!/usr/bin/env python3
import os
import sys
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Rutas
data_path = sys.argv[1] if len(sys.argv) > 1 else os.getenv("DATA_PATH", "wine_clean.csv")
model_output = os.getenv("MODEL_OUTPUT", "/models/wine_model.pkl")

# Cargar datos
df = pd.read_csv(data_path)
X = df.drop("Class", axis=1)
y = df["Class"].astype(int)

print(f"Entrenando Random Forest con {len(df)} observaciones...")

# mtry en R = max_features en sklearn
param_grid = {
    "max_features": [2, 4, 6, 8, 10],
    "n_estimators": [500]
}

# 10-fold CV estratificada
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=2)

rf = RandomForestClassifier(random_state=2)
grid = GridSearchCV(rf, param_grid, cv=cv, scoring="accuracy", n_jobs=-1)
grid.fit(X, y)

best_model = grid.best_estimator_
y_pred = best_model.predict(X)

accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred, average="weighted")
recall = recall_score(y, y_pred, average="weighted")
f1 = f1_score(y, y_pred, average="weighted")

print(f"Mejor max_features: {grid.best_params_['max_features']}")
print(f"Accuracy CV: {grid.best_score_:.7f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precisión: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Guardar modelo
os.makedirs(os.path.dirname(model_output), exist_ok=True)
with open(model_output, "wb") as f:
    pickle.dump(grid.best_estimator_, f)

print(f"Modelo guardado en: {model_output}")