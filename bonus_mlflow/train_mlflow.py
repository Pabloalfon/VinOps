import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

import os

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

tracking_uri = "http://mlflow:5000"
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("DISIA Bonus MLflow")

data_path = os.getenv("DATA_PATH", "wine_clean.csv")

df = pd.read_csv(data_path)
X = df.drop("Class", axis=1)
y = df["Class"].astype(int)

param_grid = {
    "max_features": [2, 4, 6, 8, 10],
    "n_estimators": [500]
}

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=2)

model = RandomForestClassifier(random_state=2)

grid = GridSearchCV(
    model,
    param_grid,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1
)

grid.fit(X, y)

best_model = grid.best_estimator_
y_pred = best_model.predict(X)

accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred, average="weighted")
recall = recall_score(y, y_pred, average="weighted")
f1 = f1_score(y, y_pred, average="weighted")

signature = infer_signature(X, best_model.predict(X))

with mlflow.start_run(run_name="wine_random_forest_bonus"):
    mlflow.log_params(grid.best_params_)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision_weighted", precision)
    mlflow.log_metric("recall_weighted", recall)
    mlflow.log_metric("f1_weighted", f1)

    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="wine_model",
        signature=signature,
        input_example=X.iloc[:5]
    )

    mlflow.set_tag("curso", "DISIA")
    mlflow.set_tag("bonus", "MLflow experiment tracking")
    mlflow.set_tag("dataset", "wine_clean.csv")

print("Run registrada correctamente en MLflow")
print(f"Mejor max_features: {grid.best_params_['max_features']}")
print(f"Accuracy CV: {grid.best_score_:.7f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precisión: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")