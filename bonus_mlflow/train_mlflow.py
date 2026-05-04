import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

tracking_uri = "http://mlflow:5000"
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("DISIA Bonus MLflow")

data_path = os.getenv("DATA_PATH", "wine_clean.csv")

df = pd.read_csv(data_path)
X = df.drop("Class", axis=1)
y = df["Class"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

params = {
    "solver": "lbfgs",
    "max_iter": 1000,
    "multi_class": "auto",
    "random_state": 42
}

model = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(**params)),
    ]
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")

signature = infer_signature(X_train, model.predict(X_train))

with mlflow.start_run(run_name="wine_logreg_bonus"):
    mlflow.log_params(params)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision_weighted", precision)
    mlflow.log_metric("recall_weighted", recall)
    mlflow.log_metric("f1_weighted", f1)

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="wine_model",
        signature=signature,
        input_example=X_train.iloc[:5]
    )

    mlflow.set_tag("curso", "DISIA")
    mlflow.set_tag("bonus", "MLflow experiment tracking")
    mlflow.set_tag("dataset", "wine_clean.csv")

print("Run registrada correctamente en MLflow")
print(f"Accuracy: {accuracy:.4f}")