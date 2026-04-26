import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
import mlflow
import mlflow.sklearn
import yaml

with open("params.yaml") as f:
    params = yaml.safe_load(f)["train"]

X_train = pd.read_csv("data/processed/X_train.csv")
y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
X_test = pd.read_csv("data/processed/X_test.csv")
y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

mlflow.set_experiment("iris_repro")

with mlflow.start_run():
    model = LogisticRegression(random_state=params["random_state"])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    mlflow.log_param("model", params["model"])
    mlflow.log_metric("accuracy", acc)
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    mlflow.log_artifact("model.pkl")
    print(f"Model trained. Accuracy = {acc:.4f}")
