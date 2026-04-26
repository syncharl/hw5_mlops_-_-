#!/bin/bash
set -e

echo "=== Инициализация проекта ==="
git init hw5_mlops_Надежкин_Дмитрий
cd hw5_mlops_Надежкин_Дмитрий
dvc init

mkdir -p data/raw data/processed src

echo "=== Скачиваем датасет Iris ==="
curl -s -o data/raw/iris.csv https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data

dvc add data/raw/iris.csv
git add data/raw/iris.csv.dvc data/raw/.gitignore
git commit -m "Add Iris dataset via DVC"

echo "=== Cтруктура и скрипты ==="
cat > src/prepare.py << 'EOF'
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml

with open("params.yaml") as f:
    params = yaml.safe_load(f)["prepare"]

df = pd.read_csv("data/raw/iris.csv", header=None)
df.columns = ["sepal_length","sepal_width","petal_length","petal_width","target"]

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=params["test_size"], random_state=params["random_state"]
)

X_train.to_csv("data/processed/X_train.csv", index=False)
X_test.to_csv("data/processed/X_test.csv", index=False)
y_train.to_csv("data/processed/y_train.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)
print("Data prepared")
EOF

cat > src/train.py << 'EOF'
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
EOF

cat > params.yaml << EOF
prepare:
  test_size: 0.2
  random_state: 42
train:
  model: LogisticRegression
  random_state: 42
EOF

cat > dvc.yaml << EOF
stages:
  prepare:
    cmd: python src/prepare.py
    deps:
    - src/prepare.py
    - data/raw/iris.csv
    outs:
    - data/processed/X_train.csv
    - data/processed/X_test.csv
    - data/processed/y_train.csv
    - data/processed/y_test.csv
    params:
    - params.yaml:prepare
  train:
    cmd: python src/train.py
    deps:
    - src/train.py
    - data/processed/X_train.csv
    - data/processed/X_test.csv
    - data/processed/y_train.csv
    - data/processed/y_test.csv
    outs:
    - model.pkl
    params:
    - params.yaml:train
EOF

cat > requirements.txt << EOF
pandas
scikit-learn
mlflow
dvc
pyyaml
EOF

pip install -r requirements.txt -q

echo "=== Проверка рабочей директории ==="
pwd
rm -f params.yaml   # на случай остатка от прошлого запуска

cat > params.yaml << 'EOF'
prepare:
  test_size: 0.2
  random_state: 42
train:
  model: LogisticRegression
  random_state: 42
EOF

echo "=== Содержимое params.yaml ==="
cat params.yaml

echo "=== Запуск пайплайна ==="
dvc repro --force

echo "=== MLflow UI ==="
echo "mlflow ui --backend-store-uri sqlite:///mlflow.db"
echo "Проект полностью воспроизводим: git clone + dvc pull + dvc repro"