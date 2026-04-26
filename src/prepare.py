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
